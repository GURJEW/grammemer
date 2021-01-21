import os
import re
import sys

import pandas as pd
import torch as pth

from google.colab import output
from torch.utils import data
from zipfile import ZipFile

from grammemer.tools import ProgressBar, read_tsv


class Data():

    '''
    DATA
    '''

    from grammemer.property import x, y

    __slots__ = (
        '_x', '_y', 'train', 'val', 'test',
        'tokenize', 'tag', 'read', 'setting',
        'letters', 'grammemes'
        )

    def __init__(self, setting, letters, grammemes):
        self._x = None
        self._y = None
        self.train = Dataset()
        self.val = Dataset()
        self.test = Dataset()
        self.read = Reader(setting)
        self.tokenize = Tokenizer()
        self.tag = Tagger()
        self.setting = setting
        self.letters = letters
        self.grammemes = grammemes

    def count_batchs(self, batch_size):
        return (
            self.train.count_batchs(batch_size)
            + self.val.count_batchs(batch_size)
            )

    def vectorize(self, del_source=True, verbose=True):
        size = len(self.read.frame)
        if verbose:
            progress = ProgressBar(
                title='Progress [vectorizer]: ',
                size=size
                )
        x_shape = [size, self.setting.sentence + 2, self.setting.word + 2]
        y_shape = [size, self.setting.sentence, len(self.grammemes)]
        self.x = pth.zeros(x_shape).long()
        self.y = pth.zeros(y_shape)
        for s, words, tags in self.read.frame.reset_index().values:
            taged_words = zip(
                map(self.letters, words),
                map(self.grammemes, tags)
                )
            for w, (word, tag) in enumerate(taged_words):
                self.x[s][w+1][1:len(word)+1] = pth.tensor(word).long()
                self.y[s][w][tag] += 1
            if verbose:
                progress.step(s)
        self.train = Dataset(self.x, self.y)
        print('\nВекторизация данных завершена.')
        if del_source:
            del self.read.frame
        
    def split(self, *ratio, del_source=True, shuffle=True):
        if shuffle:
            indices = pth.randperm(len(self.x))
            self.x = self.x[indices]
            self.y = self.y[indices]
        ratio = pth.tensor(ratio)
        ratio = (ratio * len(self.x) // ratio.sum()).int()
        parts = zip(
            ['train', 'val', 'test'],
            self.x[:ratio.sum()].split(tuple(ratio)),
            self.y[:ratio.sum()].split(tuple(ratio))
        )
        for name, x, y in parts:
            setattr(self, name, Dataset(x.long(), y))
            print(f'Набор {name}: {len(getattr(self, name, ""))} примеров.')
        if del_source:
            del self.x
            del self.y

    def load(self, suffix=''):
        self.x = pth.load(f'data/x{suffix}.pth')
        self.y = pth.load(f'data/y{suffix}.pth')

    def save(self, suffix=''):
        self.x = pth.save(self.x, f'data/x{suffix}.pth')
        self.y = pth.save(self.y, f'data/y{suffix}.pth')

    def write(self, suffix=''):
        self.read.frame.to_csv(f'data/frame{suffix}.tsv', sep='\t')


class Dataset(data.Dataset):

    '''
    DATASET
    '''

    from grammemer.property import x, y

    __slots__ = '_x', '_y'

    def __init__(self, x=None, y=None):
        super().__init__()
        self._x = x
        self._y = y

    def __len__(self):
        if self._x is None:
            return 0 
        return len(self._x)
    
    def __getitem__(self, index):
        return self._x[index], self._y[index]
    
    def count_batchs(self, batch_size):
        return len(self) // batch_size


class Loader(data.DataLoader):

    def __init__(self, dataset, batch_size):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True            
            )


class Reader():

    from grammemer.property import frame

    __slots__ = '_frame', '_progress', 'setting'

    def __init__(self, setting):
        self.setting = setting
        self._frame = None

    def _link(self, number, name, doc_dir, tag_dir):
        tagged_doc = pd.concat(
            [
            read_tsv(f'{doc_dir}/{name}'),
            read_tsv(f'{tag_dir}/{name}')
            ],
            axis=1
        )
        tagged_doc.columns = ['words', 'tags']
        self._progress.step(number)
        return tagged_doc

    def _cut_outliers(self, frame):
        length = frame.words.apply(len)
        sentence_mask = (length > 1) & (length <= self.setting.sentence)
        word_mask = frame.loc[sentence_mask, 'words'].apply(
            lambda words: max(map(len, words))
            ) <= self.setting.word
        return frame.loc[sentence_mask].loc[word_mask].reset_index(drop=True)

    def __call__(
            self,
            path=None,
            doc_dir='data/docs',
            tag_dir='data/tags'
            ):
        if path is None:
            names = os.listdir(tag_dir)
            self._progress = ProgressBar('Progress [reader]: ', len(names))
            self.frame = self._cut_outliers(
                pd.concat(
                        [
                        self._link(*name, doc_dir=doc_dir, tag_dir=tag_dir)
                        for name in enumerate(names)
                        ]
                    )
                )
        else:
            self.frame = self._cut_outliers(read_tsv(path))


class Tagger():
    '''
    Tagger based on the probability estimation of a sequence of grammemes.
    '''

    class Node():
        '''
        Homonymy removal node. Attributes:
            tag - the tag of last parsed word
            score - probability score of the tag
            path - sequence of previous word tags
        '''

        def __init__(self, tag, score, path=[]):
            self.tag = tag
            self.score = score
            self.path = path


    def __init__(self):
        self.parsed_words = {}

    @property
    def parsed_files(self):
        return os.listdir(self.tag_dir)   

    @property
    def file_names(self):
        return os.listdir(self.doc_dir)   

    def parse(self, word):
        if word not in self.parsed_words:
            self.parsed_words[word] = self.analyzer.parse(word)
        return self.parsed_words[word]

    def get_node(self, version):
        tag = list(
            set(re.split(r'[, ]', str(version.tag)))
            & set(self.occurrences.index)
            )
        return Tagger.Node(tag, version.score)

    def get_score(self, left, right):
        return (
            self.occurrences.loc[left.tag, right.tag].min().min()
            * left.score * right.score
            )

    def get_max_score(self, lefts, right):
        max_score = 0
        index = 0
        for current, left in enumerate(lefts):
            score = self.get_score(left, right)
            if score >= max_score:
                max_score = score
                index = current
        return max_score, index

    def get_tags(self, word_list):
        lefts = [Tagger.Node(['START'], 1)]
        for word in word_list:
            rights = [self.get_node(version) for version in self.parse(word)]
            for right in rights:
                right.score, index = self.get_max_score(lefts, right)
                right.path = lefts[index].path + [lefts[index].tag]
            lefts = rights
        _, index = self.get_max_score(lefts, Tagger.Node(['END'], 1))
        return lefts[index].path[1:] + [lefts[index].tag]

    def __call__(
        self,
        doc_dir='data/docs',
        tag_dir='data/tags',
        occurrences='data/occurrences.csv'
        ):
        from pymorphy2 import MorphAnalyzer


        self.analyzer = MorphAnalyzer()
        self.occurrences = pd.read_csv(
            filepath_or_buffer=occurrences,
            index_col=[0, 1]
            ).unstack().occurrences
        
        self.doc_dir = doc_dir
        self.tag_dir = tag_dir
        try:
            os.makedirs(tag_dir)
        except:
            pass
        progress = ProgressBar('Progress [tagger]: ', len(self.file_names))
        for number, file_name in enumerate(self.file_names):
            if file_name not in self.parsed_files:
                (
                read_tsv(f'{self.doc_dir}/{file_name}')
                .sentence.apply(self.get_tags)
                .to_frame(name='tag')
                .to_csv(f'{self.tag_dir}/{file_name}', sep='\t')
                )
            progress.step(number)


class Tokenizer():
        
    @property
    def parsed_list(self):
        return os.listdir(self.doc_dir)    

    @staticmethod
    def fb2_to_paragraphs(fb2):
        return re.findall(
            r'<p>.+[.!?]</p>',
            fb2.read().decode().replace('\xa0', ' ')
            )

    @staticmethod
    def paragraph_to_sentences(paragraph):
        return pd.Series(re.split(r'[.!?…] ', paragraph.lower()))

    @staticmethod
    def sentence_to_words(sentence):
        return re.findall(r'[а-яё-]+', sentence)

    def _parse_fb2(self, fb2):
        return pd.concat(
            list(map(self.paragraph_to_sentences, self.fb2_to_paragraphs(fb2))),
            ignore_index=True
            ).apply(self.sentence_to_words).to_frame(name='sentence')

    def __call__(self, zip_file, doc_dir='data/docs'):
        self.doc_dir = doc_dir
        try:
            os.makedirs(doc_dir)
        except:
            pass
        with ZipFile(zip_file) as zf:
            name_list = zf.namelist()
            progress = ProgressBar('Progress [tokenizer]: ', len(name_list))
            for number, input_name in enumerate(name_list):
                output_name = input_name[:-3] + 'tsv'
                if output_name not in self.parsed_list:
                    with zf.open(input_name) as fb2:
                        self._parse_fb2(fb2).to_csv(
                            f'{self.doc_dir}/{output_name}',
                            sep='\t'
                            )
                progress.step(number)






'''
EOF
'''