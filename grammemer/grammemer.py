import os
import pandas as pd
import torch as pth

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR

from IPython.display import clear_output

from grammemer.data import Data, Loader
from grammemer.network import Network
from grammemer.tools import (
    History, Loss, Metrics, ProgressBar, Vocabulary, read_tsv
    )


device = 'cuda' if pth.cuda.is_available() else 'cpu'
grammemes = read_tsv('data/grammemes.tsv')


class Grammemer():

    def __init__(self, **setting):
        super().__init__()
        self.setting = pd.Series(
            {
                'depth': 5,
                'embedding': 128,
                'sentence': 30,
                'word': 30
            },
            dtype=int
        )
        self.setting.update(setting)
        self.name = ''.join(
            [i[0] + str(self.setting[i]) for i in self.setting.index]
            )
        self.letters = Vocabulary(
            data=[chr(ord('а')+i) for i in range(32)]+['ё', '-'],
            name='letters',
            index_start=1
            )
        self.grammemes = Vocabulary(
            data=grammemes.grammeme.to_list(),
            name='grammemes',
            update=pd.Series({'gen2': 10, 'acc2': 12, 'loc2': 14})
            )
        self.descriptions = Vocabulary(
            data=grammemes.description.to_list(),
            name='descriptions',
            index=grammemes.grammeme.to_list()
            )
        self.setting['inputs'] = len(self.letters) + 1
        self.setting['outputs'] = len(self.grammemes)
        self.data = Data(self.setting, self.letters, self.grammemes)
        self.network = Network(self.setting).to(device)
        self.history = None
        self.loss = Loss()
        self.metrics = Metrics()
        
    def reset(self):
        self.network = Network(self.setting).to(device)

    def save(self, suffix=''):
        if self.name not in os.listdir('data'):
            os.mkdir(f'data/{self.name}')
        pth.save(self.network, f'data/{self.name}/network{suffix}.pth')
        self.history.data.to_csv(
            f'data/{self.name}/history{suffix}.tsv',
            sep='\t'
            )
    
    def load(self, suffix=''):
        self.network = pth.load(f'data/{self.name}/network{suffix}.pth')
        self.history = History(
            data=read_tsv(f'data/{self.name}/history{suffix}.tsv')
            )

    def fit(
        self,
        epochs=1,
        batch_size=64,
        learning_rate=0.001,
        gamma=0.99,
        patience=10,
        verbose=True
        ):
        optimizer = Adam(self.network.parameters(), learning_rate)
        scheduler = {
            'R': ReduceLROnPlateau(
                optimizer,
                patience=patience
                ),
            'E': ExponentialLR(
                optimizer,
                gamma=gamma
                )
            }
        validation = False if self.data.val is None else True
        if self.history is None:
            self.history = History()
        start_epoch = len(self.history) // (validation + 1)
        for epoch in range(start_epoch, start_epoch + epochs):
            if verbose:
                progress = ProgressBar(
                    f'Epoch: [{epoch + 1}/{start_epoch + epochs}]',
                    self.data.count_batchs(batch_size)
                    )
            for train, (x, y) in enumerate(Loader(self.data.train, batch_size)):
                self.network.train()
                optimizer.zero_grad()
                p = self.network(x.to(device))
                y = y.to(device)
                loss = self.loss(p, y)
                self.metrics(p.round(), y)
                loss.backward(retain_graph=True) 
                optimizer.step()
                if verbose:
                    progress.step(train)
            self.history.add_record(
                pd.concat([self.metrics.values(), self.loss.values()]),
                epoch=epoch+1,
                stage='training',
                learning_rate=optimizer.param_groups[0]['lr'],
                batch_size=batch_size
                )
            if validation:
                for val, (x, y) in enumerate(Loader(self.data.val, batch_size)):
                    self.network.eval()
                    with pth.no_grad():
                        p = self.network(x.to(device))
                        y = y.to(device)
                        self.loss(p, y)
                        self.metrics(p.round(), y)
                    if verbose:
                        progress.step(train+val)
                self.history.add_record(
                    pd.concat([self.metrics.values(), self.loss.values()]),
                    stage='validation'
                    )
            scheduler['R'].step(loss)
            scheduler['E'].step()
            
            clear_output()
            self.history.plot(patience)
            print('\n', self.history.data.iloc[-2:], end='\n\n')

    def evaluate(self, batch_size=32, verbose=True):
        if verbose:
            progress = ProgressBar(
                f'Progress: [evaluate]',
                self.data.test.count_batchs(batch_size)
                )
        for test, (x, y) in enumerate(Loader(self.data.test, batch_size)):
            self.network.eval()
            with pth.no_grad():
                p = self.network(x.to(device))
                y = y.to(device)
                self.loss(p, y)
                self.metrics(p.round(), y)
            if verbose:
                progress.step(test)
        print()
        return pd.concat([self.metrics.values(), self.loss.values()])

    def infer(self, x, p, path):
        result = ''
        for sentence, (words, tags) in enumerate(zip(x, p)):
            header = f'\n ** SENTENCE {sentence+1} **\n\n'
            if path is None:
                print(header, end='')
            else:
                result += header
            for word, tag in zip(words[1:], tags):
                word = ''.join(
                    self.letters[word[word.bool()].numpy()].to_list()
                    ).upper()
                tag = self.grammemes[tag.bool().numpy()]
                tag = self.descriptions[tag].to_list()
                description = f'{word} {tag}\n'
                if len(word):
                    if path is None:
                        print(description, end='')
                    else:
                        result += description
        if path is not None:
            with open(path, 'w') as output:
                output.write(result)

    def __call__(self, x, path=None):
        if isinstance(x, str):
            if x[-4:] == '.txt':
                with open(x, 'r') as txt:
                    x = txt.read()
            sentences = [
                self.data.tokenize.sentence_to_words(sentence)
                for sentence in self.data.tokenize.paragraph_to_sentences(x)
                ]
            size = len(sentences)
            x = pth.zeros(
                [size, self.setting.sentence + 2, self.setting.word + 2]
                ).long()
            for s, words in enumerate(sentences):
                for w, word in enumerate(map(self.letters, words)):
                    x[s][w+1][1:len(word)+1] = pth.tensor(word).long()
        p = self.network(x.to(device)).round().cpu()
        self.infer(x, p, path)



'''
EOF
'''