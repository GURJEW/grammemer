import sys
import pandas as pd
import torch as pth

from google.colab import output
from torch.nn import BCELoss

from plotly import graph_objs as go
from plotly.subplots import make_subplots

from time import monotonic


class Accumulator(pd.Series):

    def __init__(self, index):
        super().__init__(data=[0]*len(index), index=index)
        self.counter = 0

    def reset(self):        
        self *= 0
        self.counter = 0

    def add(self, values):
        self += values
        self.counter +=1

    def __call__(self):
        values = self / self.counter
        self.reset()
        return values


class History():

    def __init__(
        self,
        data=None,
        parameters=['stage', 'learning_rate', 'epoch', 'batch_size'],
        metrics=['precision', 'recall', 'f1_score'],
        losses=['loss']
        ):
        self.parameters = parameters
        self.losses = losses
        self.metrics = metrics
        self.data = pd.DataFrame(data=data, columns=parameters+metrics+losses)

    def add_record(self, record, **params):
        for key, value in params.items():
            record.loc[key] = value
        self.data.loc[len(self), record.index] = record
        self.data.fillna(method='ffill', inplace=True)

    def __len__(self):
        return len(self.data)

    def plot(self, tail):
        history = self.data.groupby(['stage', 'epoch']).first()
        fig = make_subplots(
            rows=4,
            cols=2,
            shared_xaxes=True,
            column_widths=(6, 4),
            vertical_spacing=0.02,
            horizontal_spacing=0.1
            )
        for stage in ['training', 'validation']:
            for i, score in enumerate(self.losses+self.metrics):
                fig.add_trace(
                    go.Scatter(
                        x=history.loc[stage].index,
                        y=history.loc[stage, score],
                        mode='lines',
                        name=f'{stage} {score}'
                    ),
                    row=i+1,
                    col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=history.loc[stage].index[-tail:],
                        y=history.loc[stage, score].iloc[-tail:],
                        mode='lines',
                        name=f'{stage} {score}'
                    ),
                    row=i+1,
                    col=2
                )
                fig.update_yaxes(title=score, row=i+1, col=1)
        fig.update_layout(
            # showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            width=800,
            height=500
            )
        fig.update_xaxes(title='epoch', row=4)
        fig.update_xaxes(
            showline=True,
            linewidth=1,
            linecolor='DarkBlue',
            mirror=True,
            showgrid=False
            )
        fig.update_yaxes(
            showline=True,
            linewidth=1,
            linecolor='DarkBlue',
            mirror=True,
            showgrid=False
            )
        fig.show()


class Metrics():

    def __init__(self):
        self.values = Accumulator(['precision', 'recall', 'f1_score'])

    @staticmethod
    def get_item(scalar):
        return scalar.cpu().item()

    def __call__(self, p, y):
        fn = ((1  - p) * y).sum()
        tp = (p * y).sum()
        fp = (p * (1 - y)).sum()
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1_score = 2 * precision * recall / (precision + recall + 1e-7)
        self.values.add(list(map(self.get_item, [precision, recall, f1_score])))


class Loss():

    def __init__(self):
        self.values = Accumulator(['loss'])
    
    def __call__(self, p, y):
        loss = BCELoss()(p, y)
        self.values.add([loss.cpu().item()])
        return loss


class ProgressBar():

    def __init__(self, title, size, tags='progress', bar_size=64):
        self.title = title
        self.size = size
        self.tags = tags
        self.time = None
        self.bar = (pd.Series(range(size+1)) / size * bar_size).round().diff()
        self.start()
        
    def start(self):
        if self.time is None:
            self.time = monotonic()
        sys.stdout.write(self.title)
        with output.use_tags(self.tags):
            sys.stdout.write(f'[0.00%]{chr(0x058D)}[0s]')
    
    def step(self, number):
        number += 1
        progress = round(100 * number / self.size, 2)
        m, s = divmod(round(monotonic() - self.time), 60)
        h, m = divmod(m, 60)
        output.clear(output_tags=self.tags)
        sys.stdout.write('#' * int(self.bar[number]))
        sys.stdout.flush()
        with output.use_tags(self.tags):
            sys.stdout.write(f'[{progress}%]{chr(0x058D)}[{h}:{m}:{s}]')
            sys.stdout.flush()


class Vocabulary(pd.Series):
    '''
    The class for encoding and decoding tokens:
        - callable for encoding;
        - indexing for decoding.
    '''

    def __init__(self, data, name, index_start=0, index=None, update=None):
        if index is None:
            index = range(index_start, index_start + len(data))
        super().__init__(data, index=index, name=name)
        self._encoder = self._inverse()
        if update is not None:
            self._encoder = pd.concat([self._encoder, update])
        self.name = name

    def _inverse(self):
        return self.reset_index(name=self.name).set_index(self.name)['index']

    def __call__(self, tokens):
        return self._encoder[list(tokens)]

    def __str__(self):
        return f'{name} vocabulary (size={len(self)})'


def read_tsv(path):
    columns = ['sentence', 'tag', 'words', 'tags', 'parts']
    converters = {column: eval for column in columns}
    return pd.read_csv(path, index_col=0, sep='\t', converters=converters)


'''
EOF
'''