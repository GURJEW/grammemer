from torch.nn import (
    AdaptiveMaxPool1d,
    Embedding,
    Conv1d,
    LeakyReLU,
    Module,
    ModuleList,
    Sequential,
    Sigmoid
)


class Block(Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = Sequential(
            Conv1d(
                in_channels=channels,
                out_channels=channels//2,
                kernel_size=3,
                padding=1
            ),
            Conv1d(
                in_channels=channels//2,
                out_channels=channels,
                kernel_size=1
            )
        )
        self.activation = LeakyReLU()

    def forward(self, x):
        return self.activation(self.conv(x)) + x


class Network(Module):

    def __init__(self, setting):
        super().__init__()
        self.embedding = Embedding(
            num_embeddings=setting.inputs,
            embedding_dim=setting.embedding,
            padding_idx=0
        )
        self.letter_convolution = Sequential(
            *   [
                Block(setting.embedding)
                for _ in range(setting.depth)
                ]
        )
        self.maxpooling = AdaptiveMaxPool1d(1)
        self.word_convolution = Sequential(
            *   [
                Block(setting.embedding)
                for _ in range(setting.depth)
                ]
        )
        self.conv = Conv1d(
            in_channels=setting.embedding,
            out_channels=setting.outputs,
            kernel_size=3
        )
        self.activation = Sigmoid()
        self.shape = [-1, setting.sentence + 2, setting.embedding]

    def forward(self, x):
        x = self.embedding(x.flatten(0, 1)).permute(0, 2, 1)
        x = self.letter_convolution(x)
        x = self.maxpooling(x).view(*self.shape).permute(0, 2, 1)
        x = self.word_convolution(x)
        x = self.conv(x).permute(0, 2, 1)
        return self.activation(x)

'''
EOF
'''