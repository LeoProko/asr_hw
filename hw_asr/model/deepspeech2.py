import torch
from torch import nn

from hw_asr.base import BaseModel


class DeepSpeech2(BaseModel):
    def __init__(
        self, n_class: int, n_feats: int, n_rnn: int, hidden_dim: int, **batch
    ):
        super().__init__(n_feats, n_class, **batch)

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        )

        self.rnn_block = nn.GRU(
            n_feats * 32 // 4,
            hidden_dim,
            num_layers=n_rnn,
            batch_first=True,
            bidirectional=True,
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(
                hidden_dim * 2,
                hidden_dim * 2,
                21,
                groups=hidden_dim * 2,
                padding=10,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim * 2),
        )

        self.logits_layer = nn.Linear(hidden_dim * 2, n_class)

    def forward(self, spectrogram: torch.Tensor, **batch):
        # spectrogram: (B, input_dim, L)

        x = self.conv_block_1(spectrogram.unsqueeze(1))
        x = x.permute(0, 3, 1, 2)
        x, _ = self.rnn_block(x.view(x.shape[0], x.shape[1], -1))
        x = self.conv_block_2(x.transpose(1, 2)).transpose(1, 2)
        x = self.logits_layer(x)

        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths // 2
