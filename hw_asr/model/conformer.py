import torch
from torch import nn

from hw_asr.base import BaseModel


class Convolution(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0

        self.norm = nn.LayerNorm(input_dim)
        self.seq = nn.Sequential(
            nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
            ),
            nn.GLU(1),
            nn.Conv1d(
                num_channels,
                num_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                groups=num_channels,
            ),
            nn.BatchNorm1d(num_channels),
            nn.SiLU(),
            nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.norm(input)
        x = x.transpose(0, 1)
        x = x.transpose(1, 2)
        x = self.seq(x)
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)
        return x


class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sequential(input)


class ConformerLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.ffn1 = FFN(input_dim, ffn_dim, dropout)

        self.attention = nn.MultiheadAttention(input_dim, num_attention_heads, dropout)
        self.attention_norm = nn.LayerNorm(input_dim)
        self.attention_dropout = nn.Dropout(dropout)

        self.conv = Convolution(
            input_dim=input_dim,
            num_channels=input_dim,
            kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )

        self.ffn2 = FFN(input_dim, ffn_dim, dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self, input: torch.Tensor, key_padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        x = self.ffn1(input) / 2 + x

        skip_connection = x
        x = self.attention_norm(x)
        x, _ = self.attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.attention_dropout(x) + skip_connection

        x = self.conv(x) + x
        x = self.norm(self.ffn2(x) / 2 + x)

        return x


class Conformer(BaseModel):
    def __init__(
        self,
        input_dim: int,
        n_class: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        **batch,
    ):
        super().__init__(input_dim, n_class, **batch)

        self.conformer_layers = nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    kernel_size,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.logits_layer = nn.Linear(in_features=ffn_dim, out_features=n_class)

    def forward(self, spectrogram, **batch) -> dict[str, torch.Tensor]:
        max_length = int(torch.max(batch["spectrogram_length"]).item())
        mask = torch.arange(max_length).expand(
            batch["spectrogram_length"].size(0), max_length
        ).to(batch["spectrogram_length"].device) >= batch[
            "spectrogram_length"
        ].unsqueeze(
            1
        )

        x = spectrogram.transpose(1, 2)
        for layer in self.conformer_layers:
            x = layer(x, mask.T)

        return {"logits": self.logits_layer(x)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
