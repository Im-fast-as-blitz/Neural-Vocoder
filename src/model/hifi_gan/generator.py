import math
import random

from torch import nn
from torch.nn import Sequential


class ResBlock(nn.Module):
    def __init__(self, in_chanels, Dr_n, Kr_n):
        super().__init__()
        self.blocks = nn.ModuleList()
        for m in range(len(Dr_n)):
            curr_block = nn.Sequential()
            for l_ind in range(len(Dr_n[m])):
                curr_block.append(nn.LeakyReLU())
                curr_block.append(
                    nn.Conv1d(
                        in_chanels,
                        in_chanels,
                        kernel_size=Kr_n,
                        dilation=Dr_n[m][l_ind],
                        padding="same",
                    )
                )
            self.blocks.append(curr_block)

    def forward(self, x):
        for block in self.blocks:
            skip_con = x
            x = block(x)
            x = skip_con + x
        return x

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class MRF(nn.Module):
    def __init__(self, in_chanels, Kr, Dr):
        super().__init__()
        self.blocks = nn.ModuleList()
        for n in range(len(Kr)):
            self.blocks.append(ResBlock(in_chanels, Dr[n], Kr[n]))

    def forward(self, x):
        result = 0
        for block in self.blocks:
            result = result + block(x)
        return result

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class Generator(nn.Module):
    def __init__(self, in_chanels, Hu, Ku, Kr, Dr):
        """
        Args:
            Hu - hidden dimension of the transposed convolutions
            Ku - kernel sizes of the transposed convolutions
            Kr - kernel sizes of MRF modules
            Dr - dilation rates of MRF modules
        """
        super().__init__()

        self.init_lay = nn.Conv1d(in_chanels, Hu, kernel_size=7, dilation=1, padding=3)

        self.generator = nn.Sequential()
        for l_ind in range(len(Ku)):
            self.generator.append(nn.LeakyReLU())
            self.generator.append(
                nn.ConvTranspose1d(
                    Hu // (2**l_ind),
                    Hu // (2 ** (l_ind + 1)),
                    kernel_size=Ku[l_ind],
                    stride=Ku[l_ind] // 2,
                    padding=Ku[l_ind] // 4,
                )
            )
            self.generator.append(MRF(Hu // (2 ** (l_ind + 1)), Kr, Dr))

        self.end = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(Hu // (2 ** len(Ku)), 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel_spect, **batch):
        mel_spect = self.init_lay(mel_spect)
        result = self.generator(mel_spect)
        result = self.end(result)
        return result

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
