import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm, weight_norm


class Discriminator(nn.Module):
    def __init__(self, ind, in_chanels):
        """
        Args:
            p - period for each descr
        """
        super().__init__()
        norm = spectral_norm if ind == 0 else weight_norm

        self.avgpool = nn.Sequential()
        for _ in range(ind):
            self.avgpool.append(nn.AvgPool1d(4, 2, 1))

        self.body = nn.ModuleList()
        self.body.append(
            nn.Sequential(
                norm(
                    nn.Conv1d(
                        in_chanels, 2 ** (5 + 1), kernel_size=5, stride=3, padding=2
                    )
                ),
                nn.LeakyReLU(),
            )
        )
        for l_ind in range(2, 5):
            self.body.append(
                nn.Sequential(
                    norm(
                        nn.Conv1d(
                            2 ** (5 + l_ind - 1),
                            2 ** (5 + l_ind),
                            kernel_size=5,
                            stride=3,
                            groups=8,
                            padding=2,
                        )
                    ),
                    nn.LeakyReLU(),
                )
            )

        self.body.append(
            nn.Sequential(
                norm(nn.Conv1d(2 ** (9), 1024, kernel_size=5, groups=8, padding=2)),
                nn.LeakyReLU(),
            )
        )

        self.end = norm(nn.Conv1d(1024, 1, kernel_size=3, padding=1))

    def forward(self, x):
        """
        Model forward method.

        Args:
        Returns:
            output (dict): output dict containing logits.
        """
        intermediate_feat = []
        x = self.avgpool(x)

        for block in self.body:
            x = block(x)
            intermediate_feat.append(x)

        x = self.end(x)
        intermediate_feat.append(x)
        return x, intermediate_feat

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


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_chanels):
        """
        Args:
        """
        super().__init__()

        self.discriminators = nn.ModuleList()
        for i in range(3):
            self.discriminators.append(Discriminator(i, in_chanels))

    def forward(self, mel_spect, **batch):
        """
        Model forward method.

        Args:
            mel_spect (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        result = []
        intermediate_feat = []
        for disc in self.discriminators:
            curr_res, curr_interm_feat = disc(mel_spect)
            result.append(curr_res)
            intermediate_feat.append(curr_interm_feat)
        return result, intermediate_feat

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
