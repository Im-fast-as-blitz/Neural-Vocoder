import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


class Discriminator(nn.Module):
    def __init__(self, in_chanels, p):
        """
        Args:
            p - period for each descr
        """
        super().__init__()
        self.p = p

        self.body = nn.ModuleList()
        self.body.append(
            nn.Sequential(
                weight_norm(
                    nn.Conv2d(
                        in_chanels,
                        2 ** (5 + 1),
                        kernel_size=(5, 1),
                        stride=(3, 1),
                        padding=(2, 0),
                    )
                ),
                nn.LeakyReLU(),
            )
        )
        for l_ind in range(2, 5):
            self.body.append(
                nn.Sequential(
                    weight_norm(
                        nn.Conv2d(
                            2 ** (5 + l_ind - 1),
                            2 ** (5 + l_ind),
                            kernel_size=(5, 1),
                            stride=(3, 1),
                            padding=(2, 0),
                        )
                    ),
                    nn.LeakyReLU(),
                )
            )

        self.body.append(
            nn.Sequential(
                weight_norm(
                    nn.Conv2d(2 ** (9), 1024, kernel_size=(5, 1), padding=(2, 0))
                ),
                nn.LeakyReLU(),
            )
        )

        self.end = weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, mel_spec):
        """
        Model forward method.

        Args:
        Returns:
            output (dict): output dict containing logits.
        """
        intermediate_feat = []
        x = F.pad(mel_spec, (0, self.p - (mel_spec.shape[-1] % self.p)), "reflect")
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] // self.p, self.p)

        # i hate intermediate_feats, i cant use my fav Sequential
        for block in self.body:
            x = block(x)
            intermediate_feat.append(x)

        x = self.end(x)
        intermediate_feat.append(x)
        return torch.flatten(x, 1, -1), intermediate_feat

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


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, in_chanels, p=[2, 3, 5, 7, 11]):
        """
        Args:
            p - period for each descr
        """
        super().__init__()
        self.discriminators = nn.ModuleList()
        for curr_p in p:
            self.discriminators.append(Discriminator(in_chanels, curr_p))

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
            curr_res, curr_inter_feat = disc(mel_spect)
            result.append(curr_res)
            intermediate_feat.append(curr_inter_feat)
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
