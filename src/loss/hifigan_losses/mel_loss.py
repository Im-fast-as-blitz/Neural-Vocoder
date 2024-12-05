import torch
import torch.nn.functional as F
from torch import nn


class MelLoss(nn.Module):
    """
    GAN loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, mel_true: torch.Tensor, mel_gen: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            mel_true (Tensor): true e,lspect.
            mel_gen (Tensor): generator output melspect.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return F.l1_loss(mel_true, mel_gen)
