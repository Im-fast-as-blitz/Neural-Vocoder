import torch
from torch import nn


class GANLoss(nn.Module):
    """
    GAN loss
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        disc_from_generator: torch.Tensor,
        disc_from_x: torch.Tensor,
        gen: bool = True,
        **batch
    ):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            disc_from_generator (Tensor): generator output predictions.
            disc_from_x (Tensor): disc_c output predictions.
            gd: flag
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        loss = 0
        if gen:
            for ind in range(len(disc_from_generator)):
                loss += torch.mean((disc_from_generator[ind] - 1) ** 2)
            return loss
        loss = 0
        for ind in range(len(disc_from_generator)):
            loss += torch.mean((disc_from_x[ind] - 1) ** 2) + torch.mean(
                (disc_from_generator[ind]) ** 2
            )

        return loss
