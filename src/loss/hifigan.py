import torch
from torch import nn

from src.loss.hifigan_losses.feat_match_loss import FeatMatchLoss
from src.loss.hifigan_losses.gan_loss import GANLoss
from src.loss.hifigan_losses.mel_loss import MelLoss


class HiFiGanLoss(nn.Module):
    """
    HiFiGan loss
    """

    def __init__(self, fm_weight: float = 2.0, m_weight: float = 45.0):
        super().__init__()
        self.fm_weight = fm_weight
        self.m_weight = m_weight

        self.mel_loss = MelLoss()
        self.feat_match_loss = FeatMatchLoss()
        self.gan_loss = GANLoss()

    def forward(self, is_gen: bool, **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:

        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        if not is_gen:
            # disc loss
            gan_disc_loss_mpd = self.gan_loss(
                disc_from_generator=batch["mpd_pred"],
                disc_from_x=batch["mpd_target"],
                gen=False,
            )
            gan_disc_loss_msd = self.gan_loss(
                disc_from_generator=batch["msd_pred"],
                disc_from_x=batch["msd_target"],
                gen=False,
            )
            loss_disc = gan_disc_loss_mpd + gan_disc_loss_msd

            return {
                "gan_disc_loss_mpd": gan_disc_loss_mpd,
                "gan_disc_loss_msd": gan_disc_loss_msd,
                "loss_disc": loss_disc,
            }
        else:
            # generator loss
            gan_gen_loss_mpd = self.gan_loss(
                disc_from_generator=batch["mpd_pred"],
                disc_from_x=batch["mpd_target"],
                gen=True,
            )
            gan_gen_loss_msd = self.gan_loss(
                disc_from_generator=batch["msd_pred"],
                disc_from_x=batch["msd_target"],
                gen=True,
            )

            mel_spect_loss = self.mel_loss(
                mel_true=batch["mel_spect_data_object"], mel_gen=batch["mel_spec_pred"]
            )

            fm_loss_mpd = self.feat_match_loss(
                disc_from_generator=batch["interm_feat_mpd_pred"],
                disc_from_x=batch["interm_feat_mpd_target"],
            )
            fm_loss_msd = self.feat_match_loss(
                disc_from_generator=batch["interm_feat_msd_pred"],
                disc_from_x=batch["interm_feat_msd_target"],
            )

            loss_gener = (
                gan_gen_loss_mpd
                + gan_gen_loss_msd
                + mel_spect_loss * self.m_weight
                + self.fm_weight * fm_loss_mpd
                + self.fm_weight * fm_loss_msd
            )

        return {
            "gan_gen_loss_mpd": gan_gen_loss_mpd,
            "gan_gen_loss_msd": gan_gen_loss_msd,
            "mel_spect_loss": mel_spect_loss,
            "fm_loss_mpd": fm_loss_mpd,
            "fm_loss_msd": fm_loss_msd,
            "loss_gener": loss_gener,
        }
