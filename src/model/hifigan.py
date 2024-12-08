import math

from torch import nn
from torch.nn import Sequential

from src.model.hifi_gan import (
    Generator,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
)
from src.transforms.mel_spect import MelSpectrogram


class HiFiGAN(nn.Module):
    """
    HiFiGAN
    """

    def __init__(self, in_chanels, Hu, Ku, Kr, Dr):
        """
        Args:
            in_chanels - in_chanels gor generator, 80 often
            Hu - hidden dimension of the transposed convolutions
            Ku - kernel sizes of the transposed convolutions
            Kr - kernel sizes of MRF modules
            Dr - dilation rates of MRF modules
        """
        super().__init__()
        self.msd = MultiScaleDiscriminator(1)
        print("Init MultiScaleDiscriminator")
        self.mpd = MultiPeriodDiscriminator(1)
        print("Init MultiPeriodDiscriminator")
        self.gen = Generator(in_chanels, Hu, Ku, Kr, Dr)
        print("Init Generator")

    def forward(
        self, is_gen, audio_data_object, mel_spect_data_object, is_train=True, **batch
    ):
        """
        Model forward method.

        Args:
            gen (bool): apply model for generator or disc.
            audio_data_object (Tensor): input raw audio vector.
            mel_spect_data_object (Tensor): input mel spect of audio vector.
        Returns:
            output (dict): output dict containing logits.
        """
        result = {}
        if "audio_pred" not in batch and "mel_spec_pred" not in batch:
            audio_pred = self.gen(mel_spect_data_object)
            mel_spec_pred = MelSpectrogram().to(audio_pred.device)(
                audio_pred.squeeze(1)
            )
            result = {
                "audio_pred": audio_pred,
                "mel_spec_pred": mel_spec_pred,
            }
            if not is_train:
                return result

        if is_gen:
            # for generator
            mpd_target, interm_feat_mpd_target = self.mpd(
                audio_data_object.unsqueeze(1)
            )
            mpd_pred, interm_feat_mpd_pred = self.mpd(batch["audio_pred"])

            msd_target, interm_feat_msd_target = self.msd(
                audio_data_object.unsqueeze(1)
            )
            msd_pred, interm_feat_msd_pred = self.msd(batch["audio_pred"])

            result.update(
                {
                    "mpd_target": mpd_target,
                    "mpd_pred": mpd_pred,
                    "msd_target": msd_target,
                    "msd_pred": msd_pred,
                    "interm_feat_mpd_target": interm_feat_mpd_target,
                    "interm_feat_mpd_pred": interm_feat_mpd_pred,
                    "interm_feat_msd_target": interm_feat_msd_target,
                    "interm_feat_msd_pred": interm_feat_msd_pred,
                }
            )
        else:
            # for disc
            mpd_target, _ = self.mpd(audio_data_object.unsqueeze(1))
            msd_target, _ = self.msd(audio_data_object.unsqueeze(1))

            mpd_pred, _ = self.mpd(result["audio_pred"].detach())
            msd_pred, _ = self.msd(result["audio_pred"].detach())

            result.update(
                {
                    "mpd_target": mpd_target,
                    "mpd_pred": mpd_pred,
                    "msd_target": msd_target,
                    "msd_pred": msd_pred,
                }
            )

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
