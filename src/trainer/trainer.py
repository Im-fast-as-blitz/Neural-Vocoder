from random import shuffle

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.transforms.mel_spect import MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            for opt in self.optimizer:
                self.optimizer[opt].zero_grad()

        outputs = self.model(is_gen=False, **batch)
        batch.update(outputs)

        all_losses = self.criterion(is_gen=False, **batch)
        batch.update(all_losses)

        if self.is_train:
            torch.autograd.set_detect_anomaly(True)
            batch["loss_disc"].backward()
            self._clip_grad_norm()
            self.optimizer["disc"].step()

        outputs = self.model(is_gen=True, **batch)
        batch.update(outputs)

        all_losses = self.criterion(is_gen=True, **batch)
        batch.update(all_losses)

        if self.is_train:
            # torch.autograd.set_detect_anomaly(True)
            self.optimizer[
                "gen"
            ].zero_grad()  # на всяйкий случай так как после disc в нем уже лежит лишний для нас градиент
            batch["loss_gener"].backward()
            self._clip_grad_norm()
            self.optimizer["gen"].step()

            if self.lr_scheduler is not None:
                for part in self.lr_scheduler:
                    self.lr_scheduler[part].step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())
        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if mode == "train":  # the method is called only every self.log_step steps
            self._log_spectrogram(**batch)
        else:
            # Log Stuff
            self._log_spectrogram(**batch)
            self._log_preds(**batch)

    def _log_spectrogram(self, mel_spect_data_object, mel_spec_pred, **batch):
        spectrogram_for_plot = mel_spect_data_object[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("true_spectrogram", image)

        spectrogram_for_plot = mel_spec_pred[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("pred_spectrogram", image)

    def _log_preds(self, examples_to_log=5, **batch):
        result = {}
        examples_to_log = min(examples_to_log, batch["audio_pred"].shape[0])

        tuples = list(
            zip(batch["audio_pred"], batch["audio_data_object"], batch["text"])
        )
        shuffle(tuples)

        for idx, (pred, target, text) in enumerate(tuples[:examples_to_log]):
            result[idx] = {
                "predict_audio": self.writer.wandb.Audio(
                    pred.squeeze(0).detach().cpu().numpy(),
                    sample_rate=MelSpectrogramConfig.sr,
                ),
                "target_audio": self.writer.wandb.Audio(
                    target.squeeze(0).detach().cpu().numpy(),
                    sample_rate=MelSpectrogramConfig.sr,
                ),
                "text": text,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(result, orient="index")
        )
