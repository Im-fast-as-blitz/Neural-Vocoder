import glob
import os
import urllib.request
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from src.metrics.base_metric import BaseMetric
from src.transforms.mel_spect import MelSpectrogramConfig

# from wvmos import get_wvmos


def extract_prefix(prefix, weights):
    result = OrderedDict()
    for key in weights:
        if key.find(prefix) == 0:
            result[key[len(prefix) :]] = weights[key]
    return result


class Wav2Vec2MOS(nn.Module):
    def __init__(self, path, freeze=True, cuda=True):
        super().__init__()
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.freeze = freeze
        self.dense = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

        if self.freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.load_state_dict(
            extract_prefix(
                "model.",
                torch.load(path, map_location=torch.device("cuda" if cuda else "cpu"))[
                    "state_dict"
                ],
            )
        )
        self.eval()
        self.cuda_flag = cuda
        if cuda:
            self.cuda()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.processor.feature_extractor.sampling_rate = MelSpectrogramConfig.sr

    def forward(self, x):
        x = self.encoder(x)["last_hidden_state"]  # [Batch, time, feats]
        x = self.dense(x)  # [batch, time, 1]
        x = x.mean(dim=[1, 2], keepdims=True)  # [batch, 1, 1]
        return x

    def train(self, mode):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()

    def calculate_one(self, x):
        # signal = librosa.load(path, sr=MelSpectrogramConfig.sr)[0]
        x = self.processor(
            x.squeeze(0),
            return_tensors="pt",
            padding=True,
            sampling_rate=MelSpectrogramConfig.sr,
        ).input_values
        with torch.no_grad():
            if self.cuda_flag:
                x = x.cuda()
            res = self.forward(x).mean()
        return res.cpu().item()


def get_wvmos(cuda=True):
    path = os.path.join(os.path.expanduser("~"), ".cache/wv_mos/wv_mos.ckpt")

    if not os.path.exists(path):
        print("Downloading the checkpoint for WV-MOS")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(
            "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1", path
        )
    print("Weights downloaded in: {} Size: {}".format(path, os.path.getsize(path)))

    return Wav2Vec2MOS(path, cuda=cuda)


class WVMOS(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = get_wvmos(cuda=(device == "cuda"))

    def __call__(self, audio_pred: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            audio_pred (Tensor): model output predictions.
        Returns:
            metric (float): calculated metric.
        """
        result = []
        for pred in audio_pred:
            result.append(self.metric.calculate_one(pred))
        return np.mean(result)
