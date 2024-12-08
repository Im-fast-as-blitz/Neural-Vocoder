import csv
import json
import logging
import os
import posixpath
import random
from typing import Optional

import torch.nn.functional as F

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)


class СustomAudioDataset(BaseDataset):
    def __init__(
        self,
        audio_size: int = 8192,
        with_texts: bool = True,
        with_audio: bool = False,
        dir: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if dir is None:
            data_dir = ROOT_PATH / "data" / "LJSpeech-1.1"
        else:
            data_dir = ROOT_PATH / dir
        data_dir.mkdir(exist_ok=True, parents=True)
        self._data_dir = data_dir
        self._with_texts = with_texts
        self._with_audio = with_audio
        self._audio_size = audio_size
        index = self._get_or_load_index(with_texts, with_audio)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, with_texts, with_audio):
        index_path = self._data_dir / "index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(with_texts, with_audio)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, with_texts, with_audio):
        index = []
        memory = {}
        audio_dir = self._data_dir / "wavs"
        trans_dir = self._data_dir / "transcriptions"

        if with_audio:
            for root, _, files in os.walk(audio_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file[-4:] != ".wav":
                        continue
                    wav_id = file.replace(".wav", "")
                    paths = {
                        "id": wav_id,
                        "path_audio": file_path,
                    }
                    memory[wav_id] = len(index)

                    index.append(paths)
        if with_texts:
            for root, _, files in os.walk(trans_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    trans_id = file.replace(".txt", "")

                    with open(file_path, "r") as csvfile:
                        text = csvfile.read()

                    if with_audio and trans_id in memory:
                        index[memory[trans_id]].update(
                            {
                                "text": text,
                            }
                        )
                    else:
                        paths = {
                            "id": trans_id,
                            "text": text,
                        }
                        index.append(paths)

        if posixpath.basename(self._data_dir) == "LJSpeech-1.1":
            with open(self._data_dir / "metadata.csv", "r") as csvfile:
                for row in csvfile.readlines():
                    wav_id, _, norm_text = row.split("|")
                    index[memory[wav_id]].update(
                        {
                            "text": norm_text[:-1],
                        }
                    )

        return index

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        if not self._with_audio:
            raise Exception("At this moment out class can not work with text only")

        audio_data_object = self.load_object(data_dict["path_audio"])
        if self._audio_size != -1:
            if audio_data_object.shape[1] >= self._audio_size:
                start = random.randint(0, audio_data_object.shape[1] - self._audio_size)
                audio_data_object = audio_data_object[
                    :, start : start + self._audio_size
                ]
            else:
                audio_data_object = F.pad(
                    audio_data_object,
                    (0, self._audio_size - audio_data_object.shape[1]),
                    "constant",
                )

        instance_data = {
            "audio_data_object": audio_data_object,
            "mel_spect_data_object": audio_data_object,
        }
        instance_data = self.preprocess_data(instance_data)

        instance_data["id"] = data_dict["id"]

        if "text" in data_dict:
            instance_data["text"] = data_dict["text"]

        return instance_data
