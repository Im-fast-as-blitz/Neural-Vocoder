import torch
from torch.nn.utils.rnn import pad_sequence

from src.transforms.mel_spect import MelSpectrogramConfig


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["audio_data_object"] = pad_sequence(
        [item["audio_data_object"].squeeze(0) for item in dataset_items],
        batch_first=True,
    )

    # for ind in range(len(dataset_items)):
    #     print(ind, dataset_items[ind]["audio_data_object"].shape, dataset_items[ind]["mel_spect_data_object"].shape)

    result_batch["mel_spect_data_object"] = pad_sequence(
        [
            item["mel_spect_data_object"].squeeze(0).permute(1, 0)
            for item in dataset_items
        ],
        batch_first=True,
        padding_value=MelSpectrogramConfig.pad_value,
    ).permute(0, 2, 1)

    result_batch["id"] = [item["id"] for item in dataset_items]
    if "text" in dataset_items[0]:
        result_batch["text"] = [item["text"] for item in dataset_items]

    return result_batch
