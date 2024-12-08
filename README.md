# Neural Vocoder

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository is a system for training the [`deepspeech2`](http://proceedings.mlr.press/v48/amodei16.pdf) model for an ASR task.

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

**IMPORTANT** all your dir with data should contain all audio in dir "wavs" and all text files in dir "transcriptions"

To train a model, run the following command:

```bash
# use default data dir path
python3 train.py -cn=hifigan_v1

# se custom data dir path
python3 train.py -cn=hifigan_v1 +dir=PATH_TO_TRAIN_DATA
```

You can add by a custom argument "dir" path to train data. Code will use all audio files to train in PATH_TO_TRAIN_DATA/wavs dir

-------

To run inference (evaluate the model or save predictions):

Dowload model:

```bash
python3 download_model.py
```

For predicts by melspec from audio:

```bash
python3 synthesize.py -cn=inference +dir=PATH_TO_TRAIN_DATA
```

For predicts by melspec from text:

```bash
python3 synthesize.py -cn=inference_from_text +dir=PATH_TO_TRAIN_DATA
```

And all output audios will be saved in dir data/saved/predict

## About model

[`WANDB`](https://drive.google.com/file/d/1LoU_kCzl20hM5p709teRPB0a4Jib8VK-/view?usp=sharing)

[`Final model`](https://drive.google.com/file/d/1LoU_kCzl20hM5p709teRPB0a4Jib8VK-/view?usp=sharing)

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
