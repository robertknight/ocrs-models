# Training Ocrs models

This document describes how to train models for use with
[ocrs](https://github.com/robertknight/ocrs).

## Prerequisites

To train the models you will need:

- Python 3.14 or later. uv will install this automatically if needed.
- A GPU. The initial training was done on NVidia A10G GPUs with 24 GB RAM, via
  [AWS EC2 G5 instances](https://aws.amazon.com/ec2/instance-types/g5/) (the
  smallest `g5.xlarge` size will work).
- Optional: A Weights and Biases account (https://wandb.ai/) to track training progress

## About Ocrs models

Ocrs uses two PyTorch models, which can be trained using this repository:

 1. Text detection
 2. Text recognition

You can mix and match default/pre-trained and custom models. For example you
may wish to use a pre-trained detection model but a custom recognition model.

## Clone the repository

```
git clone https://github.com/robertknight/ocrs-models.git
cd ocrs-models
```

## Download the dataset

Following the instructions in
https://github.com/google-research-datasets/hiertext#getting-started, clone the
HierText repository and download the training data.

Note that you do **not** need to follow the step about decompressing the
`.jsonl.gz` files. The training tools will do this for you.

The compressed dataset is ~3.6 GB in total size.

```
# Clone the HierText repository. This contains the ground truth data.
mkdir -p datasets/
cd datasets
git clone https://github.com/google-research-datasets/hiertext.git
cd hiertext

# Download the training, validation and test images.
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/train.tgz .
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/validation.tgz .
aws s3 --no-sign-request cp s3://open-images-dataset/ocr/test.tgz .

# Decompress the datasets.
tar -xf train.tgz
tar -xf validation.tgz
tar -xf test.tgz
```

## Set up the training environment

1. Install [uv](https://docs.astral.sh/uv/)
2. Install dependencies:

	 ```
	 uv sync
	 ```

   This installs PyTorch from PyPI. On Linux this is a build for NVIDIA GPUs
   using CUDA 13. On macOS it is a build which supports Apple GPUs. If you
   need a different build (eg. for an older CUDA version or AMD GPUs), see
   https://docs.astral.sh/uv/guides/integration/pytorch/.

3. Start a dummy training run of text detection training to verify everything is working:

	 ```
	 uv run python -m ocrs_models.train_detection hiertext datasets/hiertext/ --max-images 100
	 ```
	
   Wait for one successful epoch of training and validation to complete and then
   exit the process with Ctrl+C.

## Set up Weights and Biases integration (optional)

The ocrs-models training scripts support tracking training progress using
[Weights and Biases](https://wandb.ai). To enable this you will need to create
an account and then set the `WANDB_API_KEY` environment variable before running
training scripts:

```
export WANDB_API_KEY=<your_api_key>
```

## Train on a remote GPU host (optional)

If the dataset is on a machine without a suitable GPU, you can train on a
remote Linux host with an NVIDIA GPU via SSH:

```
just train-detection-ssh user@host
just train-recognition-ssh user@host
```

These commands copy the code and dataset to `~/ocrs-models` on the remote host
using rsync, train the model and export it to ONNX, then copy the checkpoint
and ONNX model back to the local repository root. The remote host needs
passwordless SSH login, rsync and [uv](https://docs.astral.sh/uv/). If
`WANDB_API_KEY` is set locally, it is forwarded to the remote host.

Extra arguments are passed to the training script, and override the default
`--max-epochs` and `--batch-size` values given in the sections below. For
example, to do a quick test run:

```
just train-detection-ssh user@host --max-epochs 1
```

Training stops if the SSH connection drops. Training runs can take a long time,
so you may want to run these commands inside `tmux` or `screen` locally.

## Train the text detection model

To launch a training run for the text detection model, run:

```
uv run python -m ocrs_models.train_detection hiertext datasets/hiertext/ \
  --max-epochs 50 \
  --batch-size 28
```

The `--batch-size` flag will need to be varied according to the amount of GPU
memory you have available. One way to do this is to start with a small value,
and then increase it until the training process is using most of the available
GPU memory. The above value was used with a GPU that has 24 GB of memory. When
training with an NVidia GPU, you can use the `nvidia-smi` tool to get memory
usage statistics.

To fine-tune an existing model, pass the `--checkpoint` flag to specify the
pre-trained model to start with.

### Export the text detection model

As training progresses, the latest checkpoint will be saved to
`text-detection-checkpoint.pt`. Once training completes, you can export the
model to ONNX via:

```
uv run python -m ocrs_models.train_detection hiertext datasets/hiertext/ \
  --checkpoint text-detection-checkpoint.pt \
  --export text-detection.onnx
```

### Use the text detection model

The exported ONNX model can be used with the `ocrs` Rust crate and `ocrs` CLI.

To use the exported model with the CLI, you can either pass the model path via
CLI arguments, or replace the default models in the cache directory
(`~/.cache/ocrs`). Example using CLI arguments:

```sh
ocrs --detect-model text-detection.onnx image.jpg
```

## Train the text recognition model

To launch a training run for the text recognition model, run:

```
uv run python -m ocrs_models.train_rec hiertext datasets/hiertext/ \
  --max-epochs 50 \
  --batch-size 250
```

The `--batch-size` flag will need to be varied according to the amount of GPU
memory you have available. One way to do this is to start with a small value,
and then increase it until the training process is using most of the available
GPU memory. The above value was used with a GPU that has 24 GB of memory.

To fine-tune an existing model, pass the `--checkpoint` flag to specify the
pre-trained model to start with.

### Export the text recognition model

As training progresses, the latest checkpoint will be saved to
`text-rec-checkpoint.pt`. Once training completes, you can export the model to
ONNX via:

```
uv run python -m ocrs_models.train_rec hiertext datasets/hiertext/ \
  --checkpoint text-rec-checkpoint.pt \
  --export text-recognition.onnx
```

### Use the text recognition model

The exported ONNX model can be used with the `ocrs` Rust crate and `ocrs` CLI.

To use the exported model with the CLI, you can either pass the model path via
CLI arguments, or replace the default models in the cache directory
(`~/.cache/ocrs`). Example using CLI arguments:

```sh
ocrs --rec-model text-recognition.onnx image.jpg
```
