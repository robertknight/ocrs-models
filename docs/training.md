# Training Ocrs models

This document describes how to train models for use with
[ocrs](https://github.com/robertknight/ocrs).

## Prerequisites

To train the models you will need:

- Python 3.10 or later
- A GPU. The initial training was done on NVidia A10G GPUs with 24 GB RAM, via
  [AWS EC2 G5 instances](https://aws.amazon.com/ec2/instance-types/g5/) (the
  smallest `g5.xlarge` size will work).
- Optional: A Weights and Biases account (https://wandb.ai/) to track training progress

## About Ocrs models

Ocrs splits the OCR process into three stages:

 1. Text detection
 2. Layout analysis
 3. Text recognition

Each of these stages corresponds to a separate PyTorch model. The layout
analysis model is incomplete and is not currently used in Ocrs.

You can mix and match default/pre-trained and custom models for the different
stages. For example you may wish to use a pre-trained detection model but a
custom recognition model.

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

1. Install [Poetry](https://python-poetry.org)
2. Install dependencies, except for PyTorch:

	 ```
	 poetry install
	 ```

3. Install the appropriate version of PyTorch for your system, in the virtualenv
   created by Poetry:

   ```
   poetry run pip install torch torchvision
   ```

   See https://pytorch.org/get-started/locally/ for an appropriate pip command
   depending on your platform and GPU.

4. Start a dummy training run of text detection training to verify everything is working:

	 ```
	 poetry run python -m ocrs_models.train_detection hiertext datasets/hiertext/ --max-images 100
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

## Train the text detection model

To launch a training run for the text detection model, run:

```
poetry run python -m ocrs_models.train_detection hiertext datasets/hiertext/ \
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
poetry run python -m ocrs_models.train_detection hiertext datasets/hiertext/ \
  --checkpoint text-detection-checkpoint.pt \
  --export text-detection.onnx
```

### Convert the text detection model

To use the exported ONNX model with Ocrs, you will need to convert it to
the `.rten` format used by [RTen][rten].

See the [RTen README](https://github.com/robertknight/rten#getting-started)
for current instructions on how to do this.

To use the converted model with the `ocrs` CLI tool, you can either pass the
model path via CLI arguments, or replace the default models in the cache
directory (`~/.cache/ocrs`). Example using CLI arguments:

```sh
ocrs --detect-model custom-detection-model.rten image.jpg
```

[rten]: https://github.com/robertknight/rten

## Train the text recognition model

To launch a training run for the text recognition model, run:

```
poetry run python -m ocrs_models.train_rec hiertext datasets/hiertext/ \
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
poetry run python -m ocrs_models.train_rec hiertext datasets/hiertext/ \
  --checkpoint text-rec.pt \
  --export text-recognition.onnx
```

### Convert the text recognition model

To use the exported ONNX models with Ocrs, convert it to `.rten` format using
the same process as for the detection model.

To use the converted model with the `ocrs` CLI tool, you can either pass the
model path via CLI arguments, or replace the default models in the cache
directory (`~/.cache/ocrs`). Example using CLI arguments:

```sh
ocrs --rec-model custom-recognition-model.rten image.jpg
```
