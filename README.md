# ocrs-models

This project contains tools for training PyTorch models for use with the
[**Ocrs**](https://github.com/robertknight/ocrs/) OCR engine.

## About the models

This repository contains the two models used by the ocrs engine:

1. **Text detection**: This is a semantic segmentation model which classifies
   each pixel in a greyscale input image as text/non-text. Consumers then
   post-process clusters of text pixels to get oriented bounding boxes for
   words.
2. **Text recognition**: This is a CRNN model that takes a greyscale image of a
   text line as input and returns a sequence of characters.

All models can be exported to ONNX for downstream use.

## Datasets

The models are trained exclusively on datasets which are a) open and b) have non-restrictive licenses. This currently includes:
- [HierText](https://github.com/google-research-datasets/hiertext) (CC-BY-SA 4.0)

## Pre-trained models

Pre-trained models are available from [Hugging
Face](https://huggingface.co/robertknight/ocrs) as PyTorch checkpoints and
[ONNX](https://onnx.ai) models.

## Training custom models

See the [Training guide](docs/training.md) for a walk-through of the process to
train models from scratch or fine-tune existing models.
