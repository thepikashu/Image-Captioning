# Image Captioning for Remote Sensing Data

## Overview

This project implements an Image Captioning system for remote sensing images using deep learning techniques.
It combines:

* CNN Encoder (ResNet18) for feature extraction
* LSTM Decoder for sequence generation
* Transformer Decoder for advanced caption generation

The model generates natural language descriptions of satellite images.

## Objectives

* Automatically generate captions for remote sensing images
* Compare LSTM vs Transformer performance
* Evaluate using standard NLP metrics

## Dataset

We use the RSICD (Remote Sensing Image Captioning Dataset).
Total images: 10,921  
Captions per image: 5  
Image size: 224 × 224  
Source: Google Earth, Baidu Maps, etc.  

Download from:
https://huggingface.co/datasets/arampacha/rsicd

## Features

* End-to-end training pipeline
* Dual architecture:
  * CNN + LSTM
  * CNN + Transformer
* Vocabulary creation with frequency filtering
* Custom dataset handling
* Caption generation (greedy decoding)
* Evaluation using BLEU & METEOR


## Model Architecture

### CNN Encoder

* Pretrained ResNet18
* Extracts image features (512-dim vector)

### LSTM Decoder

* Embedding layer
* LSTM network
* Fully connected output

### Transformer Decoder

* Positional Encoding
* Multi-head attention
* Transformer decoder layers

## Workflow

1. Load dataset
2. Tokenize captions (NLTK)
3. Build vocabulary
4. Encode images & captions
5. Train models
6. Evaluate performance
7. Generate captions

## Future Improvements

* Beam search decoding
* Attention visualization
* Fine-tune full CNN
* Use larger pretrained models
* Better preprocessing

## Author

Developed as part of a Deep Learning / Image Captioning project.
