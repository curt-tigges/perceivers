# Perceivers
My implementation of the [Perceiver](https://arxiv.org/abs/2103.03206) and [Perceiver IO](https://arxiv.org/abs/2107.14795) papers.

## Overview
This repository contains my from-scratch implementation of the Perceiver model and the subsequent Perceiver IO architecture. You can see the full details in my [blog post](), which I wrote as an introduction/tutorial for Perceivers. This particular one I trained on MNIST, though that is only the tiniest tip of the iceberg of what Perceivers can do.


### Data
I used the MNIST dataset for training, but this architecture works well with many similar datasets, including ImageNet for vision, StarCraft II, and many others.

## Environment & Setup
This model was trained with the following packages:
- `pytorch 1.8.2`
- `torchvision 0.9.2`
- `pytorch-lightning 1.6.1`
- `torchmetrics 0.8.0`

## Repo Structure
### / (root)
- pl_perceiver_training_module.py - PyTorch Lightning training module
- pl_perceiver_io_training_module.py - PyTorch Lightning training module for Perceiver IO variant
- perceiver-demo.ipynb - Demo training notebook with all modules defined 
- perceiver-io-demo.ipynb - Demo training notebook for Perceiver IO with all modules defined

### data/
- mnist.py - Data module for MNIST
- cifar10.py - Data module for CIFAR10
- cifar100.py - Data module for CIFAR100

### models/
- positional_image_embedding.py - Implementation of flattening, position encoding, etc. for Perceiver image model training
- perceiver.py - A simple module that will download and initialize any desired backbone from the TIMM library.
- perceiver_io.py - Includes my implementation of the overall architecture.

## Usage
### Training
To train this model with MNIST, simply run through the `perceiver-demo.ipynb` or `perceiver-io-demo.ipynb` notebook.

## Results
This model was able to get 97% accuracy on MNIST. Settings for achieving this are explained in the [blog post](https://medium.com/@curttigges/building-a-transformer-powered-sota-image-labeller-cfe25e6d69f1).

## To Do
- [x] Create initial complete Perceiver model
- [x] Train model on MNIST
- [ ] Implement Perceiver IO decoder
- [ ] Train model on larger image dataset
