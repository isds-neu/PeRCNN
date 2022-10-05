# PeRCNN

Encoding physics to learn reaction-diffusion processes

## Overview
Modeling complex dynamical systems such as the reaction-diffusion processes have largely relied on partial differential equations (PDEs). However, due to insufficient prior knowledge on the concerned system and the lack of explicit PDE formulation used for describing the nonlinear reaction process, predicting the system evolution remains a challenging issue in many scientific problems commonly seen in chemistry, biology, geology, physics and ecology. Unifying measurement data and our limited prior physics knowledge via machine learning provides us a new path of solution to this problem. Existing physics-informed learning paradigms impose physics through ``soft'' penalty constraints, whose solution quality largely depends on a trial-and-error proper setting of hyperparameters. Since the core of such methods is still rooted on “black-box” neural networks, the resulting model generally lacks interpretability and suffers from critical issues of extrapolation and generalization. To this end, we propose a novel deep learning framework that forcibly encodes/preserves given physics structure to facilitate the learning of the spatiotemporal dynamics in sparse data regimes. We show how the proposed approach can be applied to a wide range of reaction-diffusion problems, including forward and inverse analysis of PDE systems, data-driven modeling, and discovery of governing PDEs. The resultant learning paradigm that encodes physics shows high accuracy, robustness, interpretability and generalizability demonstrated via extensive numerical experiments.


## System Requirements

### Hardware requirements

We train our ``PeRCNN`` and the baseline models on an Nvidia DGX with four Tesla V100 GPU of 32 GB memory. 

### Software requirements

#### OS requirements
 
 - Window 10 Pro
 - Linux: Ubuntu 18.04.3 LTS

#### Python requirements

- Python 3.6.13
- [Pytorch](https://pytorch.org/) 1.6.0
- Numpy 1.16.5
- Matplotlib 3.2.2
- scipy 1.3.1

## Installtion guide

It is recommended to install Python from Anaconda with GPU support, and then install the related packages via conda setting.  

## How to run

### Dataset

Considering the traing data size being over large, we provide a Google drive link for testing our models. Besides, we also uploaded the simulation code with high-order finite difference methods for readers to play with. 

### Implementation

Generally, we evaluate our `PeRCNN` on four tasks: 
- Solving PDEs (compare w/ [PINN](https://www.sciencedirect.com/science/article/pii/S0021999118307125), [ConvLSTM](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf))
- Coefficients identification
- Data reconstruction (compare w/ [ConvLSTM](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf), [Recurrent ResNet](https://arxiv.org/pdf/1610.00081.pdf), [PDE-Net](https://arxiv.org/pdf/1710.09668.pdf), [DHPM](https://arxiv.org/pdf/1801.06637.pdf))
- Discovering PDEs (compare w/ [ConvLSTM](https://proceedings.neurips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf), [Recurrent ResNet](https://arxiv.org/pdf/1610.00081.pdf), [PDE-Net](https://arxiv.org/pdf/1710.09668.pdf), [DHPM](https://arxiv.org/pdf/1801.06637.pdf))

We present three folders for solving PDEs, data reconstruction and equation discovery. The coefficients identification can be referred to the equation discovery folder, which is essentially the Stage-3 part. More implementation details can be found in each folder. 

## License

This project is covered under the MIT License (MIT).

