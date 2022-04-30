# 10708 Project - Active Learning with Generative Models
Obtaining labeled data is a significant challenge in many supervised machine learning tasks such as image classification. Active learning has shown to perform well in cases where we have a limited budget to label the training data. In this project, we aim to use generative models such as GANs and VAEs to augment the data to improve the performance of active learning tasks. We examine the image classification task in settings where we do not have many labeled data samples. We use active learning to identify highly uncertain samples, use generative models to augment the data to obtain more similar samples, and then actively label the data to improve the classifier using these additional samples. We compare the results of our method with pre-existing active learning approaches and provide thorough insights into the use of generative models for active learning.

# Requirements
- python 3.8.12
- pytorch 1.10.0
- torchvision 0.11.1
- tensorflow
- keras
- modAL
- skorch
- matplotlib

# Instructions for running the code:
For MNIST, please follow the instructions below:
- Run `submit/MNIST/dbal_pytorch_paper.ipynb` for running the code for Active Learning on MNIST without data augmentation
- Run `submit/MNIST/dbal_pytorch_paper_vae_gan.ipynb` for running the code for Active Learning on MNIST with VAEGAN
- Run `submit/MNIST/dbal_pytorch_vae.ipynb` for running the code for Active Learning on MNIST with VAE
- Run `submit/MNIST/dbal_pytorch_duplicate.ipynb` for running the code for Active Learning on MNIST with duplicated samples
- Run `submit/MNIST/bgadl_mnist.py` for running the code for Active Learning on MNIST with VAE-ACGAN joint training

For CIFAR-10, please follow the instructions below:
- Run `submit/CIFAR10/dbal_pytorch_paper_cifar.py` for running the code for Active Learning on CIFAR10 without data augmentation
- Run `submit/CIFAR10/dbal_pytorch_paper_cifar_pix.py` for running the code for Active Learning on CIFAR10 with Pix2Pix
- Run `submit/CIFAR10/dbal_pytorch_paper_cifar_vae.py` for running the code for Active Learning on CIFAR10 with VAE
- Run `submit/CIFAR10/dbal_pytorch_paper_cifar_duplicate.ipynb` for running the code for Active Learning on CIFAR10 with duplicated samples
- Run `submit/CIFAR10/bgadl_cifar.py` for running the code for Active Learning on CIFAR10 with VAE-ACGAN joint training

For plotting the graphs, run `submit/plotting/plotting_cifar_smoothen.ipynb` (for CIFAR-10) and `submit/plotting/plotting_mnist_smoothen.ipynb` (for MNIST) after running the code above
