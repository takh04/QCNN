# Quantum Convolutional Neural Network (QCNN)

This repository implements a Quantum Convolutional Neural Network (QCNN) for classical data classification, based on the paper [Quantum Convolutional Neural Networks](https://arxiv.org/abs/2108.00661). The implementation uses PennyLane for quantum circuit simulation and focuses on classifying MNIST and Fashion MNIST datasets.

## Features

- Implementation of QCNN with various circuit architectures
- Support for multiple data embedding methods
- Classical data preprocessing with PCA and Autoencoder
- Training with both MSE and Cross Entropy loss functions
- Benchmarking against classical CNN
- Support for both MNIST and Fashion MNIST datasets
- Hierarchical Quantum Classifier implementation

## Project Structure

### 1. Data Preparation (`data.py`)
- Handles dataset loading (MNIST/Fashion MNIST)
- Implements classical preprocessing methods:
  - PCA (8, 12, 16, 30, 32 dimensions)
  - Autoencoder (8, 12, 16, 30, 32 dimensions)
  - Image resizing
- Supports binary classification with customizable classes

### 2. Quantum Circuit Implementation
- `QCNN_circuit.py`: Main QCNN implementation with three variants:
  - Standard QCNN with convolution and pooling layers
  - QCNN without pooling layers
  - 1D QCNN circuit
- `unitary.py`: Contains unitary ansatze for:
  - Convolution layers (U_TTN, U_5, U_6, U_9, U_13, U_14, U_15, U_SO4, U_SU4)
  - Pooling layers
- `embedding.py`: Implements five data embedding methods:
  - Amplitude Embedding
  - Angle Embedding
  - Compact Angle Embedding
  - Hybrid Direct Embedding
  - Hybrid Angle Embedding
- `Angular_hybrid.py`: Implementation of Hybrid Angle Embedding

### 3. Training (`Training.py`)
- Implements quantum circuit training with:
  - Nesterov momentum optimizer
  - Batch size of 25
  - 200 training iterations
  - Support for both MSE and Cross Entropy loss
- Configurable parameters for different circuit architectures

### 4. Benchmarking (`Benchmarking.py`)
- Trains and evaluates quantum circuits
- Compares performance with classical CNN
- Saves training history and test accuracy
- Supports various combinations of:
  - Datasets
  - Unitary ansatze
  - Embedding methods
  - Feature reduction techniques



## Binary Classification Labels

- When using MSE loss: Labels can be either `{1, -1}` (binary=True) or `{1, 0}` (binary=False)
- When using Cross Entropy loss: Always use `{1, 0}` labels (binary=False)

## Citation

If you use this code in your research, please cite:
```
@article{hur2022quantum,
  title={Quantum convolutional neural network for classical data classification},
  author={Hur, Tak and Kim, Leeseok and Park, Daniel K},
  journal={Quantum Machine Intelligence},
  volume={4},
  number={1},
  pages={3},
  year={2022},
  publisher={Springer}
}
```

## License

This project is licensed under the terms of the license included in the repository.