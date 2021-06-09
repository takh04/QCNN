# This generates the results of the bechmarking code

import Benchmarking

U = ['U_TTN']
U_params = [2]
dataset = 'MNIST'
Encoding = ['resize256']
circuit = 'QCNN'

Benchmarking.Benchmarking(dataset, U, U_params, Encoding, circuit)