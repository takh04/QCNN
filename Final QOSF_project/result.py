# This generates the results of the bechmarking code

import Benchmarking

U = ['U_TTN']
U_params = [2]
dataset = 'mnist'
Encoding = ['resize256']
circuit = 'QCNN'
classes = [0,1]

Benchmarking.Benchmarking(dataset, classes, U, U_params, Encoding, circuit)