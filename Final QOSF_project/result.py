# This generates the results of the bechmarking code

import Benchmarking
import qiskit

#########
# Here are possible combinations of benchmarking user could try.
# Unitaries = [U_TTN, U_5, U_6, U_9, U_13, U_14, U_15, U_SO4, U_SU4]
# U_num_params = [2, 10, 10, 2, 6, 6, 4, 6, 15]
# dataset = 'mnist' , 'fashion_mnist'
# circuit = 'QCNN', 'Hierarchical'
#########

Unitaries = ['U_SU4']
U_num_params = [15]
Encodings = ['pca8']
dataset = 'mnist'
circuit = 'QCNN'
classes = [0,1]


Benchmarking.Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit)

