# This generates the results of the bechmarking code

import Benchmarking


#########
# Here are possible combinations of benchmarking user could try.
# Unitaries = [U_TTN, U_5, U_6, U_9, U_13, U_14, U_15, U_SO4, U_SU4]
# U_num_params = [2, 10, 10, 2, 6, 6, 4, 6, 15]
# dataset = 'mnist' , 'fashion_mnist'
# circuit = 'QCNN', 'Hierarchical'
#########

Unitaries = ['U_SU4']
U_num_params = [15]
Encodings = ['pca32-3', 'autoencoder32-3']
dataset = 'fashion_mnist'
classes = [0,1]
binary = False
cost_fn = 'cross_entropy'

Benchmarking.Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit='QCNN', cost_fn= cost_fn, binary=binary)
#Benchmarking.Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit='Hierarchical', cost_fn=cost_fn, binary=binary)

