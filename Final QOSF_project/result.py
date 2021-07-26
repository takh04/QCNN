# This generates the results of the bechmarking code

import Benchmarking


#########
# Here are possible combinations of benchmarking user could try.
# Unitaries = [U_TTN, U_5, U_6, U_9, U_13, U_14, U_15, U_SO4, U_SU4]
# U_num_params = [2, 10, 10, 2, 6, 6, 4, 6, 15]
# dataset = 'mnist' , 'fashion_mnist'
# circuit = 'QCNN', 'Hierarchical'
#########

Unitaries = ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_without_pooling']
U_num_params = [2, 10, 10, 2, 6, 6, 4, 6, 15, 15]
Encodings = ['pca30-3', 'autoencoder30-3', 'pca32-3', 'autoencoder32-3']
dataset = 'mnist'
circuit = 'QCNN'
classes = [0,1]

for i in range(5):
    Benchmarking.Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit)

