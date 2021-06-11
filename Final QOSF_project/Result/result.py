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
Encodings = ['pca32', 'autoencoder32', 'pca16', 'autoencoder16']
dataset = 'mnist'
circuit = 'QCNN'
classes = [0,1]

#Benchmarking.Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit)

dataset = 'mnist'
classes = [0,1]
Encodings = ['pca16', 'autoencoder16']
Embeddings = ['Hybrid16-1', 'Hybrid16-2', 'Hybrid16-3', 'Hybrid16-4']
circuit = 'QCNN'
#Benchmarking.Benchmarking_hybrid_embedding16_1(dataset, classes, Encodings, Embeddings, circuit, binary = True)
#Benchmarking.Benchmarking_hybrid_embedding16_2(dataset, classes, Encodings, Embeddings, circuit, binary = True)
#Benchmarking.Benchmarking_hybrid_embedding16_3(dataset, classes, Encodings, Embeddings, circuit, binary = True)

dataset = 'mnist'
classes = [0,1]
Encodings = ['pca32', 'autoencoder32']
Embeddings = ['Hybrid32-1', 'Hybrid32-2', 'Hybrid32-3', 'Hybrid32-4']
circuit = 'QCNN'
#Benchmarking.Benchmarking_hybrid_embedding32_1(dataset, classes, Encodings, Embeddings, circuit, binary = True)
#Benchmarking.Benchmarking_hybrid_embedding32_2(dataset, classes, Encodings, Embeddings, circuit, binary = True)
#Benchmarking.Benchmarking_hybrid_embedding32_3(dataset, classes, Encodings, Embeddings, circuit, binary = True)

dataset = 'mnist'
classes = [0,1]
Encodings = ['pca16', 'autoencoder16', 'pca32', 'autoencoder32']
Benchmarking.Data_norm(dataset, classes, Encodings, binary=True)