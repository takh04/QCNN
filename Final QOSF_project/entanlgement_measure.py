# This module implements the measurement method of the entangling capability
import pennylane as qml
import QCNN_circuit
import unitary
import embedding
import numpy as np
import data
import Training
import Benchmarking

dev = qml.device('default.qubit', wires=8)

qml.enable_tape()
@qml.qnode(dev)
def QCNN_partial_trace(X, params, embedding_type='Angular-Hybrid4', qubit_index=0):
    embedding.data_embedding(X, embedding_type)
    QCNN_circuit.conv_layer1(unitary.U_SU4, params)
    return qml.density_matrix(wires=qubit_index)

def Meyer_Wallach(X, params, embedding_type):
   n = 8
   measure = 0
   for j in range(n):
        rho = QCNN_partial_trace(X, params, embedding_type, qubit_index=j)
        rho_squared = np.matmul(rho, rho)
        rho_squared_traced = np.matrix.trace(rho_squared)
        measure = measure + 1/2 * (1 - rho_squared_traced)
   return measure * 4 / n


dataset = 'mnist'
classes = [0,1]
encodings32 = ['autoencoder32', 'pca32']
embeddings32 = ['Hybrid32-1', 'Hybrid32-2', 'Hybrid32-3', 'Hybrid32-4']

for i in range(len(encodings32)):
    for j in range(len(embeddings32)):
        for k in range(3):
            encoding = encodings32[i]
            embedding = embeddings32[j]

            f = open("Result/entanglement_measure/" + encoding + "_"+ embedding + "_" + "trained_params.txt", 'a')

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset=dataset, classes=classes, feature_reduction=encoding, binary=True)
            X_test, Y_test = X_test[:10], Y_test[:10]
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, 'U_SU4', 15, embedding, 'QCNN')
            predictions = [QCNN_circuit.QCNN(x, trained_params, 'U_SU4', 15, embedding) for x in X_test]
            accuracy = Benchmarking.accuracy_test(predictions, Y_test, True)

            f.write(str(trained_params))
            f.write("\n")
            f.write(str(accuracy))
            f.write("\n")
            f.close()

#random_index = np.random.randint(0, len(X_test), 1000)
#X_test = X_test[random_index]
#mean_entanglement_measure = np.mean([Meyer_Wallach(X, trained_params, 'Hybrid32-1') for X in X_test])
#print(mean_entanglement_measure)


