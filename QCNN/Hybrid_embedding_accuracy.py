# This module implements the measurement method of the entangling capability
import pennylane as qml
import QCNN_circuit
import unitary
import embedding
import numpy as np
import data
import Training
from Benchmarking import Encoding_to_Embedding, accuracy_test


dev = qml.device('default.qubit', wires=8)

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

def Benchmarking_Hybrid_Accuracy(dataset, classes, Unitary, U_num_param, Encodings, circuit, binary=True):
    U = Unitary
    U_params = U_num_param
    J = len(Encodings)
    best_trained_params_list = []

    for j in range(J):
        Encoding = Encodings[j]
        Embedding = Encoding_to_Embedding(Encoding)
        f = open('Result/Hybrid_result_' + str(Encoding) + '.txt', 'a')
        trained_params_list = []
        accuracy_list = []
        for n in range(5):

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)

            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " " + Encoding)

            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit)
            predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding) for x in X_test]
            accuracy = accuracy_test(predictions, Y_test, binary)
            print("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))

            trained_params_list.append(trained_params)
            accuracy_list.append(accuracy)

            f.write("Trained Parameters: \n")
            f.write(str(trained_params))
            f.write("\n")
            f.write("Accuracy: \n")
            f.write(str(accuracy))
            f.write("\n")

        index = accuracy_list.index(max(accuracy_list))
        best_trained_params_list.append(trained_params_list[index])

    f.close()
    return best_trained_params_list

def Benchmarking_Hybrid_Entanglement(dataset, classes, Encodings, N_samples, best_trained_params_list):
    for i in range(len(Encodings)):
        Encoding = Encodings[i]
        print("Processing " + str(Encoding) + ".....\n")
        Embedding = Encoding_to_Embedding(Encoding)
        best_trained_params = best_trained_params_list[i]
        best_trained_params = best_trained_params[:15]

        X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset=dataset, classes=classes, feature_reduction=Encoding, binary=True)
        random_index = np.random.randint(0, len(X_test), N_samples)
        X_test = X_test[random_index]

        entanglement_measure = [Meyer_Wallach(X, best_trained_params, Embedding) for X in X_test]
        mean_entanglement_measure = np.mean(entanglement_measure)
        stdev_entanglement_measure = np.std(entanglement_measure)

        f = open('Result/Hybrid_result_' + str(Encoding) + '.txt', 'a')
        f.write("\n")
        f.write("Entanglement measure Mean: ")
        f.write(str(mean_entanglement_measure))
        f.write("\n")
        f.write("Entanglement measure Standard Deviation: ")
        f.write(str(stdev_entanglement_measure))
        f.write("\n")
        f.close()

dataset = 'mnist'
classes = [0,1]
Unitary = 'U_SU4'
U_num_param = 15
circuit = 'QCNN'
N_samples = 1000


#Encodings = ['pca30-1', 'autoencoder30-1', 'pca12-1', 'autoencoder12-1', 'pca32-1', 'autoencoder32-1', 'pca16-1', 'autoencoder16-1',
#             'pca30-2', 'autoencoder30-2', 'pca12-2', 'autoencoder12-2', 'pca32-2', 'autoencoder32-2', 'pca16-2', 'autoencoder16-2',
#             'pca30-3', 'autoencoder30-3', 'pca12-3', 'autoencoder12-3', 'pca32-3', 'autoencoder32-3', 'pca16-3', 'autoencoder16-3',
#             'pca30-4', 'autoencoder30-4', 'pca12-4', 'autoencoder12-4', 'pca32-4', 'autoencoder32-4', 'pca16-4', 'autoencoder16-4']

Encodings = ['autoencoder30-3']
#best_trained_params_list = Benchmarking_Hybrid_Accuracy(dataset, classes, Unitary, U_num_param, Encodings, circuit, binary=True)
#Benchmarking_Hybrid_Entanglement(dataset, classes, Encodings, N_samples, best_trained_params_list)