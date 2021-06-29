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


def Benchmarking_Hybrid_Accuracy(dataset, classes, Encodings, Embeddings, N):
    I = len(Encodings)
    J = len(Embeddings)
    Best_trained_params_list = []
    for i in range(I):
        for j in range(J):
            trained_params_list = []
            accuracy_list = []
            for k in range(N):
                Encoding = Encodings[i]
                Embedding = Embeddings[j]

                f = open("Result/entanglement_measure/" + Encoding + "_" + Embedding + ".txt", 'a')

                X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset=dataset, classes=classes,
                                                                              feature_reduction=Encoding, binary=True)

                loss_history, trained_params = Training.circuit_training(X_train, Y_train, 'U_SU4', 15, embedding, 'QCNN')
                predictions = [QCNN_circuit.QCNN(x, trained_params, 'U_SU4', 15, embedding) for x in X_test]
                accuracy = Benchmarking.accuracy_test(predictions, Y_test, True)

                trained_params_list.append(trained_params)
                accuracy_list.append(accuracy)

                f.write("Trained Paramameters: \n")
                f.write(str(trained_params))
                f.write("\n")
                f.write("Accuracy: ")
                f.write(str(accuracy))
                f.write("\n")
                f.close()

            Best_accuracy = max(accuracy_list)
            for l in range(N):
                accuracy = accuracy_list[l]
                trained_params = trained_params_list[l]
                if accuracy == Best_accuracy:
                    Best_trained_params = trained_params

            Best_trained_params_list.append(Best_trained_params)

    Best_trained_params_list = np.array(Best_trained_params_list)
    Best_trained_params_list = np.reshape(Best_trained_params_list, (I,J,45))
    return Best_trained_params_list


def Benchmarking_Hybrid_Entanglement(dataset, classes, Encodings, Embeddings, N_samples, Best_trained_params_list):
    for i in range(len(Encodings)):
        for j in range(len(Embeddings)):
            Encoding = Encodings[i]
            Embedding = Embeddings[j]

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset=dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=True)
            random_index = np.random.randint(0, len(X_test), N_samples)
            X_test = X_test[random_index]
            Best_trained_params = Best_trained_params_list[i][j]
            Best_trained_params = Best_trained_params[:15]
            print(Best_trained_params)

            entanglement_measure = [Meyer_Wallach(X, Best_trained_params, 'Hybrid32-1') for X in X_test]
            mean_entanglement_measure = np.mean(entanglement_measure)
            stdev_entanglement_measure = np.std(entanglement_measure)

            f = open("Result/entanglement_measure/" + Encoding + "_" + Embedding + ".txt", 'a')
            f.write("Entanglement measure Mean: ")
            f.write(str(mean_entanglement_measure))
            f.write("\n")
            f.write("Entanglement measure Standard Deviation: ")
            f.write(str(stdev_entanglement_measure))
            f.write("\n")
            f.close()




dataset = 'mnist'
classes = [0,1]
N = 5
N_samples = 1000

# Amplitude Hybrid Test 4 qubits
#Encodings = ['autoencoder32', 'pca32']
#Embeddings = ['Amplitude-Hybrid4-1', 'Amplitude-Hybrid4-2', 'Amplitude-Hybrid4-3', 'Amplitude-Hybrid4-4']

# Angular Hybrid Test 4 qubits
Encodings = ['autoencoder30', 'pca30']
Embeddings = ['Angular-Hybrid4-1', 'Angular-Hybrid4-2', 'Angular-Hybrid4-3', 'Angular-Hybrid4-4']
Best_trained_params_list = Benchmarking_Hybrid_Accuracy(dataset, classes, Encodings, Embeddings, N)
Benchmarking_Hybrid_Entanglement(dataset, classes, Encodings, Embeddings, N_samples, Best_trained_params_list)

# Amplitude Hybrid Test 2 qubits
#Encodings = ['autoencoder16', 'pca16']
#Embeddings = ['Amplitude-Hybrid2-1', 'Amplitude-Hybrid2-2', 'Amplitude-Hybrid2-3', 'Amplitude-Hybrid2-4']

# Angular Hybrid Test 2 qubits
Encodings = ['autoencoder12', 'pca12']
Embeddings = ['Angular-Hybrid2-1', 'Angular-Hybrid2-2', 'Angular-Hybrid2-3', 'Angular-Hybrid2-4']

#Run the code
Best_trained_params_list = Benchmarking_Hybrid_Accuracy(dataset, classes, Encodings, Embeddings, N)
Benchmarking_Hybrid_Entanglement(dataset, classes, Encodings, Embeddings, N_samples, Best_trained_params_list)

