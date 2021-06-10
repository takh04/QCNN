
import data
import Training
import QCNN_circuit
import Hierarchical_circuit
import numpy as np

def accuracy_test(predictions, labels, binary = True):
    if binary == True:
        acc = 0
        for l, p in zip(labels, predictions):
            if np.abs(l - p) < 1:
                acc = acc + 1
        return acc / len(labels)

    else:
        acc = 0
        for l, p in zip(labels, predictions):
            if np.abs(l - p) < 0.5:
                acc = acc + 1
        return acc / len(labels)


def Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit, binary=True):
    I = len(Unitaries)
    J = len(Encodings)

    for i in range(I):
        for j in range(J):
            U = Unitaries[i]
            U_params = U_num_params[i]
            Encoding = Encodings[j]
            if Encoding == 'resize256':
                Embedding = 'Amplitude'
            elif Encoding == 'pca8':
                Embedding = 'Angle'
            elif Encoding == 'autoencoder8':
                Embedding = 'Angle'
            elif Encoding == 'pca32':
                Embedding = 'Hybrid32'
            elif Encoding = 'autoencoder32':
                Embedding = 'Hybrid32'
            elif Encoding = 'pca16':
                Embedding = 'Hybrid16'
            elif Encoding = 'autoencoder16':
                Embedding = 'Hybrid16'

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)

            # Test Code Ends
            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " " + Encoding)
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit)

            if circuit == 'QCNN':
                predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding) for x in X_test]
            elif circuit == 'Hierarchical':
                predictions = [Hierarchical_circuit.Hierarchical_classifier(x, trained_params, U, U_params, Embedding) for x in X_test]

            accuracy = accuracy_test(predictions, Y_test, binary)
            print("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))

            f = open('result.txt', 'w')
            f.write("Loss History for " + circuit + " circuits, " + U + " " + Encoding)
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            f.write("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))
            f.write("\n")
            f.write("\n")
            f.close()

