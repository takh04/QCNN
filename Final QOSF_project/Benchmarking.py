
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

    # save the result in the result.txt file
    f = open('Result/result.txt', 'w')

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
            elif Encoding == 'autoencoder32':
                Embedding = 'Hybrid32'
            elif Encoding == 'pca16':
                Embedding = 'Hybrid16'
            elif Encoding == 'autoencoder16':
                Embedding = 'Hybrid16'

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)


            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " " + Encoding)
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit)

            if circuit == 'QCNN':
                predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding) for x in X_test]
            elif circuit == 'Hierarchical':
                predictions = [Hierarchical_circuit.Hierarchical_classifier(x, trained_params, U, U_params, Embedding) for x in X_test]

            accuracy = accuracy_test(predictions, Y_test, binary)
            print("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))

            f.write("Loss History for " + circuit + " circuits, " + U + " " + Encoding)
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            f.write("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))
            f.write("\n")
            f.write("\n")

    f.close()

######## Test Code Block Start
# This is a test code block for benchmarking various Hybrid Embedding
def Benchmarking_hybrid_embedding16_1(dataset, classes, Encodings, Embeddings, circuit, binary=True):
    I = len(Embeddings)
    J = len(Encodings)
    U = 'U_SU4'
    U_params = 15

    # save the result in the result.txt file
    f = open('Result/result16_1.txt', 'w')

    for i in range(I):
        for j in range(J):
            Embedding = Embeddings[i]
            Encoding = Encodings[j]

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)

            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " " + Encoding + " " + Embedding)
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit)

            if circuit == 'QCNN':
                predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding) for x in X_test]
            elif circuit == 'Hierarchical':
                predictions = [Hierarchical_circuit.Hierarchical_classifier(x, trained_params, U, U_params, Embedding) for x in X_test]

            accuracy = accuracy_test(predictions, Y_test, binary)
            print("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))

            f.write("Loss History for " + circuit + " circuits, " + U + " " + Encoding)
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            f.write("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))
            f.write("\n")
            f.write("\n")
    f.close()


def Benchmarking_hybrid_embedding32_1(dataset, classes, Encodings, Embeddings, circuit, binary=True):
    I = len(Embeddings)
    J = len(Encodings)
    U = 'U_SU4'
    U_params = 15

    # save the result in the result.txt file
    f = open('Result/result32_1.txt', 'w')

    for i in range(I):
        for j in range(J):
            Embedding = Embeddings[i]
            Encoding = Encodings[j]

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)

            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " " + Encoding + " " + Embedding)
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit)

            if circuit == 'QCNN':
                predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding) for x in X_test]
            elif circuit == 'Hierarchical':
                predictions = [Hierarchical_circuit.Hierarchical_classifier(x, trained_params, U, U_params, Embedding)
                               for x in X_test]

            accuracy = accuracy_test(predictions, Y_test, binary)
            print("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))

            f.write("Loss History for " + circuit + " circuits, " + U + " " + Encoding)
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            f.write("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))
            f.write("\n")
            f.write("\n")
    f.close()

def Benchmarking_hybrid_embedding16_2(dataset, classes, Encodings, Embeddings, circuit, binary=True):
    I = len(Embeddings)
    J = len(Encodings)
    U = 'U_SU4'
    U_params = 15

    # save the result in the result.txt file
    f = open('Result/result16_2.txt', 'w')

    for i in range(I):
        for j in range(J):
            Embedding = Embeddings[i]
            Encoding = Encodings[j]

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)

            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " " + Encoding + " " + Embedding)
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit)

            if circuit == 'QCNN':
                predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding) for x in X_test]
            elif circuit == 'Hierarchical':
                predictions = [Hierarchical_circuit.Hierarchical_classifier(x, trained_params, U, U_params, Embedding) for x in X_test]

            accuracy = accuracy_test(predictions, Y_test, binary)
            print("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))

            f.write("Loss History for " + circuit + " circuits, " + U + " " + Encoding)
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            f.write("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))
            f.write("\n")
            f.write("\n")
    f.close()


def Benchmarking_hybrid_embedding32_2(dataset, classes, Encodings, Embeddings, circuit, binary=True):
    I = len(Embeddings)
    J = len(Encodings)
    U = 'U_SU4'
    U_params = 15

    # save the result in the result.txt file
    f = open('Result/result32_2.txt', 'w')

    for i in range(I):
        for j in range(J):
            Embedding = Embeddings[i]
            Encoding = Encodings[j]

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)

            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " " + Encoding + " " + Embedding)
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit)

            if circuit == 'QCNN':
                predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding) for x in X_test]
            elif circuit == 'Hierarchical':
                predictions = [Hierarchical_circuit.Hierarchical_classifier(x, trained_params, U, U_params, Embedding)
                               for x in X_test]

            accuracy = accuracy_test(predictions, Y_test, binary)
            print("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))

            f.write("Loss History for " + circuit + " circuits, " + U + " " + Encoding)
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            f.write("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))
            f.write("\n")
            f.write("\n")
    f.close()

def Benchmarking_hybrid_embedding16_3(dataset, classes, Encodings, Embeddings, circuit, binary=True):
    I = len(Embeddings)
    J = len(Encodings)
    U = 'U_SU4'
    U_params = 15

    # save the result in the result.txt file
    f = open('Result/result16_3.txt', 'w')

    for i in range(I):
        for j in range(J):
            Embedding = Embeddings[i]
            Encoding = Encodings[j]

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)

            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " " + Encoding + " " + Embedding)
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit)

            if circuit == 'QCNN':
                predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding) for x in X_test]
            elif circuit == 'Hierarchical':
                predictions = [Hierarchical_circuit.Hierarchical_classifier(x, trained_params, U, U_params, Embedding) for x in X_test]

            accuracy = accuracy_test(predictions, Y_test, binary)
            print("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))

            f.write("Loss History for " + circuit + " circuits, " + U + " " + Encoding)
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            f.write("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))
            f.write("\n")
            f.write("\n")
    f.close()


def Benchmarking_hybrid_embedding32_3(dataset, classes, Encodings, Embeddings, circuit, binary=True):
    I = len(Embeddings)
    J = len(Encodings)
    U = 'U_SU4'
    U_params = 15

    # save the result in the result.txt file
    f = open('Result/result32_3.txt', 'w')

    for i in range(I):
        for j in range(J):
            Embedding = Embeddings[i]
            Encoding = Encodings[j]

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)

            print("\n")
            print("Loss History for " + circuit + " circuits, " + U + " " + Encoding + " " + Embedding)
            loss_history, trained_params = Training.circuit_training(X_train, Y_train, U, U_params, Embedding, circuit)

            if circuit == 'QCNN':
                predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding) for x in X_test]
            elif circuit == 'Hierarchical':
                predictions = [Hierarchical_circuit.Hierarchical_classifier(x, trained_params, U, U_params, Embedding)
                               for x in X_test]

            accuracy = accuracy_test(predictions, Y_test, binary)
            print("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))

            f.write("Loss History for " + circuit + " circuits, " + U + " " + Encoding)
            f.write("\n")
            f.write(str(loss_history))
            f.write("\n")
            f.write("Accuracy for " + U + " " + Encoding + " :" + str(accuracy))
            f.write("\n")
            f.write("\n")
    f.close()
######## Test Code Block Ends




#### Test Code Block Starts
import numpy as np
def Data_norm(dataset, classes, Encodings, binary=True):
    J = len(Encodings)

    f = open('Result/data_norm.txt', 'w')
    for j in range(J):
        Encoding = Encodings[j]

        X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)

        if Encoding ==  'pca32' or Encoding == 'autoencoder32':
            f.write("Norm of 32 bit Hybrid Embedding")
            f.write("\n")
            if Encoding == 'pca32':
                f.write("Norm for pca32")
            elif Encoding == 'autoencoder32':
                f.write("Norm for autoencoder32")

            f.write("\n")
            for i in range(200):
                index = np.random.randint(0, len(X_train))
                X = X_train[index]

                X1 = X[:2 ** 4]
                X2 = X[2 ** 4:2 ** 5]
                norm_X1, norm_X2 = np.linalg.norm(X1), np.linalg.norm(X2)
                f.write(str(norm_X1))
                f.write("\n")
                f.write(str(norm_X2))
                f.write("\n")

                f.write("\n")

            f.write("\n")
            f.write("\n")
        elif Encoding == 'pca16' or Encoding == 'autoencoder16':
            f.write("Norm of 16 bit Hybrid Embedding")
            f.write("\n")
            if Encoding == 'pca16':
                f.write("Norm for pca16")
            elif Encoding == 'autoencoder16':
                f.write("Norm for autoencoder16")
            f.write("\n")
            for i in range(200):
                index = np.random.randint(0, len(X_train))
                X = X_train[index]

                X1 = X[:4]
                X2 = X[4:8]
                X3 = X[8:12]
                X4 = X[12:16]
                norm_X1, norm_X2, norm_X3, norm_X4 = np.linalg.norm(X1), np.linalg.norm(X2), np.linalg.norm(
                    X3), np.linalg.norm(X4)
                f.write(str(norm_X1))
                f.write("\n")
                f.write(str(norm_X2))
                f.write("\n")
                f.write(str(norm_X3))
                f.write("\n")
                f.write(str(norm_X4))
                f.write("\n")

                f.write("\n")

    f.close()

#### Test Code Block Ends