import matplotlib.pyplot as plt
from Benchmarking import Encoding_to_Embedding
from data import data_load_and_process
from Training import circuit_training
from CNN import CNN_training


U = 'U_SU4'
U_params = 15
dataset = 'mnist'
classes = [0,1]
Encodings = ['resize256', 'pca8', 'autoencoder8', 'pca16-compact', 'autoencoder16-compact']
Encodings_size = [256, 8, 8, 16, 16]

def plot_lost_history(Encodings, Encodings_size):
    for i in range(len(Encodings)):
        Encoding = Encodings[i]
        input_size = Encodings_size[i]
        Embedding = Encoding_to_Embedding(Encoding)

        #CNN - binary must be False for CrossEntropyLoss
        X_train, X_test, Y_train, Y_test = data_load_and_process(dataset, classes=classes,
                                                                      feature_reduction=Encoding, binary=False)
        optimizer = 'adam'
        loss_history_CNN, accuracy, N_params = CNN_training(X_train, X_test, Y_train, Y_test, input_size, optimizer, steps=200, n_feature=3, batch_size=25)

        X_train, X_test, Y_train, Y_test = data_load_and_process(dataset, classes=classes,
                                                                      feature_reduction=Encoding, binary=True)
        # QCNN
        loss_history_QCNN, trained_params_QCNN = circuit_training(X_train, Y_train, U, U_params, Embedding, circuit='QCNN')

        # TTN
        loss_history_TTN, trained_params_TTN = circuit_training(X_train, Y_train, U, U_params, Embedding, circuit='Hierarchical')

        plt.plot(loss_history_QCNN)
        plt.plot(loss_history_CNN)
        plt.plot(loss_history_TTN)
        plt.title('QCNN vs CNN vs TTN with ' + str(Encoding))
        plt.ylabel('loss')
        plt.xlabel('iterations')
        plt.legend(['QCNN', 'CNN', 'TTN'], loc='upper left')
        plt.show()

plot_lost_history(Encodings, Encodings_size)

