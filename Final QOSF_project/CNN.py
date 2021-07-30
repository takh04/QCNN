import data
import numpy as np
import torch
import torch.nn as nn

def get_n_params(model):
    np=0
    for p in list(model.parameters()):
        np += p.nelement()
    return np

def accuracy_test(predictions, labels):
    acc = 0
    for (p,l) in zip(predictions, labels):
        if p[0] >= p[1]:
            pred = 0
        else:
            pred = 1

        if pred == l:
            acc = acc + 1
    acc = acc / len(labels)
    return acc

steps = 200
n_feature = 2
batch_size = 25
def Benchmarking_CNN(dataset, classes, Encodings, Encodings_size, binary, optimizer):
    for i in range(len(Encodings)):
        Encoding = Encodings[i]
        input_size = Encodings_size[i]
        final_layer_size = int(input_size / 4)
        X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes, feature_reduction=Encoding, binary=binary)

        CNN = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_feature, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=n_feature, out_channels=n_feature, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(n_feature * final_layer_size, 2)
        )

        loss_history = []
        for it in range(steps):
            batch_idx = np.random.randint(0, len(X_train), batch_size)
            X_train_batch = np.array([X_train[i] for i in batch_idx])
            Y_train_batch = np.array([Y_train[i] for i in batch_idx])

            X_train_batch_torch = torch.tensor(X_train_batch, dtype=torch.float32)
            X_train_batch_torch.resize_(batch_size, 1, input_size)
            Y_train_batch_torch = torch.tensor(Y_train_batch, dtype=torch.long)

            criterion = nn.CrossEntropyLoss()
            if optimizer == 'adam':
                opt = torch.optim.Adam(CNN.parameters(), lr=0.01, betas=(0.9, 0.999))
            elif optimizer == 'nesterov':
                opt = torch.optim.SGD(CNN.parameters(), lr=0.01, momentum=0.9, nesterov=True)

            Y_pred_batch_torch = CNN(X_train_batch_torch)

            loss = criterion(Y_pred_batch_torch, Y_train_batch_torch)
            loss_history.append(loss.item())
            if it % 10 == 0:
                print("[iteration]: %i, [LOSS]: %.6f" % (it, loss.item()))

            opt.zero_grad()
            loss.backward()
            opt.step()

            X_test_torch = torch.tensor(X_test, dtype=torch.float32)
            X_test_torch.resize_(len(X_test), 1, input_size)
            Y_pred = CNN(X_test_torch).detach().numpy()
            accuracy = accuracy_test(Y_pred, Y_test)
            N_params = get_n_params(CNN)


        f = open('Result/result_CNN.txt', 'a')
        f.write("Loss History for CNN with " + str(Encoding) + ":" )
        f.write("\n")
        f.write(str(loss_history))
        f.write("\n")
        f.write("Accuracy for CNN with " + str(Encoding) + " " +optimizer + ": " + str(accuracy))
        f.write("\n")
        f.write("Number of Parameters used to train CNN: " + str(N_params))
        f.write("\n")
        f.write("\n")

    f.close()

steps = 200
dataset = 'fashion_mnist'
classes = [0,1]
binary = False
Encodings = ['pca8', 'autoencoder8', 'pca16-compact', 'autoencoder16-compact']
Encodings_size = [8, 8, 16, 16]

for i in range(5):
    Benchmarking_CNN(dataset=dataset, classes=classes, Encodings=Encodings, Encodings_size=Encodings_size,
                     binary=binary, optimizer='adam')
    #Benchmarking_CNN(dataset=dataset, classes=classes, Encodings=Encodings, Encodings_size=Encodings_size,
    #                 binary=binary, optimizer='nesterov')
