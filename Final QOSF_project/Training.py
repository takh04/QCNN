# Implementation of Quantum circuit training procedure
import QCNN_circuit
import Hierarchical_circuit
import pennylane as qml
import numpy as np

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def cost(params, X, Y, U, U_params, embedding_type, circuit):
    if circuit == 'QCNN':
        predictions = [QCNN_circuit.QCNN(x, params, U, U_params, embedding_type) for x in X]
    elif circuit == 'Hierarchical':
        predictions = [Hierarchical_circuit.Hierarchical_classifier(x, params, U, U_params, embedding_type) for x in X]
    return square_loss(Y, predictions)


# Circuit training parameters

steps = 150
#steps = 5
learning_rate = 0.1
batch_size = 25

def circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit):
    if circuit == 'QCNN':
        total_params = U_params * 3
    elif circuit == 'Hierarchical':
        total_params = U_params * 7

    params = np.random.randn(total_params)
    opt = qml.NesterovMomentumOptimizer(learning_rate)
    loss_history = []

    for it in range(steps):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]

        #####
        # This is a Test Code for Hybrid Embedding
        # Test Code Block Starts
        if embedding_type == 'Hybrid16':
            for i in range(len(X_batch)):
                X1 = X_batch[i][:4]
                X2 = X_batch[i][4:8]
                X3 = X_batch[i][8:12]
                X4 = X_batch[i][12:16]
                norm_X1, norm_X2, norm_X3, norm_X4 = np.linalg.norm(X1), np.linalg.norm(X2), np.linalg.norm(X3), np.linalg.norm(X4)
                if norm_X1 == 0 or norm_X2 == 0 or norm_X3 == 0 or norm_X4 == 0 :
                    X_batch.remove(X_batch[i])

        elif embedding_type == 'Hybrid32':
            for i in range(len(X_batch)):
                X1 = X_batch[i][:2 ** 4]
                X2 = X_batch[i][2 ** 4:2 ** 5]
                norm_X1, norm_X2 = np.linalg.norm(X1), np.linalg.norm(X2)
                if norm_X1 == 0 or norm_X2 == 0:
                    X_batch.remove(X_batch[i])

        # Test Code Block Ends
        ####

        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit),
                                             params)
        loss_history.append(cost_new)
        if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)
    return loss_history, params