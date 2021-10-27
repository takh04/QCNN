# Implementation of Quantum circuit training procedure
import QCNN_circuit
import Hierarchical_circuit
import pennylane as qml
import numpy as np
import autograd.numpy as anp

def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy

    return -1 * loss

def cost(params, X, Y, U, U_params, embedding_type, circuit, cost_fn):
    if circuit == 'QCNN':
        predictions = [QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]
    elif circuit == 'Hierarchical':
        predictions = [Hierarchical_circuit.Hierarchical_classifier(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]

    if cost_fn == 'mse':
        loss = square_loss(Y, predictions)
    elif cost_fn == 'cross_entropy':
        loss = cross_entropy(Y, predictions)
    return loss

# Circuit training parameters

steps = 200
learning_rate = 0.01


batch_size = 25

def circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit, cost_fn):
    if circuit == 'QCNN':
        total_params = U_params * 3 + 2 * 3
    elif circuit == 'Hierarchical':
        total_params = U_params * 7

    params = np.random.randn(total_params)
    opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    loss_history = []

    for it in range(steps):

        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]

        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn),
                                                     params)
        loss_history.append(cost_new)
        if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)

    return loss_history, params


