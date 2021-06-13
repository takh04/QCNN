# Implementation of Quantum circuit training procedure
import QCNN_circuit
import Hierarchical_circuit
import numpy as np
import torch

def square_loss(labels, predictions):
    square_loss = torch.zeros_like(labels)
    for i in range(len(labels)):
        square_loss[i] = (labels[i] - predictions[i]).pow(2)
    return square_loss


steps = 150
learning_rate = 0.1
batch_size = 25

def circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit):
    if circuit == 'QCNN':
        total_params = U_params * 3
    elif circuit == 'Hierarchical':
        total_params = U_params * 7

    params = np.random.randn(total_params)
    params_torch = torch.tensor(params, requires_grad=True)
    opt = torch.optim.Adam([params_torch], lr=learning_rate)
    loss_history = []

    for it in range(steps):
        print(it)
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]

        X_batch_torch = torch.tensor(X_batch, device=torch.device('cuda'), requires_grad=False)
        Y_batch_torch = torch.tensor(Y_batch, device=torch.device('cuda'), requires_grad=False)

        def closure():
            opt.zero_grad()
            predictions = torch.stack([QCNN_circuit.QCNN(x, params_torch, U, U_params, embedding_type) for x in X_batch_torch])
            loss = torch.mean(square_loss(Y_batch_torch.float(), predictions.float()))
            current_loss = loss.detach().numpy().item()
            loss_history.append(current_loss)
            if it % 10 == 0:
                print("iteration: ", it, ", loss: ", current_loss)
            loss.backward()
            return loss

        opt.step(closure)

    return loss_history, params_torch
