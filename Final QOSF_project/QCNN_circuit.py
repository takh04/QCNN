# Implementation of Quantum Convolutional Neural Network (QCNN) circuit structure.

import pennylane as qml
import unitary
import embedding
import torch

def conv_layer1(U, params):
    U(params, wires=[0, 7])
    for i in range(0, 8, 2):
        U(params, wires=[i, i + 1])
    for i in range(1, 7, 2):
        U(params, wires=[i, i + 1])
def conv_layer2(U, params):
    U(params, wires=[0, 2])
    U(params, wires=[4, 6])
    U(params, wires=[2, 4])
    U(params, wires=[0, 6])
def pooling_layer1(V, params):
    for i in range(0, 8, 2):
        V(params, wires = [i + 1, i])
def pooling_layer2(V, params):
    V(params, wires = [2,0])
    V(params, wires = [6,4])
def FullyConnectedLayer(U, params):
    U(params, wires = [0,4])


def QCNN_structure(U, params, U_params):

    param1 = params[0:U_params]
    param2 = params[U_params: 2 * U_params]
    param3 = params[2 * U_params: 3 * U_params ]

    conv_layer1(U, param1)
    #pooling_layer1(unitary.Pooling_ansatz1, param2)
    conv_layer2(U, param2)
    #pooling_layer2(unitary.Pooling_ansatz1, param4)
    FullyConnectedLayer(U, param3)



dev = qml.device('default.qubit', wires = 8)
@qml.qnode(dev, interface = 'torch', diff_method='parameter-shift')
def QCNN(X, params, U, U_params, embedding_type='Amplitude'):


    # Data Embedding
    embedding.data_embedding(X, embedding_type=embedding_type)

    # Quantum Convolutional Neural Network
    if U == 'U_TTN':
        QCNN_structure(unitary.U_TTN, params, U_params)
    elif U == 'U_5':
        QCNN_structure(unitary.U_5, params, U_params)
    elif U == 'U_6':
        QCNN_structure(unitary.U_6, params, U_params)
    elif U == 'U_9':
        QCNN_structure(unitary.U_9, params, U_params)
    elif U == 'U_13':
        QCNN_structure(unitary.U_13, params, U_params)
    elif U == 'U_14':
        QCNN_structure(unitary.U_14, params, U_params)
    elif U == 'U_15':
        QCNN_structure(unitary.U_15, params, U_params)
    elif U == 'U_SO4':
        QCNN_structure(unitary.U_SO4, params, U_params)
    elif U == 'U_SU4':
        QCNN_structure(unitary.U_SU4, params, U_params)

    else:
        print("Invalid Unitary Ansatze")
        return False

    return qml.expval(qml.PauliZ(4))
