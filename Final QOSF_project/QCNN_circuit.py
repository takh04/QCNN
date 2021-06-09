# Implementation of Quantum Convolutional Neural Network (QCNN) circuit structure.

import pennylane as qml
import unitary
import embedding

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
        V(params, wires = [i, i + 1])
def pooling_layer2(V, params):
    V(params, wires = [0,2])
    V(params, wires = [4,6])


def QCNN_structure(U, params, U_params):

    param1 = params[0:U_params]
    param2 = params[U_params:U_params + 2]
    param3 = params[U_params + 2: 2 * U_params + 2]
    param4 = params[2 * U_params + 2: 2 * U_params + 4]
    param5 = params[2 * U_params + 4]

    conv_layer1(U, param1)
    pooling_layer1(unitary.Pooling_ansatz1, param2)
    conv_layer2(U, param3)
    pooling_layer2(unitary.Pooling_ansatz1, param4)
    unitary.FullyConnectedLayer(param5, wires = [0,4])



dev = qml.device('default.qubit', wires = 8)
@qml.qnode(dev)
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

    return qml.expval(qml.PauliZ(4))
