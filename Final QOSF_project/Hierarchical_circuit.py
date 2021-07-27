# Implementaion of Hierarchical Quantum Classifier Structure.
import pennylane as qml
import unitary
import embedding

dev_TTN = qml.device('default.qubit', wires=8)

def Hierarchical_structure(U, params, U_params):
    param1 = params[0 * U_params:1 * U_params]
    param2 = params[1 * U_params:2 * U_params]
    param3 = params[2 * U_params:3 * U_params]
    param4 = params[3 * U_params:4 * U_params]
    param5 = params[4 * U_params:5 * U_params]
    param6 = params[5 * U_params:6 * U_params]
    param7 = params[6 * U_params:7 * U_params]

    # 1st Layer
    U(param1, wires=[0, 1])
    U(param2, wires=[2, 3])
    U(param3, wires=[4, 5])
    U(param4, wires=[6, 7])
    # 2nd Layer
    U(param5, wires=[1, 3])
    U(param6, wires=[5, 7])
    # 3rd Layer
    U(param7, wires=[3, 7])



@qml.qnode(dev_TTN)
def Hierarchical_classifier(X, params, U, U_params, embedding_type='Amplitude', cost_fn='cross_entropy'):
    embedding.data_embedding(X, embedding_type=embedding_type)
    if U == 'U_TTN':
        Hierarchical_structure(unitary.U_TTN, params, U_params)
    elif U == 'U_5':
        Hierarchical_structure(unitary.U_5, params, U_params)
    elif U == 'U_6':
        Hierarchical_structure(unitary.U_6, params, U_params)
    elif U == 'U_9':
        Hierarchical_structure(unitary.U_9, params, U_params)
    elif U == 'U_13':
        Hierarchical_structure(unitary.U_13, params, U_params)
    elif U == 'U_14':
        Hierarchical_structure(unitary.U_14, params, U_params)
    elif U == 'U_15':
        Hierarchical_structure(unitary.U_15, params, U_params)
    elif U == 'U_SO4':
        Hierarchical_structure(unitary.U_SO4, params, U_params)
    elif U == 'U_SU4':
        Hierarchical_structure(unitary.U_SU4, params, U_params)
    else:
        print("Invalid Unitary Ansatz")
        return False
    if cost_fn == 'mse':
        result = qml.expval(qml.PauliZ(7))
    elif cost_fn == 'cross_entropy':
        result = qml.probs(wires=7)
    return result
