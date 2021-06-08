# Implementation of Quantum circuit training procedure
import QCNN_circuit
import Hierarchical_circuit

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

def accuracy_test(predictions, labels, binary = True):
    if binary == True:
        acc = 0
        for l, p in zip(labels, predictions):
            if np.abs(l - p) < 1:
                acc = acc + 1
        return acc / len(labels)
    else binary == False:
        acc = 0
        for l, p in zip(labels, predictions):
            if np.abs(l - p) < 0.5:
                acc = acc + 1
        return acc / len(labels)

# Circuit training parameters
steps = 150
learning_rate = 0.1
batch_size = 25

def circuit_training(X_train, Y_train, U, U_params, embedding_type, circuit):
    if circuit == 'QCNN':
        total_params = U_params * 2 + 2 * 2 + 1
    elif circuit == 'Hierarchical':
        total_params = U_params * 7

    params = np.random.randn(total_params)
    opt = qml.NesterovMomentumOptimizer(learning_rate)
    loss_history = []

    for it in range(steps):
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]
        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit),
                                             params)
        loss_history.append(cost_new)
        if it % 10 == 0:
            print("iteration: ", it, " cost: ", cost_new)
    return loss_history, params