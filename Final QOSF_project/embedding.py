# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# and Hierarchical Quantum Classifier circuit.
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from pennylane.templates.state_preparations import MottonenStatePreparation


def data_embedding(X, embedding_type='Amplitude'):
    if embedding_type == 'Amplitude':
        AmplitudeEmbedding(X, wires=range(8), normalize=True)
    elif embedding_type == 'Angle':
        AngleEmbedding(X, wires=range(8), rotation='Y')


    #############
    # This is a test code block for Hybrid Embedding Scheme
    # Test Code Block Starts
    elif embedding_type == 'Hybrid':
        X1 = X[:2 ** 4]
        X2 = X[2 ** 4:2 ** 5]
        norm_X1, norm_X2 = np.linalg.norm(X1), np.linalg.norm(X2)
        X1, X2 = X1 / norm_X1, X2 / norm_X2

        MottonenStatePreparation(X1, wires=range(4))
        MottonenStatePreparation(X2, wires=range(4, 8))

    # Hybrid Embedding for 16 classical data
    elif embedding_type == 'Hybrid16':
        X1 = X[:4]
        X2 = X[4:8]
        X3 = X[8:12]
        X4 = X[12:16]
        norm_X1, norm_X2, norm_X3, norm_X4 = np.linalg.norm(X1), np.linalg.norm(X2), np.linalg.norm(X3), np.linalg.norm(
            X4)
        X1, X2, X3, X4 = X1 / norm_X1, X2 / norm_X2, X3 / norm_X3, X4 / norm_X4

        MottonenStatePreparation(X1, wires=range(2))
        MottonenStatePreparation(X2, wires=range(2, 4))
        MottonenStatePreparation(X3, wires=range(4, 6))
        MottonenStatePreparation(X4, wires=range(6, 8))
    # Test Code Block Ends

