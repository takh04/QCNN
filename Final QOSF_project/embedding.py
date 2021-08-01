# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# and Hierarchical Quantum Classifier circuit.
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from pennylane.templates.state_preparations import MottonenStatePreparation
import numpy as np
from Angular_hybrid import Angular_Hybrid_4, Angular_Hybrid_2
def data_embedding(X, embedding_type='Amplitude'):
    if embedding_type == 'Amplitude':
        AmplitudeEmbedding(X, wires=range(8), normalize=True)
    elif embedding_type == 'Angle':
        AngleEmbedding(X, wires=range(8), rotation='Y')
    elif embedding_type == 'Angle-compact':
        AngleEmbedding(X[:8], wires=range(8), rotation='X')
        AngleEmbedding(X[8:16], wires=range(8), rotation='Y')

    # Hybrid Direct Embedding (HDE)
    elif embedding_type == 'Amplitude-Hybrid4-1' or embedding_type == 'Amplitude-Hybrid4-2' or \
            embedding_type == 'Amplitude-Hybrid4-3' or embedding_type == 'Amplitude-Hybrid4-4':
        X1 = X[:2 ** 4]
        X2 = X[2 ** 4:2 ** 5]
        norm_X1, norm_X2 = np.linalg.norm(X1), np.linalg.norm(X2)
        X1, X2 = X1 / norm_X1, X2 / norm_X2

        if embedding_type == 'Amplitude-Hybrid4-1':
            MottonenStatePreparation(X1, wires=[0, 1, 2, 3])
            MottonenStatePreparation(X2, wires=[4, 5, 6, 7])
        elif embedding_type == 'Amplitude-Hybrid4-2':
            MottonenStatePreparation(X1, wires=[0, 2, 4, 6])
            MottonenStatePreparation(X2, wires=[1, 3, 5, 7])
        elif embedding_type == 'Amplitude-Hybrid4-3':
            MottonenStatePreparation(X1, wires=[0, 1, 6, 7])
            MottonenStatePreparation(X2, wires=[2, 3, 4, 5])
        elif embedding_type == 'Amplitude-Hybrid4-4':
            MottonenStatePreparation(X1, wires=[0, 3, 4, 7])
            MottonenStatePreparation(X2, wires=[1, 2, 5, 6])

    elif embedding_type == 'Amplitude-Hybrid2-1' or embedding_type == 'Amplitude-Hybrid2-2' \
            or embedding_type == 'Amplitude-Hybrid2-3' or embedding_type == 'Amplitude-Hybrid2-4':
        X1 = X[:4]
        X2 = X[4:8]
        X3 = X[8:12]
        X4 = X[12:16]
        norm_X1, norm_X2, norm_X3, norm_X4 = np.linalg.norm(X1), np.linalg.norm(X2), np.linalg.norm(X3), np.linalg.norm(
            X4)
        X1, X2, X3, X4 = X1 / norm_X1, X2 / norm_X2, X3 / norm_X3, X4 / norm_X4

        if embedding_type == 'Amplitude-Hybrid2-1':
            MottonenStatePreparation(X1, wires=[0,1])
            MottonenStatePreparation(X2, wires=[2,3])
            MottonenStatePreparation(X3, wires=[4,5])
            MottonenStatePreparation(X4, wires=[6,7])
        elif embedding_type == 'Amplitude-Hybrid2-2':
            MottonenStatePreparation(X1, wires=[0,4])
            MottonenStatePreparation(X2, wires=[1,5])
            MottonenStatePreparation(X3, wires=[2,6])
            MottonenStatePreparation(X4, wires=[3,7])
        elif embedding_type == 'Amplitude-Hybrid2-3':
            MottonenStatePreparation(X1, wires=[0,7])
            MottonenStatePreparation(X2, wires=[1,6])
            MottonenStatePreparation(X3, wires=[2,5])
            MottonenStatePreparation(X4, wires=[3,4])
        elif embedding_type == 'Amplitude-Hybrid2-4':
            MottonenStatePreparation(X1, wires=[0,2])
            MottonenStatePreparation(X2, wires=[1,3])
            MottonenStatePreparation(X3, wires=[4,6])
            MottonenStatePreparation(X4, wires=[5,7])

    # Hybrid Angle Embedding (HAE)
    elif embedding_type == 'Angular-Hybrid4-1' or embedding_type == 'Angular-Hybrid4-2' or \
            embedding_type == 'Angular-Hybrid4-3' or embedding_type == 'Angular-Hybrid4-4':
        N = 15 # 15 classical data in 4 qubits
        X1 = X[:N]
        X2 = X[N:2*N]

        if embedding_type == 'Angular-Hybrid4-1':
            Angular_Hybrid_4(X1, wires=[0, 1, 2, 3])
            Angular_Hybrid_4(X2, wires=[4, 5, 6, 7])
        elif embedding_type == 'Angular-Hybrid4-2':
            Angular_Hybrid_4(X1, wires=[0, 2, 4, 6])
            Angular_Hybrid_4(X2, wires=[1, 3, 5, 7])
        elif embedding_type == 'Angular-Hybrid4-3':
            Angular_Hybrid_4(X1, wires=[0, 1, 6, 7])
            Angular_Hybrid_4(X2, wires=[2, 3, 4, 5])
        elif embedding_type == 'Angular-Hybrid4-4':
            Angular_Hybrid_4(X1, wires=[0, 3, 4, 7])
            Angular_Hybrid_4(X2, wires=[1, 2, 5, 6])

    elif embedding_type == 'Angular-Hybrid2-1' or embedding_type == 'Angular-Hybrid2-2' \
            or embedding_type == 'Angular-Hybrid2-3' or embedding_type == 'Angular-Hybrid2-4':
        N = 3  # 3 classical bits in 2 qubits
        X1 = X[:N]
        X2 = X[N:2*N]
        X3 = X[2*N:3*N]
        X4 = X[3*N:4*N]

        if embedding_type == 'Angular-Hybrid2-1':
            Angular_Hybrid_2(X1, wires=[0,1])
            Angular_Hybrid_2(X2, wires=[2,3])
            Angular_Hybrid_2(X3, wires=[4,5])
            Angular_Hybrid_2(X4, wires=[6,7])
        elif embedding_type == 'Angular-Hybrid2-2':
            Angular_Hybrid_2(X1, wires=[0,4])
            Angular_Hybrid_2(X2, wires=[1,5])
            Angular_Hybrid_2(X3, wires=[2,6])
            Angular_Hybrid_2(X4, wires=[3,7])
        elif embedding_type == 'Angular-Hybrid2-3':
            Angular_Hybrid_2(X1, wires=[0,7])
            Angular_Hybrid_2(X2, wires=[1,6])
            Angular_Hybrid_2(X3, wires=[2,5])
            Angular_Hybrid_2(X4, wires=[3,4])
        elif embedding_type == 'Angular-Hybrid2-4':
            Angular_Hybrid_2(X1, wires=[0,2])
            Angular_Hybrid_2(X2, wires=[1,3])
            Angular_Hybrid_2(X3, wires=[4,6])
            Angular_Hybrid_2(X4, wires=[5,7])





