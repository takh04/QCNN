# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# and Hierarchical Quantum Classifier circuit.
import pennylane as qml
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from pennylane.templates.state_preparations import MottonenStatePreparation
import numpy as np
from Angular_hybrid import Angular_Hybrid_4, Angular_Hybrid_2

def CompactAngleEmbedding(X, wires):
    X1, X2, X3, X4, X5, X6, X7, X8 = X[:2], X[2:4], X[4:6], X[6:8], X[8:10], X[10:12], X[12:14], X[14:15]
    X = [X1, X2, X3, X4, X5, X6, X7, X8]
    for i in range(8):
        qml.RX(X[i], wires=wires[i])
        qml.RY(X[i], wires=wires[i])

def data_embedding(X, embedding_type='Amplitude'):
    if embedding_type == 'Amplitude':
        AmplitudeEmbedding(X, wires=range(8), normalize=True)
    elif embedding_type == 'Angle':
        AngleEmbedding(X, wires=range(8), rotation='Y')
    elif embedding_type == 'Angle-compact':
        CompactAngleEmbedding(X, wires=range(8))
    elif embedding_type == 'Amplitude-Hybrid4':
        X1 = X[:2 ** 4]
        X2 = X[2 ** 4:2 ** 5]
        norm_X1, norm_X2 = np.linalg.norm(X1), np.linalg.norm(X2)
        X1, X2 = X1 / norm_X1, X2 / norm_X2

        MottonenStatePreparation(X1, wires=[0,2,4,6])
        MottonenStatePreparation(X2, wires=[1,3,5,7])
    elif embedding_type == 'Amplitude-Hybrid2':
        X1 = X[:4]
        X2 = X[4:8]
        X3 = X[8:12]
        X4 = X[12:16]
        norm_X1, norm_X2, norm_X3, norm_X4 = np.linalg.norm(X1), np.linalg.norm(X2), np.linalg.norm(X3), np.linalg.norm(
            X4)
        X1, X2, X3, X4 = X1 / norm_X1, X2 / norm_X2, X3 / norm_X3, X4 / norm_X4
        MottonenStatePreparation(X1, wires=[0,4])
        MottonenStatePreparation(X2, wires=[1,5])
        MottonenStatePreparation(X3, wires=[2,6])
        MottonenStatePreparation(X4, wires=[3,7])
    elif embedding_type == 'Angular-Hybrid4':
        X1 = X[:15]
        X2 = X[15:30]
        Angular_Hybrid_4(X1, wires=[0, 1, 2, 3])
        Angular_Hybrid_4(X2, wires=[4, 5, 6, 7])
    elif embedding_type == 'Angular-Hybrid2':
        X1 = X[:3]
        X2 = X[3:6]
        X3 = X[6:9]
        X4 = X[9:12]
        Angular_Hybrid_2(X1, wires=[0, 1])
        Angular_Hybrid_2(X2, wires=[2, 3])
        Angular_Hybrid_2(X3, wires=[4, 5])
        Angular_Hybrid_2(X4, wires=[6, 7])







    #########################################################################
    #########################################################################
    # When using Hybrid Embedding there are different combinations that we can encode the data into qubit blocks.
    # In this section of the code we will investigate the performances of the different possible combinations.
    # The performances will be evaluated by accuracy measure and entanglement measure.
    # If possible we will try to deduce the reason for such results
    #### Test Code Block Start

    # Amplitude Hybrid
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

    # Angular Hybrid Embedding
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
    #### Test Code Block Ends
    #########################################################################
    #########################################################################




