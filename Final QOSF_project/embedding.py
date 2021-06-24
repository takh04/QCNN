# This is an implementation of data_embedding function used for 8 qubits Quantum Convolutional Neural Network (QCNN)
# and Hierarchical Quantum Classifier circuit.
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from pennylane.templates.state_preparations import MottonenStatePreparation
import numpy as np
import Mottonen


def data_embedding(X, embedding_type='Amplitude'):
    if embedding_type == 'Amplitude':
        AmplitudeEmbedding(X, wires=range(8), normalize=True)
    elif embedding_type == 'Angle':
        AngleEmbedding(X, wires=range(8), rotation='Y')
    elif embedding_type == 'Hybrid32':
        X1 = X[:2 ** 4]
        X2 = X[2 ** 4:2 ** 5]
        norm_X1, norm_X2 = np.linalg.norm(X1), np.linalg.norm(X2)
        X1, X2 = X1 / norm_X1, X2 / norm_X2

        MottonenStatePreparation(X1, wires=[0,2,4,6])
        MottonenStatePreparation(X2, wires=[1,3,5,7])
    elif embedding_type == 'Hybrid16':
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


    #### Test Code Block Start
    # This is a test code for testing different variations of Hybrid Embedding
    elif embedding_type == 'Hybrid32-1' or embedding_type == 'Hybrid32-2' or embedding_type == 'Hybrid32-3' or embedding_type == 'Hybrid32-4':
        X1 = X[:2 ** 4]
        X2 = X[2 ** 4:2 ** 5]
        norm_X1, norm_X2 = np.linalg.norm(X1), np.linalg.norm(X2)
        X1, X2 = X1 / norm_X1, X2 / norm_X2

        if embedding_type == 'Hybrid32-1':
            MottonenStatePreparation(X1, wires=[0, 1, 2, 3])
            MottonenStatePreparation(X2, wires=[4, 5, 6, 7])
        elif embedding_type == 'Hybrid32-2':
            MottonenStatePreparation(X1, wires=[0, 2, 4, 6])
            MottonenStatePreparation(X2, wires=[1, 3, 5, 7])
        elif embedding_type == 'Hybrid32-3':
            MottonenStatePreparation(X1, wires=[0, 1, 6, 7])
            MottonenStatePreparation(X2, wires=[2, 3, 4, 5])
        elif embedding_type == 'Hybrid32-4':
            MottonenStatePreparation(X1, wires=[0, 3, 4, 7])
            MottonenStatePreparation(X2, wires=[1, 2, 5, 6])



    elif embedding_type == 'Hybrid16-1' or embedding_type == 'Hybrid16-2' or embedding_type == 'Hybrid16-3' or embedding_type == 'Hybrid16-4':
        X1 = X[:4]
        X2 = X[4:8]
        X3 = X[8:12]
        X4 = X[12:16]
        norm_X1, norm_X2, norm_X3, norm_X4 = np.linalg.norm(X1), np.linalg.norm(X2), np.linalg.norm(X3), np.linalg.norm(
            X4)
        X1, X2, X3, X4 = X1 / norm_X1, X2 / norm_X2, X3 / norm_X3, X4 / norm_X4


        if embedding_type == 'Hybrid16-1':
            MottonenStatePreparation(X1, wires=[0,1])
            MottonenStatePreparation(X2, wires=[2,3])
            MottonenStatePreparation(X3, wires=[4,5])
            MottonenStatePreparation(X4, wires=[6,7])
        elif embedding_type == 'Hybrid16-2':
            MottonenStatePreparation(X1, wires=[0,4])
            MottonenStatePreparation(X2, wires=[1,5])
            MottonenStatePreparation(X3, wires=[2,6])
            MottonenStatePreparation(X4, wires=[3,7])
        elif embedding_type == 'Hybrid16-3':
            MottonenStatePreparation(X1, wires=[0,7])
            MottonenStatePreparation(X2, wires=[1,6])
            MottonenStatePreparation(X3, wires=[2,5])
            MottonenStatePreparation(X4, wires=[3,4])
        elif embedding_type == 'Hybrid16-4':
            MottonenStatePreparation(X1, wires=[0,2])
            MottonenStatePreparation(X2, wires=[1,3])
            MottonenStatePreparation(X3, wires=[4,6])
            MottonenStatePreparation(X4, wires=[5,7])

    #### Test Code Block Ends



    #### Test Code Block 2 Starts
    # This is a test code block for Anglular Mottonen State Preparation
    elif embedding_type == 'Hybrid16-Angle':
        X1 = X[:4]
        X2 = X[4:8]
        X3 = X[8:12]
        X4 = X[12:16]
        Mottonen.Mottonen_16(X1, wires=[0, 1])
        Mottonen.Mottonen_16(X2, wires=[2, 3])
        Mottonen.Mottonen_16(X3, wires=[4, 5])
        Mottonen.Mottonen_16(X4, wires=[6, 7])

    elif embedding_type == 'Hybrid32-Angle':
        X1 = X[:2**4]
        X2 = X[2**4:2**5]
        Mottonen.Mottonen_32(X1, wires=[0,1,2,3])
        Mottonen.Mottonen_32(X2, wires=[4,5,6,7])
