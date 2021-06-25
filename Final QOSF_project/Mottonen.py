# This is an implementation of an alternative Mottonen State Preparation to avoid normalization problem.
import pennylane as qml

# For Hybrid 16 / 3 bits of information is embedded in 2 wires
def Angular_Mottonen_16(X, wires):
    # This is not an accurate code / must be modified
    qml.RY(X[0], wires=wires[0])
    qml.CRY(X[1], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRY(X[2], wires=[wires[0], wires[1]])

# For Hybrid 32 / 15 bits of information is embedded in 4 wires
def Angular_Mottonen_32(X, wires):
    # This is not an accurate code / must be modified
    qml.RY(X[0], wires=wires[0])
    qml.CRY(X[1], wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRY(X[2], wires=[wires[0], wires[1]])
