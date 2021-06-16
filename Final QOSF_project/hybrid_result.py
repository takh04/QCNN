# This generates result of benchmarking code of hybrid embedding
###################
# Aim to test
# Encodings = ['pca16', 'autoencoder16'] / ['pca32', 'autoencoder32']
# embeedings = ['Hybrid16-1', 'Hybrid16-2', 'Hybrid16-3', 'Hybrid16-4'] / ['Hybrid32-1'. 'Hybrid32-2'. 'Hybrid32-3'. 'Hybrid32-4']
###################
import Benchmarking

dataset = 'mnist'
classes = [0,1]
circuit = 'QCNN'

# Benchmarking target
Encoding = 'autoencoder16'
Embedding = 'Hybrid16-2'

Benchmarking.Benchmarking_hybrid_embedding(dataset, classes, Encoding, Embedding, circuit, binary=True)


