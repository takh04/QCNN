# This is a result for QCNN double layer

import numpy as np

####################
# SU4 Unitary Ansatz
####################

#MNIST
resize256_SU4_mnist = np.array([])
pca8_SU4_mnist = np.array([])
AE8_SU4_mnist = np.array([])
pca16_SU4_mnist = np.array([])
AE16_SU4_mnist = np.array([])

#Fashion MNIST
resize256_SU4_fashion = np.array([0.9125, 0.899, 0.8865, 0.888, 0.9095])
pca8_SU4_fashion = np.array([0.876, 0.893, 0.9205, 0.893, 0.9035])
AE8_SU4_fashion = np.array([0.94, 0.9415, 0.9455, 0.9385, 0.924])
pca16_SU4_fashion = np.array([0.787, 0.7865, 0.7875, 0.7855, 0.7765])
AE16_SU4_fashion = np.array([0.769, 0.9425, 0.9355, 0.9345, 0.874])


resize256_SU4_mnist_mean, resize256_SU4_mnist_std = resize256_SU4_mnist.mean(), resize256_SU4_mnist.std()
pca8_SU4_mnist_mean, pca8_SU4_mnist_std = pca8_SU4_mnist.mean(), pca8_SU4_mnist.std()
AE8_SU4_mnist_mean, AE8_SU4_mnist_std = AE8_SU4_mnist.mean(), AE8_SU4_mnist.std()
pca16_SU4_mnist_mean, pca16_SU4_mnist_std = pca16_SU4_mnist.mean(), pca16_SU4_mnist.std()
AE16_SU4_mnist_mean, AE16_SU4_mnist_std = AE16_SU4_mnist.mean(), AE16_SU4_mnist.std()

resize256_SU4_fashion_mean, resize256_SU4_fashion_std = resize256_SU4_fashion.mean(), resize256_SU4_fashion.std()
pca8_SU4_fashion_mean, pca8_SU4_fashion_std = pca8_SU4_fashion.mean(), pca8_SU4_fashion.std()
AE8_SU4_fashion_mean, AE8_SU4_fashion_std = AE8_SU4_fashion.mean(), AE8_SU4_fashion.std()
pca16_SU4_fashion_mean, pca16_SU4_fashion_std = pca16_SU4_fashion.mean(), pca16_SU4_fashion.std()
AE16_SU4_fashion_mean, AE16_SU4_fashion_std = AE16_SU4_fashion.mean(), AE16_SU4_fashion.std()

####################
# U9 Unitary Ansatz
####################

#MNIST
resize256_U9_mnist = np.array([])
pca8_U9_mnist = np.array([])
AE8_U9_mnist = np.array([])
pca16_U9_mnist = np.array([])
AE16_U9_mnist = np.array([])

#Fashion MNIST
resize256_U9_fashion = np.array([])
pca8_U9_fashion = np.array([])
AE8_U9_fashion = np.array([])
pca16_U9_fashion = np.array([])
AE16_U9_fashion = np.array([])


resize256_U9_mnist_mean, resize256_U9_mnist_std = resize256_U9_mnist.mean(), resize256_U9_mnist.std()
pca8_U9_mnist_mean, pca8_U9_mnist_std = pca8_U9_mnist.mean(), pca8_U9_mnist.std()
AE8_U9_mnist_mean, AE8_U9_mnist_std = AE8_U9_mnist.mean(), AE8_U9_mnist.std()
pca16_U9_mnist_mean, pca16_U9_mnist_std = pca16_U9_mnist.mean(), pca16_U9_mnist.std()
AE16_U9_mnist_mean, AE16_U9_mnist_std = AE16_U9_mnist.mean(), AE16_U9_mnist.std()

resize256_U9_fashion_mean, resize256_U9_fashion_std = resize256_U9_fashion.mean(), resize256_U9_fashion.std()
pca8_U9_fashion_mean, pca8_U9_fashion_std = pca8_U9_fashion.mean(), pca8_U9_fashion.std()
AE8_U9_fashion_mean, AE8_U9_fashion_std = AE8_U9_fashion.mean(), AE8_U9_fashion.std()
pca16_U9_fashion_mean, pca16_U9_fashion_std = pca16_U9_fashion.mean(), pca16_U9_fashion.std()
AE16_U9_fashion_mean, AE16_U9_fashion_std = AE16_U9_fashion.mean(), AE16_U9_fashion.std()


print("Result for SU4 Double Layers \n")
print("MNIST: \n")
print("MNIST-Resize256: " + str(resize256_SU4_mnist_mean) +" "+ str(resize256_SU4_mnist_std) + "\n")
print("MNIST-pca8: " + str(pca8_SU4_mnist_mean) +" "+ str(pca8_SU4_mnist_std) + "\n")
print("MNIST-AE8: " + str(AE8_SU4_mnist_mean) +" "+ str(AE8_SU4_mnist_std) + "\n")
print("MNIST-pca16: " + str(pca16_SU4_mnist_mean) +" "+ str(pca16_SU4_mnist_std) + "\n")
print("MNIST-AE16: " + str(AE16_SU4_mnist_mean) +" "+ str(AE16_SU4_mnist_std) + "\n")
print("Fashion MNIST: \n")
print("Fashion-Resize256: " + str(resize256_SU4_fashion_mean) +" "+ str(resize256_SU4_fashion_std) + "\n")
print("Fashion-pca8: " + str(pca8_SU4_fashion_mean) +" "+ str(pca8_SU4_fashion_std) + "\n")
print("Fashion-AE8: " + str(AE8_SU4_fashion_mean) +" "+ str(AE8_SU4_fashion_std) + "\n")
print("Fashion-pca16: " + str(pca16_SU4_fashion_mean) +" "+ str(pca16_SU4_fashion_std) + "\n")
print("Fashion-AE16: " + str(AE16_SU4_fashion_mean) +" "+ str(AE16_SU4_fashion_std) + "\n")
print("\n")
print("Result for U9 Double Layers \n")
print("MNIST: \n")
print("MNIST-Resize256: " + str(resize256_U9_mnist_mean) +" "+ str(resize256_U9_mnist_std) + "\n")
print("MNIST-pca8: " + str(pca8_U9_mnist_mean) +" "+ str(pca8_U9_mnist_std) + "\n")
print("MNIST-AE8: " + str(AE8_U9_mnist_mean) +" "+ str(AE8_U9_mnist_std) + "\n")
print("MNIST-pca16: " + str(pca16_U9_mnist_mean) +" "+ str(pca16_U9_mnist_std) + "\n")
print("MNIST-AE16: " + str(AE16_U9_mnist_mean) +" "+ str(AE16_U9_mnist_std) + "\n")
print("Fashion MNIST: \n")
print("Fashion-Resize256: " + str(resize256_U9_fashion_mean) +" "+ str(resize256_U9_fashion_std) + "\n")
print("Fashion-pca8: " + str(pca8_U9_fashion_mean) +" "+ str(pca8_U9_fashion_std) + "\n")
print("Fashion-AE8: " + str(AE8_U9_fashion_mean) +" "+ str(AE8_U9_fashion_std) + "\n")
print("Fashion-pca16: " + str(pca16_U9_fashion_mean) +" "+ str(pca16_U9_fashion_std) + "\n")
print("Fashion-AE16: " + str(AE16_U9_fashion_mean) +" "+ str(AE16_U9_fashion_std) + "\n")


