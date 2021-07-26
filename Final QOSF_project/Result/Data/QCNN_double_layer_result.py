# This is a mean and standard deviation of Double Layer QCNN with U9 and SU4 unitary ansatz
import numpy as np

U9_resize256_MNIST = np.array([])
U9_AE8_MNIST = np.array([])
U9_PCA8_MNIST = np.array([])
U9_PCA16_MNIST = np.array([])
U9_AE16_MNIST = np.array([])

SU4_resize256_MNIST = np.array([])
SU4_AE8_MNIST = np.array([])
SU4_PCA8_MNIST = np.array([])
SU4_PCA16_MNIST = np.array([])
SU4_AE16_MNIST = np.array([])

U9_resize256_FASHION = np.array([])
U9_AE8_FASHION = np.array([])
U9_PCA8_FASHION = np.array([])
U9_PCA16_FASHION = np.array([])
U9_AE16_FASHION = np.array([])

SU4_resize256_FASHION = np.array([])
SU4_AE8_FASHION = np.array([])
SU4_PCA8_FASHION = np.array([])
SU4_PCA16_FASHION = np.array([])
SU4_AE16_FASHION = np.array([])


print("Result for MNIST dataset with Double Layer QCNN structure\n")
print("Result with U_9: \n")
print("resize256: " + str(U9_resize256_MNIST.mean()) +" +/- "+ str(U9_resize256_MNIST.std()))
print("PCA8: " + str(U9_PCA8_MNIST.mean()) + " +/- " + str(U9_PCA8_MNIST.std()))
print("AE8: " + str(U9_AE8_MNIST.mean()) + " +/- " + str(U9_AE8_MNIST.std()))
print("PCA16: " + str(U9_PCA16_MNIST.mean()) + " +/- " + str(U9_PCA16_MNIST.std()))
print("AE16: " + str(U9_AE16_MNIST.mean()) + " +/- " + str(U9_AE16_MNIST.std()))
print("Result with SU4: \n")
print("resize256: " + str(SU4_resize256_MNIST.mean()) +" +/- "+ str(SU4_resize256_MNIST.std()))
print("PCA8: " + str(SU4_PCA8_MNIST.mean()) + " +/- " + str(SU4_PCA8_MNIST.std()))
print("AE8: " + str(SU4_AE8_MNIST.mean()) + " +/- " + str(SU4_AE8_MNIST.std()))
print("PCA16: " + str(SU4_PCA16_MNIST.mean()) + " +/- " + str(SU4_PCA16_MNIST.std()))
print("AE16: " + str(SU4_AE16_MNIST.mean()) + " +/- " + str(SU4_AE16_MNIST.std()))

print("Result for Fashion MNIST dataset with Double Layer QCNN structure\n")
print("Result with U_9: \n")
print("resize256: " + str(U9_resize256_FASHION.mean()) +" +/- "+ str(U9_resize256_FASHION.std()))
print("PCA8: " + str(U9_PCA8_FASHION.mean()) + " +/- " + str(U9_PCA8_FASHION.std()))
print("AE8: " + str(U9_AE8_FASHION.mean()) + " +/- " + str(U9_AE8_FASHION.std()))
print("PCA16: " + str(U9_PCA16_FASHION.mean()) + " +/- " + str(U9_PCA16_FASHION.std()))
print("AE16: " + str(U9_AE16_FASHION.mean()) + " +/- " + str(U9_AE16_FASHION.std()))
print("Result with SU4: \n")
print("resize256: " + str(SU4_resize256_FASHION.mean()) +" +/- "+ str(SU4_resize256_FASHION.std()))
print("PCA8: " + str(SU4_PCA8_FASHION.mean()) + " +/- " + str(SU4_PCA8_FASHION.std()))
print("AE8: " + str(SU4_AE8_FASHION.mean()) + " +/- " + str(SU4_AE8_FASHION.std()))
print("PCA16: " + str(SU4_PCA16_FASHION.mean()) + " +/- " + str(SU4_PCA16_FASHION.std()))
print("AE16: " + str(SU4_AE16_FASHION.mean()) + " +/- " + str(SU4_AE16_FASHION.std()))



