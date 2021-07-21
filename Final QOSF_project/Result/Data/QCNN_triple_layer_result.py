# This is a result for QCNN triple layers
import numpy as np

####################
# SU4 Unitary Ansatz
####################

#MNIST
resize256_SU4_mnist = np.array([0.9706855791962175, 0.9711583924349881, 0.9829787234042553, 0.9891252955082742, 0.9806146572104019])
pca8_SU4_mnist = np.array([0.9787234042553191, 0.9853427895981087, 0.983451536643026, 0.9881796690307328, 0.9867612293144208])
AE8_SU4_mnist = np.array([0.9744680851063829, 0.968321513002364, 0.9886524822695035, 0.9895981087470449, 0.9886524822695035])
pca16_SU4_mnist = np.array([0.9541371158392435, 0.9309692671394799, 0.9475177304964539, 0.9394799054373523, 0.9522458628841608])
AE16_SU4_mnist = np.array([0.9276595744680851, 0.9617021276595744, 0.9777777777777777, 0.9200945626477541, 0.9536643026004729])

#Fashion MNIST
resize256_SU4_fashion = np.array([0.918, 0.918, 0.8945, 0.911, 0.88])
pca8_SU4_fashion = np.array([0.9335, 0.8915, 0.87, 0.899, 0.92])
AE8_SU4_fashion = np.array([])
pca16_SU4_fashion = np.array([0.798, 0.787, 0.7835, 0.7875, 0.784])
AE16_SU4_fashion = np.array([0.9525, 0.9045, 0.9385, 0.942, 0.968])



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
resize256_U9_mnist = np.array([0.950354609929078, 0.9555555555555556, 0.8463356973995272, 0.9408983451536643, 0.9782505910165484])
pca8_U9_mnist = np.array([0.975886524822695, 0.9385342789598109, 0.9692671394799054, 0.9768321513002364, 0.9626477541371158])
AE8_U9_mnist = np.array([0.991016548463357, 0.9678486997635933, 0.9867612293144208, 0.924822695035461, 0.9073286052009456])
pca16_U9_mnist = np.array([0.926241134751773, 0.8822695035460993, 0.9044917257683215, 0.9635933806146572, 0.9191489361702128])
AE16_U9_mnist = np.array([0.8312056737588652, 0.8888888888888888, 0.9735224586288416, 0.9546099290780142, 0.9385342789598109])

#Fashion MNIST
resize256_U9_fashion = np.array([0.8715, 0.8905, 0.907, 0.867, 0.874])
pca8_U9_fashion = np.array([0.8665, 0.8165, 0.8865, 0.8575, 0.8565])
AE8_U9_fashion = np.array([0.819, 0.889, 0.9275, 0.9165, 0.922])
pca16_U9_fashion = np.array([0.7835, 0.782, 0.7825, 0.7885, 0.7815])
AE16_U9_fashion = np.array([0.94, 0.891, 0.75, 0.8375, 0.824])



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