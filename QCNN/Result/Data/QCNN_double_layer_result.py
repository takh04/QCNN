# This is a mean and standard deviation of Double Layer QCNN with U9 and SU4 unitary ansatz
import numpy as np

U9_resize256_MNIST = np.array([0.8884160756501182, 0.891725768321513, 0.9347517730496454, 0.8222222222222222, 0.8600472813238771])
U9_AE8_MNIST = np.array([0.9598108747044918, 0.9465721040189126, 0.9527186761229315, 0.9730496453900709, 0.968321513002364])
U9_PCA8_MNIST = np.array([0.9768321513002364, 0.9782505910165484, 0.9900709219858156, 0.9678486997635933, 0.9768321513002364])
U9_PCA16_MNIST = np.array([0.9678486997635933, 0.9560283687943263, 0.9309692671394799, 0.9749408983451536, 0.9617021276595744])
U9_AE16_MNIST = np.array([0.9598108747044918, 0.9276595744680851, 0.91725768321513, 0.9059101654846335, 0.9182033096926714])

SU4_resize256_MNIST = np.array([0.9867612293144208, 0.9801418439716312, 0.9536643026004729, 0.9825059101654846, 0.9843971631205674])
SU4_AE8_MNIST = np.array([0.9692671394799054, 0.9895981087470449, 0.9607565011820332, 0.9867612293144208, 0.9408983451536643])
SU4_PCA8_MNIST = np.array([0.9886524822695035, 0.984869976359338, 0.9881796690307328, 0.9806146572104019, 0.983451536643026])
SU4_PCA16_MNIST = np.array([0.9763593380614657, 0.9886524822695035, 0.9806146572104019, 0.9801418439716312, 0.9853427895981087])
SU4_AE16_MNIST = np.array([0.9739952718676123, 0.9546099290780142, 0.9673758865248226, 0.941371158392435, 0.9891252955082742])

U9_resize256_FASHION = np.array([0.909, 0.9165, 0.889, 0.874, 0.913])
U9_AE8_FASHION = np.array([0.9285, 0.8135, 0.8715, 0.8485, 0.929])
U9_PCA8_FASHION = np.array([0.8125, 0.847, 0.884, 0.8305, 0.8075])
U9_PCA16_FASHION = np.array([0.778, 0.9275, 0.8925, 0.912, 0.829])
U9_AE16_FASHION = np.array([0.9135, 0.8945, 0.871, 0.9065, 0.8365])

SU4_resize256_FASHION = np.array([0.908, 0.9245, 0.9165, 0.9035, 0.916])
SU4_AE8_FASHION = np.array([0.939, 0.9325, 0.9335, 0.9315, 0.939])
SU4_PCA8_FASHION = np.array([0.9105, 0.8495, 0.8685, 0.8585, 0.8775])
SU4_PCA16_FASHION = np.array([0.9025, 0.8995, 0.9165, 0.884, 0.8865])
SU4_AE16_FASHION = np.array([0.919, 0.937, 0.9345, 0.9435, 0.911])


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



