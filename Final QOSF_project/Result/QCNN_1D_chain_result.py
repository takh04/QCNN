# This is a mean and standard deviation of 1D chain QCNN with SU4 unitary ansatz
import numpy as np

resize256 = np.array([0.9706855791962175, 0.9843971631205674, 0.9815602836879432, 0.9423167848699764
, 0.968321513002364])

AE8 = np.array([0.9508274231678487, 0.9735224586288416, 0.9617021276595744, 0.9617021276595744, 0.9446808510638298])

PCA8 = np.array([0.9886524822695035, 0.9862884160756501, 0.9801418439716312, 0.9820330969267139, 0.9810874704491725])

PCA16 = np.array([0.9527186761229315, 0.9361702127659575, 0.9267139479905437, 0.9446808510638298, 0.9536643026004729])

AE16 = np.array([0.8657210401891253, 0.9687943262411347, 0.968321513002364, 0.8869976359338061, 0.8988179669030733])

resize256_mean, resize256_std = resize256.mean(), resize256.std()
PCA8_mean, PCA8_std = PCA8.mean(), PCA8.std()
AE8_mean, AE8_std = AE8.mean(), AE8.std()
PCA16_mean, PCA16_std = PCA16.mean(), PCA16.std()
AE16_mean, AE16_std = AE16.mean(), AE16.std()

print("resize256: " + str(resize256_mean) +" "+ str(resize256_std))
print("PCA8: " + str(PCA8_mean) + " " + str(PCA8_std))
print("AE8: " + str(AE8_mean) + " " + str(AE8_std))
print("PCA16: " + str(PCA16_mean) + " " + str(PCA16_std))
print("AE16: " + str(AE16_mean) + " " + str(AE16_std))