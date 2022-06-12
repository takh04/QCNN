import numpy as np


CNN_resize256_adam = np.array([0.96, 0.9585, 0.5, 0.9535, 0.9525])
CNN_pca8_adam = np.array([0.925, 0.91, 0.8745, 0.8985, 0.935])
CNN_ae8_adam = np.array([0.928, 0.9495, 0.9495, 0.9215, 0.9395])
CNN_pca16_adam = np.array([0.906, 0.944, 0.909, 0.9465, 0.9195])
CNN_ae16_adam = np.array([0.961, 0.952, 0.949, 0.903, 0.9455])

CNN_resize256_nesterov = np.array([0.8715, 0.936, 0.945, 0.9535, 0.948])
CNN_pca8_nesterov = np.array([0.5, 0.908, 0.8005, 0.6215, 0.737])
CNN_ae8_nesterov = np.array([0.708, 0.6145, 0.5905, 0.7565, 0.7385])
CNN_pca16_nesterov = np.array([0.7045, 0.5, 0.788, 0.851, 0.5])
CNN_ae16_nesterov = np.array([0.828, 0.5005, 0.8855, 0.9155, 0.5])

print("Resize256(ADAM) result: " + str(CNN_resize256_adam.mean()) + " +/- " +  str(CNN_resize256_adam.std()))
print("PCA8(ADAM) result: " + str(CNN_pca8_adam.mean()) + " +/- " + str(CNN_pca8_adam.std()))
print("AE8(ADAM) result: " + str(CNN_ae8_adam.mean()) + " +/- " + str(CNN_ae8_adam.std()))
print("PCA16(ADAM) result: " + str(CNN_pca16_adam.mean()) + " +/- " + str(CNN_pca16_adam.std()))
print("AE16(ADAM) result: " + str(CNN_ae16_adam.mean()) + " +/- " + str(CNN_ae16_adam.std()))
print("\n")
print("Resize256(nesterov) result: " + str(CNN_resize256_nesterov.mean()) + " +/- " + str(CNN_resize256_nesterov.std()))
print("PCA8(nesterov) result: " + str(CNN_pca8_nesterov.mean()) + " +/- " + str(CNN_pca8_nesterov.std()))
print("AE8(nesterov) result: " + str(CNN_ae8_nesterov.mean()) + " +/- " + str(CNN_ae8_nesterov.std()))
print("PCA16(nesterov) result: " + str(CNN_pca16_nesterov.mean()) + " +/- " + str(CNN_pca16_nesterov.std()))
print("AE16(nesterov) result: " + str(CNN_ae16_nesterov.mean()) + " +/- " + str(CNN_ae16_nesterov.std()))