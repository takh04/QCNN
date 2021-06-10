import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import data

dataset = 'mnist'
classes = [0,1]
feature_reduction = 'pca8'
binary = True
X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes, feature_reduction, binary)

X_train = np.array([np.reshape(X_train[i], (8,1,1)) for i in range(len(X_train))])
X_test = np.array([np.reshape(X_test[i], (8,1,1)) for i in range(len(X_test))])







learning_rate = 0.01
iteration = 300
batch_size = 25

optimizer = tf.keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-08, amsgrad=False,
    name='Adam')

accuracy = []


    model = models.Sequential()
    model.add(layers.Conv2D(3, (2, 1), activation='relu', input_shape=(8, 1, 1)))
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(layers.Conv2D(3, (2, 1), activation='relu'))
    model.add(layers.MaxPooling2D((2, 1)))
    model.add(layers.Flatten())
    # model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    # Compile CNN
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    for i in range(iteration):
        # Generate Random Batch
        batch_index = np.random.randint(0, 12665, (batch_size,))
        X_batch = np.array([x[i] for i in batch_index])
        Y_batch = np.array([y_train_01[i] for i in batch_index])

        # Train CNN
        model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=1)

    test_loss, test_acc = model.evaluate(xt, y_test_01, verbose=2)

    accuracy.append(test_acc)

    # Clear CNN Model
    tf.keras.backend.clear_session()

print(accuracy)
