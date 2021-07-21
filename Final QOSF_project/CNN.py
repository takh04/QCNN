import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
import data

dataset = 'mnist'
classes = [0,1]
Encodings = ['pca8']

learning_rate = 0.01
steps = 200
batch_size = 25

def CNN(Encodings):
    for Encoding in Encodings:

        X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes, feature_reduction = Encoding, binary = True)
        X_train = np.array([np.reshape(X_train[i], (8, 1, 1)) for i in range(len(X_train))])
        X_test = np.array([np.reshape(X_test[i], (8, 1, 1)) for i in range(len(X_test))])

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-08, amsgrad=False,name='Adam')

        accuracy = []


        model = models.Sequential()
        model.add(layers.Conv2D(3, (2, 1), activation='relu', input_shape=(8, 1, 1)))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Conv2D(3, (2, 1), activation='relu'))
        model.add(layers.MaxPooling2D((2, 1)))
        model.add(layers.Flatten())
        # model.add(layers.Dense(4, activation='relu'))
        model.add(layers.Dense(1, activation='softmax'))

        # Compile CNN
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

        for it in range(steps):
            # Generate Random Batch
            batch_index = np.random.randint(0, 12665, (batch_size,))
            X_batch = np.array([X_train[i] for i in batch_index])
            Y_batch = np.array([Y_train[i] for i in batch_index])

            # Train CNN
            model.fit(X_batch, Y_batch, batch_size=batch_size, epochs=1)

            Y_pred = model(X_batch)

            loss = tf.keras.losses.mean_squared_error(Y_pred, Y_batch)
            if it % 10 == 0:
                print(loss)

        test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
        accuracy.append(test_acc)
        # Clear CNN Model
        tf.keras.backend.clear_session()

CNN(Encodings)