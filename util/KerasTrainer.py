from datetime import date
import numpy as np
import tensorflow as tf

dataset = np.loadtxt('datasets/NoiseArrayTrain.txt', delimiter=' ')

X = dataset[:,0:2]
y = dataset[:,2]

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(24, input_shape=(2,), activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=1000, batch_size=5)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

#tf.keras.models.save_model(model, "model.h5")

model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model/model.h5")
print("Saved model to disk")
