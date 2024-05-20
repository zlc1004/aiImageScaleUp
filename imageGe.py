import math
import tqdm
import json
import keras
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np
epochs = 1000

config = tf.compat.v1.ConfigProto(device_count={"CPU": 8})
K.set_session(tf.compat.v1.Session(config=config))
with open("imageTrainData.json") as f:
    data = json.load(f)

x_train = np.array(data[0])
y_train = np.array(data[1])
input_shape = int(math.sqrt(len(x_train[0])))
output_shape = int(math.sqrt(len(y_train[0])))
print("Input shape:", input_shape, "x", input_shape)
print("Output shape:", output_shape, "x", output_shape)
print("Training", epochs, "epochs")


model = keras.models.Sequential()
model.add(keras.layers.Dense(units=12, input_shape=(input_shape**2,)))
model.add(keras.layers.Dense(units=18))
model.add(keras.layers.Dense(units=12))
model.add(keras.layers.Dense(units=output_shape**2))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.MeanAbsoluteError())

for i in tqdm.tqdm(range(epochs)):
    model.fit(x=x_train, y=y_train, epochs=1, verbose=0)

model.save("model"+str(input_shape)+"to"+str(output_shape)+".keras")
