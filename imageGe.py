import os
os.environ["KERAS_BACKEND"] = "torch"

import keras_core as keras
import math
import tqdm
import json
from rich.padding import Padding
from rich.panel import Panel
from rich.console import Console
import numpy as np
print=Console().print

def printPanel(text, style="on blue"):
    print(Panel.fit(Padding(
        text,(2, 5), style=style, expand=False)))


epochs = 1000

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

printPanel("Backend:"+ keras.backend.backend()+"\n"+"Device:"+ keras.src.backend.torch.core.get_device())

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.MeanAbsoluteError())

for i in tqdm.tqdm(range(epochs)):
    model.fit(x=x_train, y=y_train, epochs=1, verbose=0)

model.save("model"+str(input_shape)+"to"+str(output_shape)+".keras")
