import keras,json,tqdm
import numpy as np

with open("imageTrainData.json") as f:
    data=json.load(f)

x_train = np.array(data[0])
y_train = np.array(data[1])

model=keras.models.Sequential()
model.add(keras.layers.Dense(units=9, input_shape=(3*3,)))
model.add(keras.layers.Dense(units=12))
model.add(keras.layers.Dense(units=18))
model.add(keras.layers.Dense(units=12))
model.add(keras.layers.Dense(units=4*4))

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanAbsoluteError())

for i in tqdm.tqdm(range(5000)):
    model.fit(x=x_train, y=y_train, epochs=1,verbose = 0)

model.save("model.keras")