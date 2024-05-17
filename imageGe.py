import keras,json,tqdm
import numpy as np

with open("imageTrainData.json") as f:
    data=json.load(f)

x_train = np.array(data[0])
print(x_train)
y_train = np.array(data[1])

model=keras.models.Sequential()
model.add(keras.layers.Dense(units=16, input_shape=(16,)))
model.add(keras.layers.Dense(units=12))
model.add(keras.layers.Dense(units=18))
model.add(keras.layers.Dense(units=12))
model.add(keras.layers.Dense(units=25))

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanAbsoluteError())

for i in tqdm.tqdm(range(1000)):
    model.fit(x=x_train, y=y_train, epochs=1,verbose = 0)

model.save("model.keras")