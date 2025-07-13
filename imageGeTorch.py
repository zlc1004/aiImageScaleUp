import math
import tqdm
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

print("Device:", device)

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

class Model(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(input_shape**2, 12)
        self.fc2 = torch.nn.Linear(12, 18)
        self.fc3 = torch.nn.Linear(18, 12)
        self.fc4 = torch.nn.Linear(12, output_shape**2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = Model(input_shape, output_shape).to(device)
optimizer = optim.Adam(model.parameters())
criterion = torch.nn.L1Loss()

x_train_tensor = torch.from_numpy(x_train).to(device).float()
y_train_tensor = torch.from_numpy(y_train).to(device).float()

dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

for _ in tqdm.tqdm(range(epochs)):
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "model"+str(input_shape)+"to"+str(output_shape)+".torch")