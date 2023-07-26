import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch
from opacus import PrivacyEngine
from torch.utils.data import Dataset, DataLoader
import torch
import math
from opacus.utils.batch_memory_manager import BatchMemoryManager


class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class FakeImageDataset(Dataset):
    def __init__(self, num_samples, image_shape):
        self.num_samples = num_samples
        self.image_shape = image_shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(self.image_shape)
        label = torch.randn(1)
        return image, label


input_size = 3*16*16
output_size = 1

model = SimpleLinearModel(input_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_samples = 1356
batch_size = 1024
max_physical_batch_size = 512

dataset = FakeImageDataset(num_samples, 3*16*16)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

privacy_engine = PrivacyEngine(accountant="rdp")
model, optimizer, dataloader_dp = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    target_epsilon=8,
    target_delta=1 / num_samples,
    max_grad_norm=1,
    epochs=20
)

max_steps = 5000
epochs = math.ceil(max_steps / len(dataloader_dp))
max_steps = epochs * len(dataloader_dp)
max_steps = (batch_size / max_physical_batch_size) * max_steps

step_counter = 0
epoch_counter = 0
while True:
    with BatchMemoryManager(
            data_loader=dataloader_dp,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer
    ) as new_train_loader:
        for batch, labels in new_train_loader:
            running_loss = 0.0
            outputs = model(batch)
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step_counter += 1
        epoch_counter += 1
        print(f"Epoch {epoch_counter}/{epochs}")
        if step_counter >= max_steps:
            break
    if step_counter >= max_steps:
        break
print("hello")

