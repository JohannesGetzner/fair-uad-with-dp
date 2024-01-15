import torch
from opacus import PrivacyEngine
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from opacus.utils.batch_memory_manager import BatchMemoryManager


class LinearModel(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_shape, output_shape)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        out = self.linear(x)
        return out


# Create a custom dataset class
class FakeImageDataset(Dataset):
    def __init__(self, num_samples, image_shape):
        self.num_samples = num_samples
        self.image_shape = image_shape

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(self.image_shape)
        return image


if __name__ == '__main__':
    num_samples = 100
    image_shape = (3, 64, 64)
    batch_size = 8

    dataset = FakeImageDataset(num_samples, image_shape)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LinearModel(input_shape=3 * 64 * 64, output_shape=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, data_loader = privacy_engine.make_private_with_epsilon(module=model, optimizer=optimizer,
        data_loader=dataloader, target_epsilon=8, target_delta=1 / num_samples, max_grad_norm=1, epochs=20)

    with BatchMemoryManager(data_loader=dataloader, max_physical_batch_size=512,
                            optimizer=optimizer) as new_train_loader:
        for batch in new_train_loader:
            print(batch.shape)
