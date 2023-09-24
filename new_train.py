from google.colab import drive
drive.mount('/content/drive')

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

from timeit import default_timer as timer
def print_train_time(start: float, end: float, device: torch.device=None):
    total_time = end - start
    print(f"Train time on {device}: {total_time}")

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Type123Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        # image = torchvision.io.read_image(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, downsample=None, stride=1):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=(3, 3),
                    stride=stride,
                    padding=1,
                    bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(in_channels=mid_channels,
                    out_channels=mid_channels*self.expansion,
                    kernel_size=(1, 1),
                    stride=1,
                    padding=0,
                    bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels*self.expansion)
        self.relu = nn.ReLU()
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x.clone()

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)

        return x

# ----------------
class ResNet(nn.Module):
    def __init__(self, block, layers, input_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv2d(in_channels=input_channels,
                            out_channels=64,
                            kernel_size=(7, 7),
                            stride=2,
                            padding=3,
                            bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], mid_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], mid_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], mid_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], mid_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn(self.conv(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, blocks_num, mid_channels, stride):
        downsample = None
        layers = []
        if stride != 1 or self.in_channels != mid_channels*4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                            out_channels=mid_channels*4,
                            kernel_size=(1, 1),
                            stride=stride,
                            bias=False),
                nn.BatchNorm2d(mid_channels*4)
            )
        layers.append(block(in_channels=self.in_channels,
                            mid_channels=mid_channels,
                            downsample=downsample,
                            stride=stride))
        self.in_channels = mid_channels*4

        for i in range(blocks_num-1):
            layers.append(block(in_channels=self.in_channels, mid_channels=mid_channels))

        return nn.Sequential(*layers)

# ----------
model = ResNet(ResidualBlock, [3, 4, 6, 3], 3, 3).to(device)

in_channels = 3
num_classes = 3
learning_rate = 0.01
batch_size = 256
Epochs = 10
#    transforms.ToPILImage(),
tfm = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()])
dataset = Type123Dataset(
    csv_file='drive/MyDrive/for_train/fortypes.csv', root_dir='drive/MyDrive/for_train/train/', transform=tfm)
train_set_size = int(len(dataset)*0.9)
test_set_size = len(dataset) - train_set_size

torch.manual_seed(42)
train_ds, val_ds = random_split(dataset, [train_set_size, test_set_size])
train_dataloader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from tqdm.auto import tqdm

train_losses = []
train_accs = []
test_losses = []
test_accs = []

start_train = timer()

torch.manual_seed(42)
torch.cuda.manual_seed(42)
for epoch in tqdm(range(Epochs)):

    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch % 2 == 0:
            print(f"Watching at {batch} / {len(train_dataloader)} samples")

    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    train_losses.append(train_loss)
    train_accs.append(train_acc)


    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)

            test_pred = model(X_test)
            t_loss = loss_fn(test_pred, y_test)
            test_loss += t_loss
            test_acc += accuracy_fn(y_true=y_test, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

        test_losses.append(test_loss)
        test_accs.append(test_acc)

    print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}%")

end_train = timer()
total_time = print_train_time(start=start_train, end=end_train, device=str(next(model.parameters()).device))

torch.save(model.state_dict(), 'model.pth')

import matplotlib.pyplot as plt

num_epoch = [x for x in range(0, Epochs)]
plt.subplot(1, 2, 1)
plt.plot(num_epoch, train_losses)
plt.title("Train Losses")

plt.subplot(1, 2, 2)
plt.subplot(num_epoch, train_accs)
plt.title("Train accuracy")

plt.subplot(1, 2, 1)
plt.plot(num_epoch, test_losses)
plt.title("Test Loss")

plt.subplot(1, 2, 2)
plt.subplot(num_epoch, test_accs)
plt.title("Test Accuracy")