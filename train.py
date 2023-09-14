import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
from PIL import ImageFile
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms


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
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
    
device = "cuda" if torch.cuda.is_available() else "cpu"

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
learning_rate = 0.001
batch_size = 64
num_epochs = 20

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()])
dataset = Type123Dataset(
    csv_file='drive/MyDrive/training/fortypes.csv', root_dir='drive/MyDrive/train/', transform=tfm)
train_set_size = int(len(dataset)*0.8)
test_set_size = len(dataset) - train_set_size
train_ds, val_ds = random_split(dataset, [train_set_size, test_set_size])
train_dl = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# -----------
# Train
for epoch in range(num_epochs):

    model.train()
    losses = []

    for batch_idx, (data, target) in enumerate(train_dl):
        data = data.to(device)
        target = target.to(device)

        y_pred = model(data)

        loss = loss_fn(y_pred, target)

        losses.append(loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    print(f"Epoch [{epoch}], Loss {(sum(losses)/len(losses)):.4f}")


    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.inference_mode():
        for x, y in val_dl:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, pred = scores.max(1)
            num_correct += (pred == y).sum()
            num_samples += pred.size(0)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: accuracy -> {(float(num_correct)/float(num_samples)):.4f}")

torch.save(model.state_dict(), 'model.pth')