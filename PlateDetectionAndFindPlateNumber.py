import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import os
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

# Define the neural network model
class PlateDetector(nn.Module):
    def __init__(self):
        super(PlateDetector, self).__init__()
        self.fc1 = nn.Linear(65536, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(-1, 65536)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set up the data loaders
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
train_dataset = datasets.ImageFolder(root='/home/moath/python/plates/train/', transform=data_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataset = datasets.ImageFolder(root='/home/moath/python/plates/test/', transform=data_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# Train the neural network
model = PlateDetector().to(torch.device('cpu'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        print(inputs.shape, labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch {epoch+1}, batch {i+1}: loss {running_loss/100:.3f}')
            running_loss = 0.0

# Test the neural network
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Save the tested image
        for i in range(len(images)):
            if predicted[i] == 0:
                # Create directory if it doesn't exist
                if not os.path.exists("predicted/0"):
                    os.makedirs("predicted/0")
                plt.imsave(f"predicted/0/{i}.png", images[i].squeeze(), cmap="gray")
            else:
                # Create directory if it doesn't exist
                if not os.path.exists("predicted/1"):
                    os.makedirs("predicted/1")
                plt.imsave(f"predicted/1/{i}.png", images[i].squeeze(), cmap="gray")

                # Extract plate number as text
                img = Image.fromarray(images[i].squeeze().numpy())
                plate_number = pytesseract.image_to_string(img)
                print(f"Plate Number: {plate_number}")

print(f'Test accuracy: {correct/total:.3f}')