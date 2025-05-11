# --- USER VARIABLES ---
batch_size = 64
test_batch_size = 1000
epochs = 10  # Adjust for speed/accuracy
learning_rate = 0.01
log_steps = 100
plot_results = True
random_seed = 0

# --- IMPORTS ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import random_split

torch.manual_seed(random_seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# --- DATA LOADING ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
# Split train_set into two disjoint sets for two clients
total_train = len(train_set)
client1_len = total_train // 2
client2_len = total_train - client1_len
client1_set, client2_set = random_split(train_set, [client1_len, client2_len], generator=torch.Generator().manual_seed(random_seed))
client1_loader = torch.utils.data.DataLoader(client1_set, batch_size=batch_size, shuffle=True)
client2_loader = torch.utils.data.DataLoader(client2_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

# --- FULL MODEL DEFINITION (LeNet5 variant) ---
class LeNetSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.block3(x)
        return x

# --- SPLIT POINTS DEFINITION ---
def get_split_models(split_point):
    # Returns (client_model, server_model) for a given split_point (1-5)
    class Client1(nn.Module):  # After block1
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
            )
        def forward(self, x): return self.block1(x)
    class Server1(nn.Module):
        def __init__(self):
            super().__init__()
            self.block2 = nn.Sequential(
                nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
            )
            self.block3 = nn.Sequential(
                nn.Linear(256, 120), nn.ReLU(),
                nn.Linear(120, 84), nn.ReLU(),
                nn.Linear(84, 10)
            )
        def forward(self, x):
            x = self.block2(x)
            x = x.view(x.size(0), -1)
            x = self.block3(x)
            return x

    class Client2(nn.Module):  # After block2[0] (2nd Conv)
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
            )
            self.conv2 = nn.Conv2d(6, 16, 5)
        def forward(self, x):
            x = self.block1(x)
            x = self.conv2(x)
            return x
    class Server2(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)
            self.block3 = nn.Sequential(
                nn.Linear(256, 120), nn.ReLU(),
                nn.Linear(120, 84), nn.ReLU(),
                nn.Linear(84, 10)
            )
        def forward(self, x):
            x = self.relu2(x)
            x = self.pool2(x)
            x = x.view(x.size(0), -1)
            x = self.block3(x)
            return x

    class Client3(nn.Module):  # After block2 (2nd Conv+ReLU+Pool)
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
            )
        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            return x
    class Server3(nn.Module):
        def __init__(self):
            super().__init__()
            self.block3 = nn.Sequential(
                nn.Linear(256, 120), nn.ReLU(),
                nn.Linear(120, 84), nn.ReLU(),
                nn.Linear(84, 10)
            )
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.block3(x)
            return x

    class Client4(nn.Module):  # After block3[0] (first FC)
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
            )
            self.fc1 = nn.Linear(256, 120)
        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            return x
    class Server4(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(120, 84)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(84, 10)
        def forward(self, x):
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x

    class Client5(nn.Module):  # After block3[2] (second FC)
        def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 6, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
            )
            self.fc1 = nn.Linear(256, 120)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(120, 84)
        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            return x
    class Server5(nn.Module):
        def __init__(self):
            super().__init__()
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(84, 10)
        def forward(self, x):
            x = self.relu2(x)
            x = self.fc3(x)
            return x

    splits = [
        (Client1, Server1),
        (Client2, Server2),
        (Client3, Server3),
        (Client4, Server4),
        (Client5, Server5)
    ]
    return splits[split_point-1][0](), splits[split_point-1][1]()

split_names = [
    "After 1st Conv+ReLU+Pool",
    "After 2nd Conv",
    "After 2nd Conv+ReLU+Pool",
    "After 1st FC",
    "After 2nd FC"
]

# --- TRAINING & EVALUATION LOOP FOR ALL SPLITS ---
results = []
for split_idx, split_name in enumerate(split_names):
    print(f"\n=== Split {split_idx+1}: {split_name} ===")
    # Each client gets its own model and optimizer; server is shared
    client1_model, server_model = get_split_models(split_idx+1)
    client2_model, _ = get_split_models(split_idx+1)  # server_model is shared
    client1_model, client2_model, server_model = client1_model.to(device), client2_model.to(device), server_model.to(device)
    client1_optimizer = optim.SGD(client1_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    client2_optimizer = optim.SGD(client2_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    server_optimizer = optim.SGD(server_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    loss_criterion = nn.CrossEntropyLoss()

    train_losses, test_losses, test_accuracies = [], [], []

    for epoch in range(1, epochs + 1):
        client1_model.train()
        client2_model.train()
        server_model.train()
        running_loss = 0.0
        # Alternate batches between clients
        for (batch1, batch2) in zip(client1_loader, client2_loader):
            for client_model, client_optimizer, (inputs, labels) in [
                (client1_model, client1_optimizer, batch1),
                (client2_model, client2_optimizer, batch2)
            ]:
                inputs, labels = inputs.to(device), labels.to(device)
                client_optimizer.zero_grad()
                server_optimizer.zero_grad()
                # --- CLIENT FORWARD ---
                split_activations = client_model(inputs)
                split_activations = split_activations.detach().requires_grad_()
                # --- SERVER FORWARD ---
                outputs = server_model(split_activations)
                loss = loss_criterion(outputs, labels)
                # --- SERVER BACKWARD ---
                loss.backward()
                # --- SEND GRADIENT TO CLIENT, CLIENT BACKWARD ---
                split_grads = split_activations.grad
                split_activations.backward(split_grads)
                # --- OPTIMIZER STEPS ---
                client_optimizer.step()
                server_optimizer.step()
                running_loss += loss.item()
        avg_train_loss = running_loss / (len(client1_loader) + len(client2_loader))
        train_losses.append(avg_train_loss)

        # --- EVALUATION (use client1 for test, or average both) ---
        client1_model.eval()
        server_model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                split_activations = client1_model(inputs)
                outputs = server_model(split_activations)
                loss = loss_criterion(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        avg_test_loss = test_loss / total
        accuracy = correct / total
        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)
        print(f"Epoch {epoch}: Test Loss: {avg_test_loss:.4f}, Test Accuracy: {accuracy*100:.2f}%")

    results.append({
        "split": split_name,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies
    })

# --- PLOTTING ---
if plot_results:
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(16,6))
    for i, res in enumerate(results):
        plt.subplot(1,2,1)
        plt.plot(epochs_range, res["test_losses"], label=res["split"])
        plt.xlabel('Epoch')
        plt.ylabel('Test Loss')
        plt.title('Test Loss for Different Split Points')
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(epochs_range, res["test_accuracies"], label=res["split"])
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy for Different Split Points')
        plt.legend()
    plt.tight_layout()
    plt.show()

# --- SAVE RESULTS TO CSV ---
csv_rows = []
for res in results:
    split = res["split"]
    for epoch, (loss, acc) in enumerate(zip(res["test_losses"], res["test_accuracies"]), 1):
        csv_rows.append({
            "split_name": split,
            "epoch": epoch,
            "test_loss": loss,
            "test_accuracy": acc
        })
df = pd.DataFrame(csv_rows)
csv_filename = "split_learning_results.csv"
df.to_csv(csv_filename, index=False)
print(f"Results saved to {csv_filename}")
