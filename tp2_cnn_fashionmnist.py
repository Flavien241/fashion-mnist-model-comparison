# coding: utf8
"""TP2 - CNN sur Fashion-MNIST avec comparaison a un MLP.

Le script entraine deux modeles PyTorch sur Fashion-MNIST :
- un MLP de reference proche du TP1 ;
- un CNN simple inspire de LeNet-5.

Exemple rapide :
    python3 tp2_cnn_fashionmnist.py --epochs 5
"""

import argparse
import copy
import random
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class FashionMLP(nn.Module):
    """MLP de reference : aplatissement 28x28 puis une couche cachee."""

    def __init__(self, hidden_size=64, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeNetFashion(nn.Module):
    """CNN simple inspire de LeNet-5 pour des images 1x28x28."""

    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        # x : (batch_size, 1, 28, 28)
        x = torch.relu(self.conv1(x))
        # x : (batch_size, 6, 24, 24)
        x = self.pool(x)
        # x : (batch_size, 6, 12, 12)
        x = torch.relu(self.conv2(x))
        # x : (batch_size, 16, 8, 8)
        x = self.pool(x)
        # x : (batch_size, 16, 4, 4)
        x = x.view(x.size(0), -1)
        # x : (batch_size, 256)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # x : (batch_size, 10)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description="TP2 CNN Fashion-MNIST")
    parser.add_argument("--data-dir", default="F_MNIST_data", help="dossier des donnees")
    parser.add_argument("--epochs", type=int, default=5, help="nombre d'epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="taille des mini-batchs")
    parser.add_argument("--lr", type=float, default=1e-3, help="taux d'apprentissage")
    parser.add_argument("--hidden-size", type=int, default=64, help="taille cachee du MLP")
    parser.add_argument("--valid-size", type=int, default=10000, help="taille validation")
    parser.add_argument("--seed", type=int, default=42, help="graine aleatoire")
    parser.add_argument("--download", action="store_true", help="telecharger Fashion-MNIST si absent")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_loaders(args):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    full_train = datasets.FashionMNIST(
        args.data_dir,
        train=True,
        download=args.download,
        transform=transform,
    )
    test_set = datasets.FashionMNIST(
        args.data_dir,
        train=False,
        download=args.download,
        transform=transform,
    )

    train_size = len(full_train) - args.valid_size
    generator = torch.Generator().manual_seed(args.seed)
    train_set, valid_set = random_split(full_train, [train_size, args.valid_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def train_model(model, train_loader, valid_loader, epochs, lr, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_valid_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_examples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_examples += batch_size

        train_loss = train_loss_sum / train_examples
        train_acc = train_correct / train_examples
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "time_s": time.time() - start,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.3f} | "
            f"valid loss={valid_loss:.4f}, acc={valid_acc:.3f} | "
            f"{row['time_s']:.1f}s"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def run_experiment(name, model, loaders, args, device):
    train_loader, valid_loader, test_loader = loaders
    print(f"\n=== {name} ===")
    print(f"Parametres entrainables : {count_parameters(model):,}")
    model, history = train_model(model, train_loader, valid_loader, args.epochs, args.lr, device)
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    best_valid_acc = max(row["valid_acc"] for row in history)
    return {
        "modele": name,
        "parametres": count_parameters(model),
        "best_valid_acc": best_valid_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device utilise :", device)

    loaders = build_loaders(args)
    results = [
        run_experiment("MLP 1 couche cachee", FashionMLP(args.hidden_size), loaders, args, device),
        run_experiment("CNN LeNetFashion", LeNetFashion(), loaders, args, device),
    ]

    print("\nComparaison finale")
    for row in results:
        print(
            f"{row['modele']:<22} | params={row['parametres']:>7} | "
            f"best val acc={row['best_valid_acc']:.3f} | "
            f"test loss={row['test_loss']:.4f} | test acc={row['test_acc']:.3f}"
        )

    delta = results[1]["test_acc"] - results[0]["test_acc"]
    print(f"\nGain CNN vs MLP sur le test : {delta:+.3f} point d'accuracy.")


if __name__ == "__main__":
    main()
