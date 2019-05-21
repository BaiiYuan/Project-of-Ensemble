import argparse
import pathlib

import sys
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from IPython import embed

import densenet
from data import get_train_val_split, get_test
import numpy as np

MODEL_NAME = sys.argv[1]

# For reproducibility
seed = 2002
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
np.random.seed(seed)

# TODO: Put into command line args
num_classes = 10
batch_size = 32
num_epochs = 10
base_lr = 1e-4
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
pathlib.Path('./snapshots').mkdir(parents=True, exist_ok=True)

# Computes polynomial decay for learning rate
def lr_poly(base_lr, i, max_i, power=0.95):
    return base_lr * ((1-i/max_i) ** power)

def train(model, train_loader, optimizer, criterion, epoch):
    acc = []
    for i, batch_data in enumerate(train_loader):
        # Update lr with polynomial decay
        lr = lr_poly(base_lr, i, len(train_loader))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        optimizer.zero_grad()

        image = batch_data[0].to(device) # torch.Size([batch, 3, 96, 96])
        label = batch_data[1].to(device) # torch.Size([batch])

        output = model(image) # torch.Size([batch, num_classes])

        acc.extend((output.max(dim=1)[1] == label).cpu().numpy())

        loss = criterion(output, label)
        print("\r[-] Epoch {0}, iter {1}/{2}, loss: {3:.3f}, acc: {4:.3f}, lr: {5:.5f}".format(
            epoch, i, len(train_loader), float(loss.data), np.mean(acc), lr), end="")
        loss.backward()
        optimizer.step()

    # Save model
    print("")
    # save_path = "snapshots/densenet201_epoch{0}.pth".format(epoch)
    # print("Saving model to {}...".format(save_path))
    # torch.save(model.state_dict(), save_path)

def validate(model, val_loader, epoch):
    ''' Performs validation on the provided model, returns validation accuracy '''
    model.eval()
    correct = 0
    total = 0
    for i, batch_data in enumerate(val_loader):
        image = batch_data[0].to(device) # torch.Size([batch, 3, 96, 96])
        label = batch_data[1].to(device) # torch.Size([batch])

        output = model(image) # torch.Size([batch, num_classes])

        preds = torch.argmax(output, dim=1)
        correct += int(torch.sum(preds == label)) # Add number of correct predictions to running total
        total += int(label.size(0)) # Add batch size to total
        print("\r[-] Validation epoch {0}, iter {1}/{2}, accuracy: {3:.3f}".format(epoch, i, len(val_loader), correct / total), end="")

    val_accuracy = correct / total
    print("")
    model.train(True)

    return val_accuracy

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    ans = []
    for i, batch_data in enumerate(test_loader):
        image = batch_data[0].to(device) # torch.Size([batch, 3, 96, 96])
        label = batch_data[1].to(device) # torch.Size([batch])
        ans.extend(label.cpu().numpy())
        output = model(image) # torch.Size([batch, num_classes])

        preds = torch.argmax(output, dim=1)
        correct += int(torch.sum(preds == label)) # Add number of correct predictions to running total
        total += int(label.size(0)) # Add batch size to total

    np.save("ans.npy", ans)
    model.train(True)
    print("[*] Accuracy: {}".format(correct / total))

def go_test():
    print("="*30)
    model = densenet.densenet201(pretrained=True)
    # Replace classification layer
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.to(device)
    # Load best performing model based on validation score
    model.load_state_dict(torch.load(f"snapshots/densenet201_best_{MODEL_NAME}.pth"))

    print("[-] Performing validation...")
    train_loader, val_loader, val_loader2 = get_train_val_split(batch_size)

    with torch.no_grad():
        val_accuracy = validate(model, val_loader, -1)
        print("[*] Validation accuracy: {}".format(val_accuracy))
        val_accuracy2 = validate(model, val_loader2, -1)
        print("[*] Validation2 accuracy: {}\n".format(val_accuracy2))

    print("[-] Performing testing...")

    test_loader = get_test(batch_size)
    with torch.no_grad():
        test(model, test_loader)

def main():
    ##################
    # Initialization #
    ##################


    # Load a model pretrained on ImageNet
    model = densenet.densenet201(pretrained=True)
    # Replace classification layer
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    ############
    # Training #
    ############
    train_loader, val_loader, val_loader2 = get_train_val_split(batch_size)

    # Use model that performs best on validation for testing
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        # Train
        print("="*30)
        train(model, train_loader, optimizer, criterion, epoch)

        # Validate
        val_accuracy = validate(model, val_loader, epoch)
        print("[*] Validation accuracy: {}".format(val_accuracy))
        val_accuracy2 = validate(model, val_loader2, epoch)
        print("[*] Validation2 accuracy: {}\n".format(val_accuracy2))

        # New best performing model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print("[*] New best accuracy!\n")
            torch.save(model.state_dict(), f"snapshots/densenet201_best_{MODEL_NAME}.pth")

    ###########
    # Testing #
    ###########
    print("="*30)
    print("[-] Performing testing...")

    # Load best performing model based on validation score
    model.load_state_dict(torch.load(f"snapshots/densenet201_best_{MODEL_NAME}.pth"))

    test_loader = get_test(batch_size)
    with torch.no_grad():
        test(model, test_loader)

if __name__ == "__main__":
    if sys.argv[2] == "1":
        go_test()
    else:
        main()