import argparse
import pathlib

import sys
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from IPython import embed

import base_models
from data import get_train_val_split, get_test
import numpy as np

MODEL_NAMES = [f"base{i+1}.pth" for i in range(6)]
PARAMETERS = [(64,32), (87,78), (77,33), (123,21), (212, 16), (212, 16)]

MODEL_NAME = MODEL_NAMES[int(sys.argv[1])]
PARAMETER = PARAMETERS[int(sys.argv[1])]

# For reproducibility
seed = 2002
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
np.random.seed(seed)

# TODO: Put into command line args
num_classes = 10
batch_size = 32
num_epochs = 300
base_lr = 1e-4
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
pathlib.Path('./snapshots').mkdir(parents=True, exist_ok=True)


def train(model, train_loader, optimizer, criterion, epoch):
    acc = []
    for i, batch_data in enumerate(train_loader):
        optimizer.zero_grad()

        image = batch_data[0].to(device) # torch.Size([batch, 3, 96, 96])
        label = batch_data[1].to(device) # torch.Size([batch])

        output = model(image) # torch.Size([batch, num_classes])

        acc.extend((output.max(dim=1)[1] == label).cpu().numpy())

        loss = criterion(output, label)
        print("\r[-] Epoch {}, iter {}/{}, loss: {:.3f}, acc: {:.3f}".format(
            epoch, i, len(train_loader), float(loss.data), np.mean(acc)), end="")
        loss.backward()
        optimizer.step()

    print("")

def validate(model, val_loader, epoch):
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
        print("\r[-] Validation epoch {}, iter {}/{}, accuracy: {:.3f}".format(epoch, i, len(val_loader), correct / total), end="")

    val_accuracy = correct / total
    print("")
    model.train(True)

    return val_accuracy

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    ans, out = [], []
    for i, batch_data in enumerate(test_loader):
        image = batch_data[0].to(device) # torch.Size([batch, 3, 96, 96])
        label = batch_data[1].to(device) # torch.Size([batch])

        output = model(image) # torch.Size([batch, num_classes])

        preds = torch.argmax(output, dim=1)
        correct += int(torch.sum(preds == label)) # Add number of correct predictions to running total
        total += int(label.size(0)) # Add batch size to total

        ans.extend(label.cpu().numpy().tolist())
        out.extend(output.cpu().numpy().tolist())

    np.save(f"ans/out_{int(sys.argv[1])}.npy", out)
    np.save(f"ans/ans.npy", ans)
    model.train(True)
    print("[*] Accuracy: {}".format(correct / total))

def go_test():
    print("="*30)
    model = base_models.base_classifier(parameter=PARAMETER)
    model.to(device)
    model.load_state_dict(torch.load(f"snapshots/{MODEL_NAME}"))

    # print("[-] Performing validation...")
    # train_loader, val_loader, val_loader2 = get_train_val_split(batch_size)

    # with torch.no_grad():
    #     val_accuracy = validate(model, val_loader, -1)
    #     print("[*] Validation accuracy: {}".format(val_accuracy))
    #     val_accuracy2 = validate(model, val_loader2, -1)
    #     print("[*] Validation2 accuracy: {}\n".format(val_accuracy2))

    print("[-] Performing testing...")

    test_loader = get_test(batch_size)
    with torch.no_grad():
        test(model, test_loader)

def main():
    ##################
    # Initialization #
    ##################

    model = base_models.base_classifier(parameter=PARAMETER)
    model.to(device)
    print(model)

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
        with torch.no_grad():
            val_accuracy = validate(model, val_loader, epoch)
            print("[*] Validation accuracy: {}".format(val_accuracy))
            val_accuracy2 = validate(model, val_loader2, epoch)
            print("[*] Validation2 accuracy: {}\n".format(val_accuracy2))

        # New best performing model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            print("[*] New best accuracy!\n")
            torch.save(model.state_dict(), f"snapshots/{MODEL_NAME}")

    ###########
    # Testing #
    ###########
    print("="*30)
    print("[-] Performing testing...")

    # Load best performing model based on validation score
    model.load_state_dict(torch.load(f"snapshots/{MODEL_NAME}"))

    test_loader = get_test(batch_size)
    with torch.no_grad():
        test(model, test_loader)

if __name__ == "__main__":
    if sys.argv[2] == "1":
        go_test()
    else:
        main()