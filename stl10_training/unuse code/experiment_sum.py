import argparse
import pathlib

import sys
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from IPython import embed

import densenet
from data import get_train_val_split, get_test
import numpy as np
import ensemble

# MODEL_NAME = sys.argv[1]
MODEL_NAMES = [f"test{i+1}" for i in range(5)]

# For reproducibility
seed = 2002
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# TODO: Put into command line args
num_classes = 10
batch_size = 20
num_epochs = 10
base_lr = 1e-5
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
pathlib.Path('./snapshots').mkdir(parents=True, exist_ok=True)

# Computes polynomial decay for learning rate
def lr_poly(base_lr, i, max_i, power=0.95):
    return base_lr * ((1-i/max_i) ** power)

def train(ensemble_model, trained_models, train_loader, optimizer, criterion, epoch):
    acc = []
    for i, batch_data in enumerate(train_loader):
        # Update lr with polynomial decay
        # lr = lr_poly(base_lr, i, len(train_loader))
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        optimizer.zero_grad()

        image = batch_data[0].to(device)
        label = batch_data[1].to(device)

        feat = [mod(image) for mod in trained_models]
        feat = [Variable(f, requires_grad=False) for f in feat]

        output = ensemble_model(feat, image)

        acc.extend((output.max(dim=1)[1] == label).cpu().numpy())

        loss = criterion(output, label)
        print("\r[-] Epoch {0}, iter {1}/{2}, loss: {3:.3f}, acc: {4:.3f} ".format(
            epoch, i, len(train_loader), float(loss.data), np.mean(acc)), end="")
        loss.backward()
        optimizer.step()

    # Save model
    print("")
    # save_path = "snapshots/densenet201_epoch{0}.pth".format(epoch)
    # print("Saving model to {}...".format(save_path))
    # torch.save(model.state_dict(), save_path)

def validate(ensemble_model, trained_models, val_loader, epoch):
    ''' Performs validation on the provided model, returns validation accuracy '''
    correct = 0
    total = 0
    for i, batch_data in enumerate(val_loader):
        image = batch_data[0].to(device)
        label = batch_data[1].to(device)

        feat = [mod(image) for mod in trained_models]
        feat = [Variable(f, requires_grad=False) for f in feat]

        output = ensemble_model(feat, image)

        preds = torch.argmax(output, dim=1)
        correct += int(torch.sum(preds == label)) # Add number of correct predictions to running total
        total += int(label.size(0)) # Add batch size to total
        print("\r[-] Validation epoch {0}, iter {1}/{2}, accuracy: {3:.3f}".format(epoch, i, len(val_loader), correct / total), end="")

    val_accuracy = correct / total
    print("")

    return val_accuracy

def test(ensemble_model, trained_models, test_loader):
    ensemble_model.eval()
    correct = 0
    total = 0

    for i, batch_data in enumerate(test_loader):
        image = batch_data[0].to(device)
        label = batch_data[1].to(device)

        feat = [mod(image) for mod in trained_models]
        feat = [Variable(f, requires_grad=False) for f in feat]

        output = ensemble_model(feat, image)

        preds = torch.argmax(output, dim=1)
        correct += int(torch.sum(preds == label)) # Add number of correct predictions to running total
        total += int(label.size(0)) # Add batch size to total

    print("[*] Accuracy: {}".format(correct / total))

def go_test():
    print("="*30)
    ensemble_model = ensemble.ensemble_ver2()
    # Replace classification layer
    ensemble_model.to(device)

    trained_models = [densenet.densenet201(pretrained=True) for i in range(5)]
    # Replace classification layer
    for i, model in enumerate(trained_models):
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        model.to(device)
        # Load best performing model based on validation score
        model.load_state_dict(torch.load(f"snapshots/densenet201_best_{MODEL_NAMES[i]}.pth"))


    print("[-] Performing testing...")


    test_loader = get_test(batch_size)
    test(model, trained_models, test_loader)

def main():
    ##################
    # Initialization #
    ##################

    # Load a model pretrained on ImageNet
    ensemble_model = ensemble.ensemble_ver2()
    # Replace classification layer
    ensemble_model.to(device)

    trained_models = [densenet.densenet201(pretrained=True) for i in range(5)]
    # Replace classification layer
    for i, model in enumerate(trained_models):
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        model.to(device)
        # Load best performing model based on validation score
        model.load_state_dict(torch.load(f"snapshots/densenet201_best_{MODEL_NAMES[i]}.pth"))

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
        train(ensemble_model, trained_models, train_loader, optimizer, criterion, epoch)

        # Validate
        val_accuracy = validate(ensemble_model, trained_models, val_loader, epoch)
        print("[*] Validation accuracy: {}".format(val_accuracy))
        val_accuracy2 = validate(ensemble_model, trained_models, val_loader2, epoch)
        print("[*] Validation2 accuracy: {}\n".format(val_accuracy2))

        # New best performing model
        if val_accuracy2 > best_val_accuracy:
            best_val_accuracy = val_accuracy2
            print("[*] New best accuracy!\n")
            torch.save(ensemble_model.state_dict(), f"snapshots/densenet201_experiment_sum.pth")

    ###########
    # Testing #
    ###########
    print("="*30)
    print("[-] Performing testing...")

    # Load best performing model based on validation score
    ensemble_model.load_state_dict(torch.load(f"snapshots/densenet201_experiment_sum.pth"))

    test_loader = get_test(batch_size)
    test(ensemble_model, trained_models, test_loader)

if __name__ == "__main__":
    if sys.argv[2] == "1":
        go_test()
    else:
        main()