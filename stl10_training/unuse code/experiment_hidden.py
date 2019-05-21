import argparse
import pathlib

import sys
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
from IPython import embed

import base_models
from data import get_train_val_split, get_test
import numpy as np
import ensemble_models

MODEL_NAMES = [f"base{i+1}.pth" for i in range(5)]
PARAMETERS = [(64,32), (87,78), (77,33), (123,21), (212, 16)]

# For reproducibility
seed = 2002
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# TODO: Put into command line args
num_classes = 10
batch_size = 32
num_epochs = 300
base_lr = 1e-4
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
pathlib.Path('./snapshots').mkdir(parents=True, exist_ok=True)


def train(ensemble_model, trained_models, train_loader, optimizer, criterion, epoch):
    acc = []
    for i, batch_data in enumerate(train_loader):
        optimizer.zero_grad()

        image = batch_data[0].to(device) # torch.Size([batch, 3, 96, 96])
        label = batch_data[1].to(device) # torch.Size([batch])

        # feat = [mod.get_last_hidden(image) for mod in trained_models]
        # feat = [Variable(f, requires_grad=False) for f in feat]
        feat=None
        output = ensemble_model(feat, image) # torch.Size([batch, num_classes])

        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        acc.extend((output.max(dim=1)[1] == label).cpu().numpy())
        print("\r[-] Epoch {0}, iter {1}/{2}, loss: {3:.3f}, acc: {4:.3f} ".format(
            epoch, i, len(train_loader), float(loss.data), np.mean(acc)), end="")


    print("")

def validate(ensemble_model, trained_models, val_loader, epoch):
    ''' Performs validation on the provided ensemble_model, returns validation accuracy '''
    ensemble_model.eval()
    correct = 0
    total = 0
    for i, batch_data in enumerate(val_loader):
        image = batch_data[0].to(device) # torch.Size([batch, 3, 96, 96])
        label = batch_data[1].to(device) # torch.Size([batch])

        # feat = [mod.get_last_hidden(image) for mod in trained_models]
        # feat = [Variable(f, requires_grad=False) for f in feat]
        feat = None
        output = ensemble_model(feat, image) # torch.Size([batch, num_classes])

        preds = torch.argmax(output, dim=1)
        correct += int(torch.sum(preds == label)) # Add number of correct predictions to running total
        total += int(label.size(0)) # Add batch size to total
        print("\r[-] Validation epoch {0}, iter {1}/{2}, accuracy: {3:.3f}".format(epoch, i, len(val_loader), correct / total), end="")

    val_accuracy = correct / total
    ensemble_model.train(True)
    print("")

    return val_accuracy

def test(ensemble_model, trained_models, test_loader):
    ensemble_model.eval()
    correct = 0
    total = 0

    for i, batch_data in enumerate(test_loader):
        image = batch_data[0].to(device) # torch.Size([batch, 3, 96, 96])
        label = batch_data[1].to(device) # torch.Size([batch])

        feat = [mod.get_last_hidden(image) for mod in trained_models]
        feat = [Variable(f, requires_grad=False) for f in feat]

        output = ensemble_model(feat, image) # torch.Size([batch, num_classes])

        preds = torch.argmax(output, dim=1)
        correct += int(torch.sum(preds == label)) # Add number of correct predictions to running total
        total += int(label.size(0)) # Add batch size to total

    ensemble_model.train(True)
    print("[*] Accuracy: {}".format(correct / total))

def go_test():
    print("="*30)
    # Load a model pretrained on ImageNet
    ensemble_model = ensemble_models.ensemble_ver1()
    # Replace classification layer
    ensemble_model.to(device)
    ensemble_model.load_state_dict(torch.load(f"snapshots/base_experiment_hidden.pth"))

    trained_models = [base_models.base_classifier(parameter=PARAMETERS[i]) for i in range(5)]
    # Replace classification layer
    for i, model in enumerate(trained_models):
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        model.to(device)
        # Load best performing model based on validation score
        model.load_state_dict(torch.load(f"snapshots/{MODEL_NAMES[i]}"))


    print("[-] Performing testing...")


    test_loader = get_test(batch_size)
    test(ensemble_model, trained_models, test_loader)

def main():
    ##################
    # Initialization #
    ##################

    ensemble_model = base_models.base_classifier(parameter=PARAMETERS[0]) #ensemble_models.ensemble_ver1()
    print(ensemble_model)
    ensemble_model.to(device)

    trained_models = [base_models.base_classifier(parameter=PARAMETERS[i]) for i in range(5)]
    for i, model in enumerate(trained_models):
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        model.to(device)
        model.load_state_dict(torch.load(f"snapshots/{MODEL_NAMES[i]}"))

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
        with torch.no_grad():
            val_accuracy = validate(ensemble_model, trained_models, val_loader, epoch)
            print("[*] Validation accuracy: {}".format(val_accuracy))
            val_accuracy2 = validate(ensemble_model, trained_models, val_loader2, epoch)
            print("[*] Validation2 accuracy: {}\n".format(val_accuracy2))

        # New best performing model
        if val_accuracy2 > best_val_accuracy:
            best_val_accuracy = val_accuracy2
            print("[*] New best accuracy!\n")
            torch.save(ensemble_model.state_dict(), f"snapshots/base_experiment_hidden.pth")

    ###########
    # Testing #
    ###########
    print("="*30)
    print("[-] Performing testing...")

    # Load best performing model based on validation score
    ensemble_model.load_state_dict(torch.load(f"snapshots/base_experiment_hidden.pth"))

    test_loader = get_test(batch_size)
    test(ensemble_model, trained_models, test_loader)

if __name__ == "__main__":
    if sys.argv[2] == "1":
        go_test()
    else:
        main()