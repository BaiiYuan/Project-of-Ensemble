import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from IPython import embed
import densenet

device = "cuda" if torch.cuda.is_available() else "cpu"

class base_classifier(nn.Module):
    def __init__(self, input_size=1920, num_of_model=5, parameter=(64,32),
                 drop_rate=0.3, num_classes=10):
        super(base_classifier, self).__init__()

        self.dropout = nn.Dropout(drop_rate)
        self.num_classes = num_classes

        self.features = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0),
                                      nn.BatchNorm2d(8),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                                      nn.Dropout(drop_rate),
                                      nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                                      nn.Dropout(drop_rate)
                                      )
        self.trans_layer = nn.Sequential(nn.Linear(7056, parameter[0]),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(drop_rate),
                                         nn.Linear(parameter[0], parameter[1]),
                                         nn.ReLU(inplace=True),
                                         )

        self.classifier = nn.Linear(parameter[1], num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.trans_layer(x)
        x = self.classifier(x)
        return x

    def get_last_hidden(self, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.trans_layer(x)
        return x