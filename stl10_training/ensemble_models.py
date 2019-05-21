import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from IPython import embed

device = "cuda" if torch.cuda.is_available() else "cpu"

class ensemble_ver1(nn.Module):
    def __init__(self, input_size=1920, num_of_model=5, parameter=(64,32),
                 drop_rate=0.3, num_classes=10):
        super(ensemble_ver1, self).__init__()

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

        self.classifier = nn.Linear(parameter[1]+32, num_classes)

        self.proj = nn.Linear(180, 32)

    def forward(self, xs, x):
        batch_size = x.shape[0]
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.trans_layer(x)

        # Concat
        xs = torch.cat(xs, dim=1)
        xs = self.proj(self.dropout(xs))
        x = torch.cat((xs,x), dim=1)

        x = self.classifier(x)

        return x


# class ensemble_ver2(nn.Module):
#     def __init__(self, input_size=1920, num_of_model=5, hidden_size=128,
#                  drop_rate=0, num_classes=10):
#         super(ensemble_ver2, self).__init__()

#         self.dropout = nn.Dropout(drop_rate)
#         self.num_classes = num_classes

#         self.net = densenet.densenet201(pretrained=True)
#         self.net.classifier = nn.Linear(self.net.classifier.in_features, num_of_model)


#     def forward(self, xs, x):
#         batch_size = x.shape[0]
#         xs = torch.stack(xs).permute(1, 0, 2)

#         weight = self.net(x)
#         weight = F.softmax(weight)
#         weight = weight.unsqueeze(dim=1)

#         # Concat
#         out = torch.bmm(weight, xs).squeeze()

#         return out

# class ensemble_ver3(nn.Module):
#     def __init__(self, input_size=1920, num_of_model=5, hidden_size=128,
#                  drop_rate=0, num_classes=10):
#         super(ensemble_ver3, self).__init__()

#         self.dropout = nn.Dropout(drop_rate)
#         self.num_classes = num_classes

#         self.weight = nn.Linear(1, num_of_model)


#     def forward(self, xs, x):
#         # x is useless
#         batch_size = x.shape[0]
#         xs = torch.stack(xs)

#         # weight = self.weight(torch.ones(1).to(device))
#         weight = torch.ones(5).to(device)
#         weight = F.softmax(weight)
#         embed()
#         # Concat
#         out = torch.stack([a*b for a,b in zip(weight, xs)]).sum(dim=0)

#         return out