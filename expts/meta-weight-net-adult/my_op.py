import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, length1, length2, classes):
        super(MLP, self).__init__()


        self.classifier = nn.Sequential(
            nn.Linear(input_size, length1),
            nn.ReLU(),
            nn.Linear(length1, length2),
            nn.ReLU(),
            nn.Linear(length2, classes),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.classifier(x)
        return x