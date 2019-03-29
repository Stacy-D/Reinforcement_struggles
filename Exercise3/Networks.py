import torch.nn as nn
import torch.nn.functional as F

# Define your neural networks in this class.
# Use the __init__ method to define the architecture of the network
# and define the computations for the forward pass in the forward method.

class ValueNetwork(nn.Module):
    def __init__(self, num_states=15, n_actions=4):
        super(ValueNetwork, self).__init__()
        self.num_states = num_states

        self.conv_1 = nn.Conv1d(1, 2, 3, 2)
        self.fc_2 = nn.Linear(14, 32)
        self.fc_3 = nn.Linear(32, n_actions)

    def forward(self, inputs):
        inputs_tr = inputs.view(1, 1, self.num_states)
        y = F.relu(self.conv_1(inputs_tr))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc_2(y))
        return self.fc_3(y)
