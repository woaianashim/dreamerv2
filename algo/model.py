import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteQ(nn.Module):
    def __init__(self, observation_space=None, action_space=None,
                 dueling=True, noisy=False,
                 n_atoms=1, min_reward=None, max_reward=None):
        """
        Initialize a deep Q-learning network as described in
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
        Arguments:
            in_channels: number of channel of input.
                i.e The number of most recent frames stacked together as describe in the paper
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DiscreteQ, self).__init__()
        self.dueling = dueling
        self.noisy = noisy
        self.n_atoms = n_atoms
        self.n_actions = action_space.n
        self.min_reward = min_reward
        self.max_reward = max_reward
        if n_atoms>1:
            self.support = nn.Parameter(torch.linspace(min_reward, max_reward, n_atoms), requires_grad=False)

        Linear = NoisyLinear if noisy else nn.Linear
        self.backbone = nn.Sequential(
                                      nn.Conv2d(observation_space.shape[0],
                                                32, kernel_size=8, stride=4), nn.ReLU(),
                                      nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                                      nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
                                     )
        self.head = nn.Sequential(Linear(7 * 7 * 64, 512), nn.ReLU(), Linear(512, action_space.n * n_atoms))
        if self.dueling:
            self.dueling_head = nn.Sequential(Linear(7*7*64, 512), nn.ReLU(), Linear(512, n_atoms))

    def forward(self, x):
        b = x.size(0)
        feats = self.backbone(x/255.).view(b, -1)
        values = self.head(feats).view(b, self.n_actions, self.n_atoms)
        if self.dueling:
            values -= values.mean(1, keepdim=True)
            values += self.dueling_head(feats).view(b, 1, self.n_atoms)
        if self.n_atoms>1:
            logits = values
            q_values = logits.softmax(-1) * self.support
            q_values = q_values.sum(-1)
        else:
            logits = None
            q_values = values.squeeze(-1)
        return q_values, logits

class NoisyLinear(nn.Module):
    #### Maybe not original for simplicity
    def __init__(self, in_features, out_features, sigma=0.5):
        super().__init__()
        self.sigma = sigma
        self.mean_linear = nn.Linear(in_features, out_features)
        self.noise_linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        weight = self.mean_linear.weight
        bias = self.mean_linear.bias
        weight = weight + self.noise_linear.weight * torch.randn_like(weight) * self.sigma
        bias = bias + self.noise_linear.bias * torch.randn_like(bias) * self.sigma
        return F.linear(x, weight, bias)

class DiscreteQRAM(nn.Module):
    def __init__(self, in_features=4, num_actions=18):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(DiscreteQRAM, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
