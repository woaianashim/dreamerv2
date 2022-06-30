from torch.utils.tensorboard import SummaryWriter
from time import time
from collections import deque
import numpy as np

class Logger:
    def __init__(self, log_period):
        self.writer = SummaryWriter("runs")
        self.period = log_period
        self.episode_rewards = deque(maxlen=10)
        self.loss = None
        self.start = None

    def add_loss(self, step, losses):
        self.loss = losses
        for loss in losses:
            self.writer.add_scalar("loss/"+loss, losses[loss], step)

    def add_episode(self, step, reward):
        self.episode_rewards.append(reward)
        self.writer.add_scalar("reward/reward", reward, step)

    def start_log(self, step):
        if self.start is None:
            self.start = time()
            self.first_step = step

    def log(self, step):
        if step % self.period == 0:
            if self.start is not None:
                fps = round((step - self.first_step)/(time()-self.start))
            else: 
                fps = "???"
            if len(self.episode_rewards) > 1:
                print(
                    f"Done {step} steps, {fps} FPS, last 10 episodes min/mean/median/max: %.2f/%.2f/%.2f/%.2f"
                    % (np.min(self.episode_rewards), np.mean(self.episode_rewards),
                       np.median(self.episode_rewards), np.max(self.episode_rewards), )
                )
            if self.loss is not None:
                print(self.loss)

