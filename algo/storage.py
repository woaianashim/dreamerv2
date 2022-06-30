import numpy as np
from random import randint, choice
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

class EpisodeBuffer:
    keys = ["obs", "rew", "action", "done",]

    def add_step(self, obs, rew, action, done, hidden):
        self.size += 1
        self.obs.append(torch.tensor(obs).cpu())
        self.action.append(torch.tensor(action).cpu())
        self.rew.append(torch.tensor(rew).cpu())
        self.done.append(torch.tensor(done).cpu())
        self.hidden.append(torch.tensor(hidden).cpu())

    def reset(self, obs=None, hidden=None):
        if obs is None:
            obs = self.obs.pop()
            hidden = self.hidden.pop()
        self.size = 0
        for key in self.keys:
            self[key] = []
        self.hidden = []
        self.obs.append(obs.cpu())
        self.hidden.append(hidden.cpu())

    def bake(self):
        episode = {}
        for key in self.keys:
            episode[key] = torch.stack(self[key], dim=0)
        episode["hidden"] = torch.stack(self.hidden, dim=0)
        episode["size"] = self.size
        self.reset()
        return episode

    def __getitem__(self, key):
        if key in self.keys or key == "hidden":
            return self.__getattribute__(key)

    def __setitem__(self, key, val):
        if key in self.keys or key == "hidden":
            return self.__setattr__(key, val)

class EpisodeStorage(IterableDataset):
    def __init__(self, maxsize,
                 batch_len,
                 observation_space=None, action_space=None):
        self.maxsize = maxsize
        self.total_steps = 0
        self.batch_len = batch_len

        self.episodes = []
        self.oldest_episode_id = 0
        self.current_episode = EpisodeBuffer()
        self.num_actions = action_space.n

    def __len__(self):
        return self.total_steps

    def add_step(self, obs, rew, action, done, hidden):
        self.current_episode.add_step(obs, rew, action, done, hidden)
        if done:
            episode = self.current_episode.bake()
            if episode["size"] >= self.batch_len:
                if self.total_steps>=self.maxsize:
                    self.total_steps -= self.episodes[self.oldest_episode_id]["size"]
                    self.total_steps += episode["size"]
                    self.episodes[self.oldest_episode_id] = episode
                    self.oldest_episode_id += 1
                    self.oldest_episode_id %= len(self.episodes)
                else:
                    self.episodes.append(episode)
                    self.total_steps += episode["size"]

    def reset(self, obs, hidden):
        self.current_episode.reset(obs, hidden)

    def __iter__(self):
        i = 0
        while True:
            episode = choice(self.episodes)
            start = randint(0, episode["size"])
            start = min(start, episode["size"]-self.batch_len)
            i += 1
            data = {k: episode[k][start:start+self.batch_len] for k in EpisodeBuffer.keys}
            data["hidden"] = episode["hidden"][start]
            data["action"]= F.one_hot(data["action"].long(), self.num_actions)

            yield data

class DreamerDataset:
    def __init__(self, maxsize, batch_size, batch_len,
                 minimal_size, num_workers=5, action_space=None):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.minimal_size = minimal_size

        self.full = False


        self.storage = EpisodeStorage(maxsize, batch_len, action_space=action_space)
        self.dataloader = DataLoader(self.storage, batch_size=batch_size, num_workers=num_workers)

    def add_step(self, obs=None, rew=None, action=None, done=None, hidden=None):
        self.storage.add_step(obs, rew, action, done, hidden)
        if not self.full and len(self.storage)>self.minimal_size:
            dataloader = DataLoader(self.storage, self.batch_size, num_workers=self.num_workers)
            self.sampler = iter(dataloader)
            self.full = True

    def sample(self):
        sample = next(self.sampler)
        sample["obs"] = sample["obs"].transpose(0,1).contiguous()
        sample["action"] = sample["action"].transpose(0,1)
        sample["rew"] = sample["rew"].transpose(0,1)
        sample["done"] = sample["done"].transpose(0,1)
        return sample

    def reset(self, obs, hidden):
        self.storage.reset(obs, hidden)
        
if __name__ == '__main__':
    pass
