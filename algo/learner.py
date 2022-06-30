import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import hydra
from hydra.utils import instantiate


class BaseLearner:
    def reset(self, obs):
        obs = torch.tensor(obs, dtype=torch.float)
        self.hidden = self.world.rssm.init_hidden.unsqueeze(0)
        self.dataset.reset(obs, self.hidden.squeeze(0))

    def store_step(self, **kwargs):
        for key in kwargs:
            kwargs[key] = (
                kwargs[key].clone().detach().cpu()
                if isinstance(kwargs[key], torch.Tensor)
                else torch.tensor(kwargs[key], dtype=torch.float)
            )
        kwargs["hidden"] = self.hidden.squeeze(0).detach().cpu()
        self.dataset.add_step(**kwargs)

    def __call__(self, obs, deterministic=False):
        obs = torch.tensor(obs, dtype=torch.float).to(self.device).unsqueeze(0)
        state, z_state = self.world.encode(obs, self.hidden)
        action = self.policy.step(state, deterministic)
        self.hidden = self.world.step(self.hidden, z_state, action)
        return action.detach().cpu().argmax()

    @property
    def full(self):
        return self.dataset.full

    def sample(self):
        data = self.dataset.sample()
        for key in data:
            data[key] = data[key].to(self.device)
        return data

    def update_target(self, hard=False):
        tau = self.tau if not hard else 1.0
        for par, targ_par in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            targ_par.data = targ_par.data * (1.0 - tau) + par.data * tau

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save(self, step):
        if step > self.last_save + self.save_period:
            self.last_save = step
            state = {
                "step": step,
                "policy": {
                    "model": self.policy.state_dict(),
                    "optim": self.policy_optim.state_dict(),
                },
                "critic": {
                    "model": self.critic.state_dict(),
                    "target": self.target_critic.state_dict(),
                    "optim": self.critic_optim.state_dict(),
                },
                "world": {
                    "model": self.world.state_dict(),
                    "optim": self.world_optim.state_dict(),
                },
            }
            torch.save(state, "checkpoint.pt")

    def load(self, path):
        state = torch.load(path)
        self.policy.load_state_dict(state["policy"]["model"])
        self.policy_optim.load_state_dict(state["policy"]["optim"])
        self.critic.load_state_dict(state["critic"]["model"])
        self.target_critic.load_state_dict(state["critic"]["target"])
        self.critic_optim.load_state_dict(state["critic"]["optim"])
        self.world.load_state_dict(state["world"]["model"])
        self.world_optim.load_state_dict(state["world"]["optim"])
        return state["step"]


class DreamLearner(BaseLearner):
    def __init__(
        self,
        dataset,
        policy,
        critic,
        world,
        lr=0.001,
        kl_alpha=0.9,
        save_period=1000,
        observation_space=None,
        action_space=None,
        device="cuda",
    ):
        self.device = device
        self.kl_alpha = kl_alpha

        self.dataset = instantiate(dataset, action_space=action_space)

        self.policy = instantiate(policy, action_space=action_space).to(device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)

        self.critic = instantiate(critic, action_space=action_space).to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        self.target_critic = instantiate(critic, action_space=action_space).to(device)

        self.world = instantiate(world, action_space=action_space).to(device)
        self.world_optim = optim.Adam(self.world.parameters(), lr=lr)

        self.last_save = 0
        self.save_period = save_period

    def learn_step(self):
        data = self.sample()
        obs, rew, action, done, hidden = (
            data["obs"],
            data["rew"],
            data["action"],
            data["done"],
            data["hidden"],
        )

        representation_loss, states, losses = self.representation_loss(obs, action, rew, done, hidden)

        self.world_optim.zero_grad()
        representation_loss.backward()
        self.world_optim.step()

        #im_states, im_rews = self.imagine_trajectories(states)
        #im_estimates, im_preds = self.imagine_estimates(im_states, im_rews)

        #actor_loss = -im_estimates.mean()
        #self.actor_optim.zero_grad()
        #actor_loss.backward()
        #self.actor_optim.step()

        #critic_loss = F.mse_loss(im_estimates.detach(), im_preds)
        #self.critic_optim.zero_grad()
        #critic_loss.backward()
        #self.critic_optim.step()

        return losses  # TODO

    def imagine_trajectories(states):
        pass #TODO

    def imagine_estimates(self, states, rews):
        preds = self.critic(states)

        # GAE estimates
        estimates = 0  # TODO

        return estimates, preds

    def representation_loss(self, obs, action, rew, done, hidden): #TODO Transition loss
        obs_mean, rew_mean, done_prob, states, post_logits, prior_logits = self.world.eval(obs, action, hidden)

        # Obs and reward reconstruction loss
        obs_loss = F.mse_loss(obs, obs_mean)
        rew_loss = F.mse_loss(rew, rew_mean)
        done_loss = F.binary_cross_entropy(done_prob, done)
        # KL loss TODO check
        post_probs = F.softmax(post_logits, dim=-1)
        prior_logprobs = F.log_softmax(prior_logits, dim=-1)
        KL_post = (post_probs*prior_logprobs.detach()).sum(-1)
        KL_prior = (post_probs.detach()*prior_logprobs).sum(-1)
        KL_loss = self.kl_alpha*KL_prior + (1-self.kl_alpha)*KL_post
        KL_loss = KL_loss.mean()

        representation_loss = obs_loss + rew_loss + done_loss - KL_loss # TODO check signs

        losses = {
                "reward": rew_loss.item(),
                "image": obs_loss.item(),
                "done": done_loss.item(),
                "kl_div": KL_loss.item(),
                "representation": representation_loss.item()
                }

        return representation_loss, states, losses
