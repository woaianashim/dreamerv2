import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import hydra
from hydra.utils import instantiate

from time import perf_counter
import numpy as np
import cv2
import os

import imageio
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def normalize(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U).astype(np.uint8)


class Plotter:
    timer = perf_counter()
    first = True
    enable = False
    period = 10

    def __init__(self, enable=False, period=50):
        Plotter.enable = enable
        Plotter.period = period

    @staticmethod
    def plot(obs, obs_new):
        """ plot gif from sequence of observation and save in test.gif """
        if not Plotter.enable:
            return

        if perf_counter() - Plotter.timer > Plotter.period or Plotter.first:
            start_tp = perf_counter()

            #obs = obs.cpu().numpy()[:, 0, 0]
            obs = obs.cpu().numpy()[:, 0]
            #obs_new = obs_new.detach().cpu().numpy()[:, 0, 0]
            obs_new = obs_new.detach().cpu().numpy()[:, 0]
            obs = np.concatenate((obs, obs_new), axis=-2) 
            obs = obs.swapaxes(1,-1).swapaxes(1,2)
            print(obs.shape)
            obs = normalize(obs)

            plt.figure()
            plt.axis('off')
            #plt.imshow(obs[0], cmap='gray')
            #plt.imshow(obs[0])
            plt.savefig('tmp.png', bbox_inches='tight', pad_inches=0.0)
            plt.close()

            images = []
            for i in range(len(obs)):
                #plt.imshow(obs[i], cmap='gray')
                #plt.imshow(obs[i])
                plt.axis('off')
                plt.savefig('tmp.png', bbox_inches='tight', pad_inches=0.0)
                plt.close()
                images.append(normalize(mpimg.imread('tmp.png')))
            imageio.mimsave('sequence.gif', images, duration=0.1)
            os.remove('tmp.png')
            print(f'>> sequence.gif | {len(images)} frames | {perf_counter() - start_tp:.2f}s')

            Plotter.first = False
            Plotter.timer = perf_counter() 


class BaseLearner:
    def reset(self, obs):
        obs = obs.float()
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
        obs = obs.float().to(self.device).unsqueeze(0)
        state, z_state = self.world.encode(obs, self.hidden)
        action, _, _ = self.policy.step(state, deterministic)
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
        world_lr= 2e-4,
        actor_lr= 4e-5,
        critic_lr= 1e-4,
        kl_alpha=0.9,
        kl_beta=0.1,
        rho = 0.8,
        gamma = 0.999,
        gae_lambda = 0.95,
        entropy_coef = 0.,
        imagine_horizon=10,
        save_period=1000,
        observation_space=None,
        action_space=None,
        device="cuda",
    ):
        self.device = device
        self.kl_alpha = kl_alpha
        self.kl_beta = kl_beta
        self.rho = rho
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.imagine_horizon = imagine_horizon

        self.dataset = instantiate(dataset, action_space=action_space)

        self.policy = instantiate(policy, action_space=action_space).to(device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=actor_lr, eps=1e-5, weight_decay=1e-6)

        self.critic = instantiate(critic,).to(device) #TODO slow update
        self.critic_target = instantiate(critic,).to(device) #TODO slow update
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)
        self.update_counter = 0
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5, weight_decay=1e-6)

        self.world = instantiate(world, action_space=action_space).to(device)
        self.world_optim = optim.Adam(self.world.parameters(), lr=world_lr, eps=1e-5, weight_decay=1e-6)

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

        representation_loss, states, z_h_states, losses = self.representation_loss(obs, action, rew, done, hidden)

        self.world_optim.zero_grad()
        representation_loss.backward()
        nn.utils.clip_grad_norm_(self.world.parameters(), 100)
        self.world_optim.step()

        actor_loss, critic_loss, ac_losses = self.actor_critic_loss(z_h_states)
        losses.update(ac_losses)
        #im_estimates, im_preds = self.imagine_estimates(im_states, im_rews)

        #actor_loss = -im_estimates.mean()
        self.policy_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.policy.parameters(), 100)
        self.policy_optim.step()

        #critic_loss = F.mse_loss(im_estimates.detach(), im_preds)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100)
        self.critic_optim.step()
        self.update_counter += 1
        if self.update_counter == 100:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.update_counter = 0

        return losses  # TODO

    def imagine_trajectories(self, z_h_states):
        z_states, hiddens = z_h_states
        z_h_states = (z_states.view(-1, z_states.shape[-1]).detach(), hiddens.view(-1, hiddens.shape[-1]).detach())
        self.world.requires_grad_(False)
        rewards = []
        dones = []
        actions_logprobs = []
        entropies = []
        states = [torch.cat(z_h_states, dim=-1)]
        for _ in range(self.imagine_horizon):
            action, actions_logprob, entropy = self.policy.step(states[-1],)
            z_h_states, reward, done = self.world.imagine_step(z_h_states, action)
            rewards.append(reward)
            dones.append(done)
            actions_logprobs.append(actions_logprob)
            entropies.append(entropy)
            states.append(torch.cat(z_h_states, dim=-1))
        rewards = torch.stack(rewards, dim=0)
        entropies = torch.stack(entropies, dim=0)
        actions_logprobs = torch.stack(actions_logprobs, dim=0)
        dones = torch.stack(dones, dim=0)
        states = torch.stack(states, dim=0)
        self.world.requires_grad_(True)
        return states, rewards, dones, actions_logprobs, entropies


    def actor_critic_loss(self, z_h_states):
        states, rewards, dones, actions_logprobs, entropy = self.imagine_trajectories(z_h_states)

        critic_preds = self.critic(states.detach()).squeeze(-1)
        critic_target_preds = self.critic_target(states.detach()).squeeze(-1)
        self.critic.requires_grad_(False)
        lambda_targets = self.lambda_target(rewards, dones, critic_target_preds)
        self.critic.requires_grad_(True)

        critic_loss = F.mse_loss(critic_preds[:-1], lambda_targets.detach())

        actor_loss_dynamic = -lambda_targets
        actor_loss_reinforce = -actions_logprobs*(lambda_targets-critic_target_preds[:-1]).detach()
        actor_loss = self.rho*actor_loss_reinforce + (1-self.rho)*actor_loss_dynamic - self.entropy_coef*entropy
        actor_loss = actor_loss.mean()

        losses = {
                  "critic_loss": critic_loss.item(),
                  "actor_loss": actor_loss.item(),
                  "actor_loss_reinforce": actor_loss_reinforce.mean().item(),
                  "actor_loss_dynamic":actor_loss_dynamic.mean().item()
                }

        return actor_loss, critic_loss, losses

    def lambda_target(self, rewards, dones, preds):
        next_value = preds[-1]
        values = []
        for t in reversed(range(rewards.shape[0])):
            value = rewards[t] + self.gamma*(1-dones[t])*(self.gae_lambda*next_value + (1-self.gae_lambda)*preds[t])
            values.insert(0, value)
            next_value = value
        lambda_target = torch.stack(values, dim=0)
        return lambda_target


    def representation_loss(self, obs, action, rew, done, hidden): #TODO Transition loss
        obs_mean, rew_mean, done_prob, states, post_logits, prior_logits, z_h_states = self.world.eval(obs, action, hidden)

        Plotter.plot(obs, obs_mean)

        # Obs and reward reconstruction loss
        obs_loss = F.mse_loss(obs, obs_mean).mean()
        rew_loss = F.mse_loss(rew, rew_mean[1:])
        done_loss = F.binary_cross_entropy(done_prob[1:], done)
        # KL loss TODO check
        post_logits = post_logits.view(*post_logits.shape[:2], self.world.n_classes, self.world.n_states)
        post_logprobs = post_logits - post_logits.logsumexp(dim=-1, keepdim=True)
        prior_logits = prior_logits.view(*post_logits.shape[:2], self.world.n_classes, self.world.n_states)
        prior_logprobs = prior_logits - prior_logits.logsumexp(dim=-1, keepdim=True)
        post_probs = F.softmax(post_logits, dim=-1)
        KL_post = -(post_probs*prior_logprobs.detach()).sum(-1) + (post_probs*post_logprobs).sum(-1)
        KL_prior = -(post_probs.detach()*prior_logprobs).sum(-1)
        KL_loss = self.kl_alpha*KL_prior + (1-self.kl_alpha)*KL_post
        KL_loss = KL_loss.mean()

        representation_loss = obs_loss + rew_loss + done_loss + self.kl_beta*KL_loss # TODO check signs

        losses = {
                "reward": rew_loss.item(),
                "image": obs_loss.item(),
                "done": done_loss.item(),
                "kl_div": KL_loss.item(),
                "representation": representation_loss.item()
                }

        return representation_loss, states, z_h_states, losses
