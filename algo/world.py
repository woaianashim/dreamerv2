import torch
from torch import jit, nn
import torch.distributions as dist
import torch.nn.functional as F
from typing import Optional

from hydra.utils import instantiate



class RSSMModel(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, feat_dim, embed_dim, hidden_dim, n_classes=32, n_states=32):
        super().__init__()
        self.n_classes = n_classes
        self.n_states = n_states

        # Recurrent model
        self.recurrent_mlp = nn.Sequential(nn.Linear(state_dim+action_dim, feat_dim),
                                           nn.LayerNorm(feat_dim),
                                           nn.ELU())
        self.recurrent_gru = nn.GRUCell(feat_dim, hidden_dim)
        self.register_buffer("init_hidden", torch.zeros(hidden_dim))
        #self.register_buffer("init_state", torch.zeros(state_dim))

        # Representation model
        self.representation = nn.Sequential(nn.Linear(hidden_dim+embed_dim, feat_dim), nn.LayerNorm(feat_dim), nn.ELU(),
                                            nn.Linear(feat_dim, state_dim))

        # Transition model
        self.transition = nn.Sequential(nn.Linear(hidden_dim, feat_dim), nn.LayerNorm(feat_dim), nn.ELU(),
                                        nn.Linear(feat_dim, state_dim))


    @jit.script_method
    def forward(self, action: torch.Tensor, embed: torch.Tensor, hidden: Optional[torch.Tensor] = None,):
        T, B = action.shape[:2]

        hidden = self.init_hidden[None].repeat(B, 1) if hidden is None else hidden

        post_logits = []
        prior_logits = []
        z_states = []
        hiddens = [hidden]

        for t in range(T+1):
            prior_logit = self.transition(hidden)
            prior_logits.append(prior_logit)
            post_logit = self.representation(torch.cat([embed[t], hidden], dim=-1))
            post_logits.append(post_logit)
            last_z_state = self.straight_sample(post_logit)
            z_states.append(last_z_state)
            # Update hidden
            if t<T:
                feats = self.recurrent_mlp(torch.cat([last_z_state, action[t]], dim=-1))
                hidden = self.recurrent_gru(feats, hidden)
                hiddens.append(hidden)

        prior_logits = torch.stack(prior_logits, dim=0)
        post_logits = torch.stack(post_logits, dim=0)
        z_states = torch.stack(z_states, dim=0)
        hiddens = torch.stack(hiddens, dim=0)
        full_states = torch.cat([z_states, hiddens], dim=-1)
        return full_states, prior_logits, post_logits, (z_states, hiddens)

    def imagine_step(self, z_h_state, action):
        z_state, hidden = z_h_state
        feats = self.recurrent_mlp(torch.cat([z_state, action], dim=-1))
        new_hidden = self.recurrent_gru(feats, hidden)
        prior_logit = self.transition(new_hidden)
        new_z_state = self.straight_sample(prior_logit)
        return (new_z_state, new_hidden)



    @torch.no_grad()
    def encode_step(self, embed, hidden,):
        post_logit = self.representation(torch.cat([embed, hidden], dim=-1))
        z_state = self.straight_sample(post_logit)
        full_state = torch.cat([z_state, hidden], dim=-1)
        return full_state, z_state

    @torch.no_grad()
    def next_hidden(self, hidden, z_state, action):
        feats = self.recurrent_mlp(torch.cat([z_state, action], dim=-1))
        new_hidden = self.recurrent_gru(feats, hidden)
        return new_hidden


    def straight_sample(self, logits):
        b, c = logits.shape
        logits = logits.view(b, self.n_classes, self.n_states)
        probs = F.softmax(logits, dim=-1)
        probs = probs.view(-1, self.n_states)
        inds = probs.multinomial(1).squeeze(-1)
        samples = F.one_hot(inds, self.n_states).to(probs)
        samples = samples.view(b, c)
        return samples + probs.view(b, c) - probs.view(b, c).detach()





class Encoder(nn.Module):
    def __init__(self, hid=32):
        super().__init__()
        self.encoder = nn.Sequential(
                                     nn.Conv2d(4, hid, 4, 2), nn.ELU(),
                                     nn.Conv2d(hid, 2*hid, 4, 2), nn.ELU(),
                                     nn.Conv2d(2*hid, 4*hid, 4, 2), nn.ELU(),
                                     nn.Conv2d(4*hid, 8*hid, 4, 2), nn.ELU(),
                                    )

    def forward(self, obs):
        t, b, c, h, w = obs.shape #TODO batch first?
        imgs = obs.view(b*t, c, h, w)
        state = self.encoder(imgs).view(t, b, -1)
        return state

class Decoder(nn.Module):
    def __init__(self, input_dim, hid=32,):
        super().__init__()
        self.pre_linear = nn.Linear(input_dim, 32*hid)
        self.model = nn.Sequential(
                                     nn.ConvTranspose2d(32*hid, 4*hid, 5, 2), nn.ReLU(),
                                     nn.ConvTranspose2d(4*hid, 2*hid, 5, 2), nn.ReLU(),
                                     nn.ConvTranspose2d(2*hid, hid, 6, 2), nn.ReLU(),
                                     nn.ConvTranspose2d(hid, 4, 6, 2), 
                )

    def forward(self, state):
        state = self.pre_linear(state)
        T, B, C = state.shape
        state = state.view(T*B, C, 1, 1)
        mean = self.model(state)
        mean = mean.view(T, B, 4, 64, 64)
        return mean

class MLP(nn.Module):
    def __init__(self, input_dim, feat_dim, output_dim, depth, activation=nn.Identity()):
        super().__init__()
        layers = [nn.Linear(input_dim, feat_dim), nn.LayerNorm(feat_dim, 1e-3), nn.ELU(),]
        for _ in range(depth-1):
            layers += [nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim, 1e-3), nn.ELU(),]
        layers += [nn.Linear(feat_dim, output_dim), activation]
        self.module = nn.Sequential(*layers)
    def forward(self, x):
        return self.module(x)

class WorldModel(nn.Module):
    def __init__(self, state_dim, hidden_dim, feat_dim, conv_hid, embed_dim, n_classes, n_states, mlp_depth, action_space=None):
        super().__init__()
        self.n_classes = n_classes
        self.n_states = n_states
        self.encoder = Encoder(hid=conv_hid)
        self.decoder = Decoder(input_dim=state_dim+hidden_dim, hid=conv_hid)
        self.rssm = RSSMModel(state_dim, action_space.n, feat_dim, embed_dim, hidden_dim, n_classes=n_classes, n_states=n_states)
        self.reward = MLP(state_dim+hidden_dim, feat_dim, 1, mlp_depth, activation=nn.Tanh())
        self.done = MLP(state_dim+hidden_dim, feat_dim, 1, mlp_depth, activation=nn.Sigmoid())


    def eval(self, obs, action, hidden=None):
        embed = self.encoder(obs)
        states, prior_logits, post_logits, z_h_states = self.rssm(action, embed, hidden=hidden)

        obs_mean = self.decoder(states)
        rew_mean = self.reward(states).squeeze(-1)
        done_prob = self.done(states).squeeze(-1)

        return obs_mean, rew_mean, done_prob, states, post_logits, prior_logits, z_h_states

    def encode(self, obs, hidden,):
        embed = self.encoder(obs.unsqueeze(0)).squeeze(0)
        state, z_state = self.rssm.encode_step(embed, hidden,)
        return state, z_state

    def step(self, hidden, z_state, action):
        return self.rssm.next_hidden(hidden, z_state, action)

    def imagine_step(self, z_h_state, action):
        new_z_h_state = self.rssm.imagine_step(z_h_state, action)
        states = torch.cat(new_z_h_state, dim=-1)
        rew_mean = self.reward(states).squeeze(-1)
        rew_sample = rew_mean + 0.08*torch.randn_like(rew_mean)
        done_prob = self.done(states).squeeze(-1)
        # TODO how to sample done?
        return new_z_h_state, rew_sample, done_prob

