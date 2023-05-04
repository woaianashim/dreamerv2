import torch
from torch import jit, nn
import torch.distributions as dist
import torch.nn.functional as F
from typing import Optional

from hydra.utils import instantiate



class RSSMModel(jit.ScriptModule):
    def __init__(self,state_dim=32*32, action_dim=6, embed_dim=1536, hidden_dim=1024, n_classes=32, n_states=32):
        super().__init__()
        self.quant = 0.
        self.n_classes = n_classes
        self.n_states = n_states
        self.hidden_dim = hidden_dim

        self.register_buffer("init_hidden", torch.zeros(hidden_dim))
        self.register_buffer("init_state", torch.zeros(state_dim))
        self.register_buffer("init_feat", torch.zeros(hidden_dim))

        # Representation model
        self.representation = nn.Sequential(nn.Linear(embed_dim+hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU(),
                                            nn.Linear(hidden_dim, state_dim))
        self.prior_representation = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU(),
                                            nn.Linear(hidden_dim, state_dim))

        # Transition model
        self.transition = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim), nn.ELU(),) 
        self.gru_layer = nn.Sequential(nn.Linear(2*hidden_dim, 3*hidden_dim),
                                       nn.LayerNorm(3*hidden_dim))
        self.embed = nn.Conv1d(1024, 1024, 1, groups=32, bias=False)


    @jit.script_method
    def forward(self, embed: torch.Tensor, action: torch.Tensor,
                hidden: Optional[torch.Tensor] = None, last_z_state: Optional[torch.Tensor] = None):
        T, B = embed.shape[:2]
        T = T-1
        print(embed.shape)

        hidden = self.init_hidden[None].repeat(B, 1) if hidden is None else hidden
        last_z_state = self.init_state[None].repeat(B, 1) if last_z_state is None else last_z_state
        prior_feat = self.init_feat[None].repeat(B, 1) if last_z_state is None else last_z_state

        post_logits = []
        prior_logits = []
        z_states = []
        hiddens = []

        zero_action = torch.zeros_like(action[:1])
        action = torch.cat([zero_action, action], dim=0)
        print(action.shape)

        for t in range(T+1):
            prior_feat = self.transition(torch.cat([action[t], last_z_state], dim=-1))
            hidden = self.gru_update(prior_feat, hidden)
            prior_logit = self.prior_representation(hidden)
            post_logit = self.representation(torch.cat([hidden, embed[t]], dim=-1))
            last_z_state = self.straight_sample(post_logit)
            post_logits.append(post_logit)
            prior_logits.append(prior_logit)
            z_states.append(last_z_state)
            hiddens.append(hidden)

        prior_logits = torch.stack(prior_logits, dim=0)
        post_logits = torch.stack(post_logits, dim=0)
        z_states = torch.stack(z_states, dim=0)
        hiddens = torch.stack(hiddens, dim=0)
        full_states = torch.cat([z_states, hiddens], dim=-1)
        return full_states, prior_logits, post_logits, (z_states, hiddens)

    @jit.script_method
    def gru_update(self, prior_feat: torch.Tensor, state: torch.Tensor):
        reset, cand, update = self.gru_layer(torch.cat([prior_feat, state], dim=-1)).split(self.hidden_dim, -1)
        reset = torch.sigmoid(reset)
        cand = F.elu(reset*cand)
        update = torch.sigmoid(update - 1.)
        return update*cand + (1-update)*state


    def straight_sample(self, logits):
        b, c = logits.shape
        logits = logits.view(b, self.n_classes, self.n_states)

        probs = F.softmax(logits, dim=-1)
        probs = probs.view(-1, self.n_states)
        inds = probs.multinomial(1).squeeze(-1)
        samples = F.one_hot(inds, self.n_states).to(probs)
        samples = samples.view(b, c)
        classes = (samples + probs.view(b, c) - probs.view(b, c).detach())
        return classes

    def imagine_step(self, z_h_state, action):
        z_state, hidden = z_h_state
        prior_feat = self.transition(torch.cat([action, z_state], dim=-1))
        new_hidden = self.gru_update(prior_feat, hidden)
        prior_logit = self.prior_representation(new_hidden)
        new_z_state = self.straight_sample(prior_logit)
        return (new_z_state, new_hidden)



class Encoder(nn.Module):
    def __init__(self, hid=48, n_channels=3):
        super().__init__()
        self.n_channels = n_channels
        self.encoder = nn.Sequential(
                                     nn.Conv2d(n_channels, hid, 4, 2), nn.ELU(),
                                     nn.Conv2d(hid, 2*hid, 4, 2), nn.ELU(),
                                     nn.Conv2d(2*hid, 4*hid, 4, 2), nn.ELU(),
                                     nn.Conv2d(4*hid, 8*hid, 4, 2), nn.ELU(),
                                    )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, obs):
        t, b, c, h, w = obs.shape #TODO batch first?
        imgs = obs.view(b*t, c, h, w)
        state = self.encoder(imgs).view(t, b, -1)
        return state

class Decoder(nn.Module):
    def __init__(self, input_dim=2048, hid=48, n_channels=3):
        super().__init__()
        self.n_channels = n_channels
        self.pre_linear = nn.Linear(input_dim, 32*hid, bias=False)
        self.model = nn.Sequential(
                                     nn.ConvTranspose2d(32*hid, 4*hid, 5, 2), nn.ELU(),
                                     nn.ConvTranspose2d(4*hid, 2*hid, 5, 2), nn.ELU(),
                                     nn.ConvTranspose2d(2*hid, hid, 6, 2), nn.ELU(),
                                     nn.ConvTranspose2d(hid, n_channels, 6, 2), 
                )

        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, state):
        state = self.pre_linear(state)
        T, B, C = state.shape
        state = state.view(T*B, C, 1, 1)
        mean = self.model(state)
        mean = mean.view(T, B, self.n_channels, 64, 64)
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
    def __init__(self, state_dim, hidden_dim, feat_dim, conv_hid, embed_dim, n_classes, n_states, mlp_depth, action_space=None, n_channels=3):
        super().__init__()
        self.n_classes = n_classes
        self.n_states = n_states
        self.encoder = Encoder(hid=conv_hid, n_channels=n_channels)
        self.decoder = Decoder(input_dim=state_dim+hidden_dim, hid=conv_hid, n_channels=n_channels)
        self.rssm = RSSMModel(state_dim=state_dim, action_dim=action_space.n, embed_dim=embed_dim, hidden_dim=hidden_dim, n_classes=n_classes, n_states=n_states)
        self.reward = MLP(state_dim+hidden_dim, feat_dim, 1, mlp_depth, activation=nn.Tanh())
        self.done = MLP(state_dim+hidden_dim, feat_dim, 1, mlp_depth, activation=nn.Sigmoid())


    def eval(self, obs, action, hidden=None):
        embed = self.encoder(obs)
        states, prior_logits, post_logits, z_h_states = self.rssm(embed, action, hidden=hidden)

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

