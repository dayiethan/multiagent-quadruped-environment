import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from matplotlib import pyplot as plt
import pdb

class DQN(nn.Module):

    def __init__(self, n_states, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_states, 64)
        self.layer2 = nn.Linear(64, 64)
        self.out = nn.Linear(64,n_actions)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.tanh(self.layer1(x))
        x = F.tanh(self.layer2(x))
        x = self.out(x)
        return x

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.memory = deque([], maxlen=capacity)
        self.Transition = Transition

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def eps_decay(itr,num_episodes, eps_start, eps_end):
    if itr>=num_episodes*100:
        return eps_end
    else:
        return (eps_end - eps_start)*itr/(num_episodes*100) + eps_start

# def action(state, eps, n_actions, policy_net):
#     p = np.random.random()
#     if p < eps:
#         a = torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)
#     else:
#         a = policy_net(state).max(1)[1].view(1, 1)

#     return a

def select_action(state, policy_net, eps, num_agents, action_dim):
    # breakpoint()
    batch_size = state.size(0)
    flat_q = policy_net(state)                        # (B, num_agents*action_dim)
    q      = flat_q.view(batch_size, num_agents, action_dim)

    if random.random() < eps:
        idx = torch.randint(0, action_dim, (batch_size, num_agents))
    else:
        idx = q.argmax(dim=2)                         # (B, num_agents)

    a_oh = F.one_hot(idx, num_classes=action_dim).float()
    return a_oh, idx


def update_network(policy_net, target_net, optimizer, memory, batch_size, gamma):
    if len(memory) < batch_size:
        return

    # sample once
    batch = memory.Transition(*zip(*memory.sample(batch_size)))

    device           = next(policy_net.parameters()).device
    state_batch      = torch.cat(batch.state).to(device)
    action_batch     = torch.cat(batch.action).to(device)
    raw_reward       = torch.cat(batch.reward, dim=0).to(device)
    reward_batch     = raw_reward.sum(dim=1)   
    next_state_batch = torch.cat(batch.next_state).to(device)

    raw_done = torch.cat(batch.done, dim=0).to(device)            # maybe (B,) or (B, A)
    done_mask = raw_done.view(batch_size, -1).any(dim=1)          # (B,)

    # Q(s,a)
    act_flat = action_batch.view(batch_size, -1)
    Q_all    = policy_net(state_batch)
    Q_s_a    = (Q_all * act_flat).sum(dim=1)

    with torch.no_grad():
        next_max, _ = target_net(next_state_batch).max(dim=1)

    y = reward_batch + gamma * next_max * (~done_mask).float()

    loss = nn.functional.mse_loss(Q_s_a, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_network(num_episodes, policy_net, target_net, optimizer, memory,
                  env, num_agents, action_dim, n_states, batch_size,
                  eps_start, eps_end, gamma, Transition):

    device = next(policy_net.parameters()).device
    if hasattr(env, 'action_scale'):
        env.action_scale = torch.as_tensor(env.action_scale, dtype=torch.float32, device=device)
    reward_array = np.zeros(num_episodes)
    itr = 0

    for ep in range(num_episodes):
        # reset returns a numpy array (num_envs, n_states)
        raw_s = env.reset()
        s     = raw_s.view(raw_s.size(0), -1).float().to(device)
        # s = env.reset().float().to(device)
        done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

        while not done.any():
            eps = eps_decay(itr, num_episodes, eps_start, eps_end)
            a_oh, a_idx = select_action(s, policy_net, eps,
                                        num_agents, action_dim)
            a_oh = a_oh.to(device)

            # step the MQE env
            # → returns torch tensors: obs (N, n_states), reward (N,), done (N,), info
            if not isinstance(env.action_scale, torch.Tensor) or env.action_scale.device != device:
                env.action_scale = torch.as_tensor(env.action_scale, dtype=torch.float32, device=device)
            obs, reward, done, info = env.step(a_oh)
            obs    = obs.to(device)
            reward = reward.to(device)
            done   = done.to(device)


            # accumulate discounted reward
            reward_array[ep] += (gamma**itr) * reward.sum().item()

            # push transition (we flatten batch of envs into individual transitions)
            raw_obs = obs
            obs     = raw_obs.view(raw_obs.size(0), -1).float().to(device)
            for b in range(env.num_envs):
                memory.push(
                    s[b:b+1],         # (1, total_obs)
                    a_oh[b:b+1],      # (1, num_agents, action_dim)
                    obs[b:b+1],       # (1, total_obs)
                    reward[b:b+1],    # (1,)
                    done[b:b+1]       # (1,)
                )

            s = obs   # next state
            itr += 1

            update_network(policy_net, target_net, optimizer, memory, batch_size, gamma)

            if itr % 1_000 == 0:
                target_net.load_state_dict(policy_net.state_dict())

    return policy_net, reward_array

# def train_network_without_targ(num_episodes, policy_net, target_net, optimizer, memory, env,
#                                 n_actions, n_states, batch_size, eps_start, eps_end, gamma, Transition):
#     reward_array = np.zeros(num_episodes)
#     itr = 0
#     for i_episode in range(num_episodes):
#         # Initialize the environment and get it's state
#         s = env.reset()
#         s = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
#         done = False
#         time = 0
#         while not done:
#             eps = eps_decay(itr,num_episodes, eps_start, eps_end)
#             a = action(s, eps, n_actions, policy_net)
#             s_next, r, done = env.step(a.item())
#             r = torch.tensor([r])

#             reward_array[i_episode] += gamma**(time)*r
#             time += 1

#             s_next = torch.tensor(s_next, dtype=torch.float32).unsqueeze(0)

#             memory.push(s, a, s_next, r, done)

#             # Move to the next state
#             s = s_next

#             # Perform one step of the optimization (on the policy network)
#             update_network(policy_net, target_net, optimizer, memory, env, eps, batch_size, Transition, gamma)

#             itr+=1

#             #if itr%1000 == 0:
#             target_net.load_state_dict(policy_net.state_dict())

#     return policy_net, reward_array
