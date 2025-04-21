import isaacgym.gymapi
from mqe.utils import get_args
from mqe.envs.utils import make_mqe_env, custom_cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
from matplotlib import pyplot as plt
import pdb
import os

from DQN import *

def main():
    device = torch.device("cuda")

    args = get_args()
    task_name = "go1sheep-hard"
    args.num_envs = 1
    args.headless = True
    args.record_video = False

    batch_size = 64
    gamma = 0.95
    eps_start = 1
    eps_end = 0.1
    num_episodes = 100

    env, env_cfg = make_mqe_env(task_name, args, custom_cfg(args))
    # breakpoint()

    num_agents    = env.num_agents
    per_agent_obs = env.observation_space.shape[-1]   # e.g. 34
    action_dim    = env.action_space.shape[-1]        # e.g. 3
     
    n_states  = num_agents * per_agent_obs            # 2 * 34 = 68
    n_actions = num_agents * action_dim               # 2 * 3  = 6

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
    # breakpoint()
    policy_net = DQN(n_states, n_actions)
    target_net = DQN(n_states, n_actions)
    policy_net.to(device)
    target_net.to(device)
    ckpt = 'policy_net.pth'
    if os.path.isfile(ckpt):
        policy_net.load_state_dict(torch.load(ckpt))
        target_net.load_state_dict(policy_net.state_dict())
        policy_net.eval()
        print(f"→ Loaded model from {ckpt}")
    else:
        target_net.load_state_dict(policy_net.state_dict())
        optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00025, alpha=0.95)
        memory    = ReplayMemory(1_000_000, Transition)

        policy_net, reward_array = train_network(
            num_episodes, policy_net, target_net, optimizer, memory,
            env, num_agents, action_dim, n_states, batch_size,
            eps_start, eps_end, gamma, Transition
        )

        # save the trained weights
        torch.save(policy_net.state_dict(), ckpt)
        print(f"→ Training done. Model saved to {ckpt}")

        # (optional) plot your learning curve
        plt.plot(reward_array)
        plt.savefig('learning_curve.png')

    policy = lambda s: (policy_net(torch.tensor(s, dtype=torch.float32).unsqueeze(0)).max(1)[1].view(1, 1)).item()

    # env.video(policy, filename='test_herding.gif')

if __name__ == '__main__':
    main()
