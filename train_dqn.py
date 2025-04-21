import os
# Ensure gymapi loads before torch to satisfy IsaacGym dependency order
try:
    import isaacgym.gymapi
except ImportError:
    pass

# Import mqe utils (loads gymapi internally)
from mqe.utils import get_args
from mqe.envs.utils import make_mqe_env, custom_cfg

# Now safe to import torch and related modules
def main():
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    import numpy as np
    import random
    from collections import namedtuple, deque
    from DQN import DQN, ReplayMemory

    # Hyperparameters
    num_episodes = 500
    batch_size    = 64
    gamma         = 0.95
    eps_start     = 1.0
    eps_end       = 0.1
    target_update = 1000
    memory_size   = 100000
    lr            = 1e-3
    # Discretization settings
    bins_per_dim  = 3  # discrete values [-1, 0, 1]

    # Setup args and environment
    args = get_args()
    args.num_envs     = 1
    args.headless     = True
    args.record_video = False
    env, cfg = make_mqe_env("go1sheep-hard", args, custom_cfg(args))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Action dimensions
    action_dim = env.action_space.shape[0]  # e.g., 3 high-level controls
    n_joint    = bins_per_dim ** action_dim

    # Function to map discrete index to continuous action
    def idx_to_action(idx: int) -> torch.Tensor:
        coords = []
        for d in reversed(range(action_dim)):
            div = bins_per_dim ** d
            a = (idx // div) % bins_per_dim
            coords.append(a - (bins_per_dim // 2))  # center at 0 -> [-1,0,1]
        # shape (1, action_dim)
        return torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)

    # Flatten observation
    def flatten_obs(obs: torch.Tensor) -> torch.Tensor:
        # obs returned from wrapper: shape (num_envs*agents, obs_dim)
        # We only learn for the first agent (go1), which is obs[0]
        go1_obs = obs[0:1]
        return go1_obs.view(1, -1)

    # Setup DQN
    Transition = namedtuple('Transition', ('state','action','next_state','reward','done'))
    obs0 = env.reset()
    state_dim = flatten_obs(obs0).shape[1]
    policy_net = DQN(state_dim, n_joint).to(device)
    target_net = DQN(state_dim, n_joint).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory    = ReplayMemory(memory_size, Transition)

    steps = 0
    for ep in range(num_episodes):
        obs = env.reset()
        state = flatten_obs(obs).to(device)
        done  = False
        total_reward = 0.0

        while not done:
            # Epsilon-greedy
            eps = max(eps_end, eps_start - (eps_start-eps_end)*(steps/(num_episodes*500)))
            if random.random() < eps:
                idx = random.randrange(n_joint)
            else:
                with torch.no_grad():
                    idx = policy_net(state).argmax().item()

            action = idx_to_action(idx)
            obs, reward, done, _ = env.step(action)
            next_state = flatten_obs(obs).to(device)
            # Handle possibly batched reward tensor (pick first env)
        if isinstance(reward, torch.Tensor):
            # extract scalar for env0 and ensure shape [1]
            r = reward.flatten()[0].unsqueeze(0).to(device)
        else:
            r = torch.tensor([reward], device=device)
            d = bool(done)

            # Store transition
            memory.push(state, torch.tensor([[idx]], device=device), next_state, r, d)

            # Move
            state = next_state
            total_reward += r.item()
            steps += 1

            # Learn
            if len(memory) >= batch_size:
                batch = Transition(*zip(*memory.sample(batch_size)))
                s_b = torch.cat(batch.state)
                a_b = torch.cat(batch.action)
                r_b = torch.cat(batch.reward)
                ns_b= torch.cat(batch.next_state)
                dones = torch.tensor(batch.done, device=device)

                q_vals = policy_net(s_b).gather(1, a_b)
                with torch.no_grad():
                    max_ns = target_net(ns_b).max(1)[0]
                    y = r_b + gamma * max_ns * (~dones)
                loss = F.mse_loss(q_vals.squeeze(), y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {ep+1}/{num_episodes}, Reward: {total_reward:.2f}, Eps: {eps:.3f}")

    # Save model & demos
    os.makedirs('models', exist_ok=True)
    torch.save(policy_net.state_dict(), 'models/dqn_mqe.pth')

    # Collect expert demonstrations
    demos = []
    obs = env.reset()
    done = False
    while not done:
        state = flatten_obs(obs).to(device)
        with torch.no_grad(): idx = policy_net(state).argmax().item()
        action = idx_to_action(idx)
        demos.append({'obs': state.cpu().numpy(), 'action': action.cpu().numpy()})
        obs, _, done, _ = env.step(action)

    import numpy as _np
    _np.save('expert_demos.npy', demos)
    print("Expert demonstrations saved to expert_demos.npy")

if __name__ == '__main__':
    main()
