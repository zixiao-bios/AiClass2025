import tensorboardX
import gymnasium as gym
import torch

from policy_gradient import PolicyGradient

run_name = 'pg_01'
learning_rate = 2e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.98
N = 5

env = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PolicyGradient(state_dim, hidden_dim, action_dim, learning_rate, gamma)

summary_writer = tensorboardX.SummaryWriter(f'runs/{run_name}')

for i in range(int(num_episodes / N)):
    data_N = []
    episode_reward = 0
    
    for j in range(N):
        data = {
            'states': [],
            'actions': [],
            'rewards': [],
        }
        state, _ = env.reset()
        
        finish = False
        while not finish:
            # 采样动作
            action = agent.take_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # 因为next_state是下一步的状态，而我们要计算的是当前步的损失函数
            data['states'].append(state)
            data['actions'].append(action)
            data['rewards'].append(reward)
            
            state = next_state
            episode_reward += reward
            
            finish = terminated or truncated
        
        data_N.append(data)
    
    # 更新策略网络
    agent.update(data_N)
    
    print(f'Episode {(i + 1) * N}, reward: {episode_reward / N}')
    summary_writer.add_scalar('reward', episode_reward / N, i)

# 保存模型
torch.save(agent.policy_net.state_dict(), f'{run_name}.pth')

summary_writer.close()
env.close()
