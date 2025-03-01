import torch
import gymnasium as gym

from model import PolicyNet


hidden_dim = 128
model_weights = 'xx.pth'

# 创建环境
env = gym.make("CartPole-v1", render_mode="human", max_episode_steps=2000)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 加载模型
model = PolicyNet(state_dim, hidden_dim, action_dim)
model.load_state_dict(torch.load(model_weights))
model.eval()

# 初始化环境
state, _ = env.reset()

# 运行环境
tot_reward = 0
finish = False
while not finish:
    state = torch.tensor([state], dtype=torch.float)
    
    # 预测动作
    probs = model(state)
    
    # 从预测的动作概率分布中采样
    # action_dist = torch.distributions.Categorical(probs)
    # action = action_dist.sample()
    
    # 或直接选择概率最大的动作
    action = torch.argmax(probs)
    
    # 执行动作
    state, reward, terminated, truncated, _ = env.step(action.item())
    
    finish = terminated or truncated
    tot_reward += reward

env.close()
print(f'Total reward: {tot_reward}')
