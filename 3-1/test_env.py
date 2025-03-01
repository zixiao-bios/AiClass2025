import gymnasium as gym


# 创建环境
env = gym.make("CartPole-v1", render_mode="human")

# 重置环境（初始化 S0）
obs, _ = env.reset()

# obs 是一个 numpy 数组，表示环境的观测值
print(type(obs))
print(obs.shape)
print(obs)

# action 是一个整数，表示动作
print(env.action_space)
action = env.action_space.sample()
print(action)

# 使用随机策略执行一个回合
tot_reward = 0
finished = False
while not finished:
    # 随机选择一个动作
    action = env.action_space.sample()
    
    # 执行动作
    obs, reward, terminated, truncated, _ = env.step(action)
    
    # 累计奖励
    tot_reward += reward
    
    finished = terminated or truncated

print(f"terminated: {terminated}, truncated: {truncated}")
print(f"reward: {tot_reward}")

# 关闭环境
env.close()
