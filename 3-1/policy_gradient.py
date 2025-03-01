import torch

from model import PolicyNet


class PolicyGradient:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device='cpu'):
        self.policy_net = PolicyNet(state_dim, hidden_dim,
                                    action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                          lr=learning_rate)  # 使用Adam优化器
        self.gamma = gamma  # 折扣因子
        self.device = device

    def take_action(self, state):  # 根据动作概率分布随机采样
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        
        # 通过神经网络计算动作概率
        probs = self.policy_net(state)
        
        # 从动作概率分布中采样
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        return action.item()

    def update(self, tracj_list):
        # 总体的损失
        tot_loss = 0

        for transition_dict in tracj_list:
            reward_list = transition_dict['rewards']
            state_list = transition_dict['states']
            action_list = transition_dict['actions']

            # G = R1 + γR2 + γ^2R3 + ... + γ^(n-1)Rn，表示从当前状态开始的折扣累计奖励
            G = 0
            
            self.optimizer.zero_grad()
            for i in reversed(range(len(reward_list))):  # 从最后一步算起
                # 第 i 步的状态、动作、奖励
                reward = reward_list[i]
                state = torch.tensor([state_list[i]],
                                    dtype=torch.float).to(self.device)
                action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
                
                # 计算当前动作的概率的对数
                # gather(1, action)表示取出action对应的概率
                log_prob = torch.log(self.policy_net(state).gather(1, action))
                
                # 计算折扣累计奖励
                G = self.gamma * G + reward
                
                # 当前步的函数
                tot_loss += -log_prob * G

        # 梯度下降
        tot_loss.backward()
        self.optimizer.step()
