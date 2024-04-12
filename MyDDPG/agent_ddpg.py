import torch
import torch.nn as nn
from collections import deque
import numpy as np
import random
import torch.optim as optim

# Hyper-parameters
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TAU = 5e-3 # update target network

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device type:",device)

## critic net
## actor net    =>   target

## relay bugffer

### ddpg_agent   => 上面三个都放入

class Actor(nn.Module):
    def __init__(self,state_dim,action_dim,hidden_dim=64):
        super(Actor,self).__init__()
        # 全连接层
        self.fc1 = nn.Linear(state_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))*2  #【-1，1】*2 => [-2,2]
        return x
    
class Critic(nn.Module):
    # Q(s,a) 的输入 输出为1
    def __init__(self,state_dim,action_dim,hidden_dim=64):
        super(Critic,self).__init__()
        # 全连接层
        self.fc1 = nn.Linear(state_dim+action_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,1)

    def forward(self,x,a):
        x = torch.cat([x,a],1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x)) #【-1，1】*2 => [-2,2]
        return self.fc3(x)


# Buffer
class ReplayMemory:
    def __init__(self,capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add_memo(self, state, action,reward, next_state,done):
        # 多维的值
        # (3, ) -> (1,3)
        state = np.expand_dims(state,0)
        next_state = np.expand_dims(next_state,0)
        self.buffer.append((state,action,reward,next_state,done))
        # 在队尾添加

    def sample(self,batch_size):
        # 将这个列表解包，再将解包的元素以tutle的形式分别赋值
        # Sample a random minibatch of N transitions (si; ai; ri; si+1) from R
        state, action,reward, next_state,done = zip(*random.sample(self.buffer,batch_size))
        return np.concatenate(state),action,reward,np.concatenate(next_state),done
    
    def __len__(self):
        # 在大于minibatch时才进行sample
        return len(self.buffer)
    

class DDPGAgent:
    def __init__(self,state_dim,action_dim):
        self.actor = Actor(state_dim,action_dim).to(device)
        self.actor_target = Actor(state_dim,action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())# 将参数调取过来
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = LR_ACTOR)

        self.critic = Critic(state_dim,action_dim).to(device)
        self.critic_target = Critic(state_dim,action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())# 将参数调取过来
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr= LR_CRITIC)

        self.reply_buffer = ReplayMemory(MEMORY_SIZE)
    
    def get_action(self,state):
        # (3,)->(1,3)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]
    
    def update(self):
        # sample之后的 TD-learning 
        # Update critic by minimizing the loss: 最小二乘
        # Update the actor policy using the sampled policy gradient: 采样梯度
        # Update the target networks:
        if len(self.reply_buffer) < BATCH_SIZE:
            return 
        
        states, actions,rewards, next_states,dones = self.reply_buffer.sample(BATCH_SIZE)
        states  = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(np.vstack(actions)).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device) # 需要升维
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device) # 需要升维

        # update critic 
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states,next_actions.detach())# 不进行梯度更新
        target_Q = rewards + (GAMMA * target_Q * (1-dones)) # 求yi

        current_Q = self.critic(states,actions)
        critic_loss = nn.MSELoss()(current_Q,target_Q)
        self.critic_optimizer.zero_grad() # 上一步梯度清零 然后再计算这一步梯度
        critic_loss.backward() # 计算loss的梯度
        self.critic_optimizer.step() # 更新参数


        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks of critic and actor
        for target_param,param, in zip(self.actor_target.parameters(),self.actor.parameters()):
            target_param.data.copy_(TAU*param.data+ (1-TAU)*target_param.data)
        
        for target_param,param, in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1-TAU)*target_param.data)

        


