import gym
import numpy as np
import random
import torch
import time
from agent_ddpg import DDPGAgent
import os
#初始化env
env = gym.make(id='Pendulum-v1') # 倒立摆连续问题
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
#用于神经网络输入输出的搭建


#下面初始化 
#Randomly initialize critic network and actor network with weights and .
#Initialize target network 
#Initialize replay buffer R

agent = DDPGAgent(STATE_DIM,ACTION_DIM) 

# Hyperparameters
NUM_EPISODE = 100 # 多少局
NUM_STEP = 200 #每局多少步
EPSILON_START =1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000 # 一共两万步，前一万探索，后一万步不探索

REWARD_BUFFER = np.empty(shape=NUM_EPISODE)
# REWARD_BUFFER
for episodo_i in range(NUM_EPISODE):
    #Initialize a random process N for action exploration
    # eplision-gradient方式
    state,info =env.reset() #查看源码找返回的内容
    episode_reward = 0
    #Receive initial observation state s1
    for step_i in range(NUM_STEP):
        # 过去每局多少步+这一局多少步，xp是衰减到哪里了
        #Initialize a random process N for action exploration
        epsilon = np.interp(x=episodo_i*NUM_STEP+step_i, xp =[0,EPSILON_DECAY],fp=[EPSILON_START,EPSILON_END])
        #left : optional float or complex corresponding to fp
        #Value to return for x < xp[0], default is fp[0]. 这里不涉及
        #right : optional float or complex corresponding to fp
        #Value to return for x > xp[-1], default is fp[-1]. 如果x>xp[-1]的话，就取fp的最后的值

        random_sample = random.random()
        # Select action  according to the current policy and exploration noise 确定性的策略
        if random_sample<=epsilon:
            action = np.random.uniform(low=-2,high=2,size=ACTION_DIM)# -2,2是env的设置，可以查看doc 
        else:
            action = agent.get_action(state)#TODO

        #Execute action at and observe reward rt and observe new state st+1
        next_state,reward,done,truncated,_=env.step(action)

        # Store transition (st; at; rt; st+1) in R
        agent.reply_buffer.add_memo(state,action,reward,next_state,done)#TODO

        state = next_state

        episode_reward += reward

        # 取经验
        # TD-learining
        agent.update() #TODO

        if done:
            break

    # 每局
    REWARD_BUFFER[episodo_i] = episode_reward
    print(f"Episode: {episodo_i+1}, Reward: {round(episode_reward,2)}") #精度

## save
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
timestamp = time.strftime("%Y%m%d%H%M%S")
if not os.path.exists(model):
    os.makedirs(model)
torch.save(agent.actor.state_dict(),model+f"ddpg_actor_{timestamp}.pth")
torch.save(agent.critic.state_dict(),model+f"ddpg_critic_{timestamp}.pth")


env.close()

