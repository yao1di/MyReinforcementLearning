import gym
import numpy as np
import random
import torch
from agent import Agent

env = gym.make("CartPole-v1",render_mode = "human")
s,_ = env.reset()
n_state = len(s)
n_action = env.action_space.n


n_episode = 5000
n_time_step = 1000

EPLISON_START = 1.0
EPLISON_END = 0.02
EPLISON_DECAY = 100000 # 5000*1000=5000000 500w
TARGET_UPDATE_FREQUENCY = 10

REWARD_BUFFER = np.empty(shape=n_episode)

agent = Agent(n_input=n_state,n_output=n_action)
for episode_i in range(n_episode):
    episode_reward = 0
    for step_i in range(n_time_step):
        epislon = np.interp(episode_i*n_time_step+step_i,xp=[0,EPLISON_DECAY],fp=[EPLISON_START,EPLISON_END])
        random_sammple = random.random() #[0,1)

        if random_sammple <= epislon:
            a = env.action_space.sample()
        else:
            a = agent.online_net.act(s) #TODO

        # 执行action
        next_state, reward,done,truncated,info= env.step(a)

        #存经验
        agent.memo.add_memo(s,a, reward,done,next_state) # TODO
        s = next_state
        episode_reward += reward

        if done:
            s,_ = env.reset()
            REWARD_BUFFER[episode_i] = episode_reward
            # 对应于y_j的第一行
            break
        #

        if np.mean(REWARD_BUFFER[:episode_i]) >= 100:
            while True:
                a = agent.online_net.act(s)
                next_s,r,done,truncated,info = env.step(a)

                env.render()
                if done:
                    env.reset()


        batch_s,batch_a,batch_r,batch_done,batch_s_next = agent.memo.sample()
        # TD-learning
        # targets
        target_q_values = agent.target_net(batch_s_next) #通过后一个观测来获取
        max_target_q_values = target_q_values.max(dim=1,keepdim=True)[0]
        targets = batch_r + agent.GAMMA * (1-batch_done)* max_target_q_values #y_j的两行写在一行中

        #需要Q-values
        q_values = agent.online_net(batch_s) # 通过当前的s来得到的
        a_q_values= torch.gather(input=q_values,dim=1,index=batch_a)


        # loss
        loss = torch.nn.functional.smooth_l1_loss(targets,a_q_values)
        
        #gradient descent
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()


        # 每C步之后更新Q与Qtarget
    
    if episode_i % TARGET_UPDATE_FREQUENCY ==0:
        agent.target_net.load_state_dict(agent.online_net.state_dict()) #TODO

    # process
        print("Episode: {}".format(episode_i))
        print("Avg. Reward: {}".format(np.mean(REWARD_BUFFER[:episode_i])))
