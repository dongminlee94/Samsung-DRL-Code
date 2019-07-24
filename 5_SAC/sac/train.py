import os
import gym
import random
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim

from utils import *
from model import Actor, Critic
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="Pendulum-v0")
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', action="store_true", default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--actor_lr', type=float, default=1e-4)
parser.add_argument('--critic_lr', type=float, default=1e-3)
parser.add_argument('--alpha_lr', type=float, default=1e-4)
parser.add_argument('--tau', type=float, default=0.005)
parser.add_argument('--max_iter_num', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--goal_score', type=int, default=-300)
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

def train_model(actor, critic, target_critic, mini_batch, 
                actor_optimizer, critic_optimizer, alpha_optimizer,
                target_entropy, log_alpha, alpha):
    mini_batch = np.array(mini_batch)
    states = np.vstack(mini_batch[:, 0])
    actions = list(mini_batch[:, 1])
    rewards = list(mini_batch[:, 2])
    next_states = np.vstack(mini_batch[:, 3])
    masks = list(mini_batch[:, 4])

    actions = torch.Tensor(actions).squeeze(1)
    rewards = torch.Tensor(rewards).squeeze(1)
    masks = torch.Tensor(masks)

    # update critic 
    criterion = torch.nn.MSELoss()
    
    # get Q-values using two Q-functions to mitigate overestimation bias
    q_value1, q_value2 = critic(torch.Tensor(states), actions)

    # get target
    mu, std = actor(torch.Tensor(next_states))
    next_policy, next_log_policy = eval_action(mu, std)
    target_next_q_value1, target_next_q_value2 = target_critic(torch.Tensor(next_states), next_policy)
    
    min_target_next_q_value = torch.min(target_next_q_value1, target_next_q_value2)
    min_target_next_q_value = min_target_next_q_value.squeeze(1) - alpha * next_log_policy.squeeze(1)
    target = rewards + masks * args.gamma * min_target_next_q_value

    critic_loss1 = criterion(q_value1.squeeze(1), target.detach()) 
    critic_optimizer.zero_grad()
    critic_loss1.backward()
    critic_optimizer.step()

    critic_loss2 = criterion(q_value2.squeeze(1), target.detach()) 
    critic_optimizer.zero_grad()
    critic_loss2.backward()
    critic_optimizer.step()

    # update actor 
    mu, std = actor(torch.Tensor(states))
    policy, log_policy = eval_action(mu, std)
    
    q_value1, q_value2 = critic(torch.Tensor(states), policy)
    min_q_value = torch.min(q_value1, q_value2)
    
    actor_loss = ((alpha * log_policy) - min_q_value).mean() 
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    # update alpha
    alpha_loss = -(log_alpha * (log_policy + target_entropy).detach()).mean() 
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    alpha = torch.exp(log_alpha) 
    
    return alpha

    
def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    print('state size:', state_size)
    print('action size:', action_size)
    
    actor = Actor(state_size, action_size, args)
    critic = Critic(state_size, action_size, args)
    target_critic = Critic(state_size, action_size, args)
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)

    hard_target_update(critic, target_critic)
    
    # initialize automatic entropy tuning
    target_entropy = -torch.prod(torch.Tensor(action_size)).item()
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha = torch.exp(log_alpha)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr)
    
    writer = SummaryWriter(args.logdir)

    replay_buffer = deque(maxlen=100000)
    recent_rewards = deque(maxlen=100)
    steps = 0

    for episode in range(args.max_iter_num):
        done = False
        score = 0

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if args.render:
                env.render()

            steps += 1

            mu, std = actor(torch.Tensor(state))
            action = get_action(mu, std)
            
            next_state, reward, done, _ = env.step(action) 

            next_state = np.reshape(next_state, [1, state_size])
            mask = 0 if done else 1

            replay_buffer.append((state, action, reward, next_state, mask))

            state = next_state
            score += reward

            if steps > args.batch_size:
                mini_batch = random.sample(replay_buffer, args.batch_size)
                
                actor.train(), critic.train(), target_critic.train()
                alpha = train_model(actor, critic, target_critic, mini_batch, 
                                    actor_optimizer, critic_optimizer, alpha_optimizer,
                                    target_entropy, log_alpha, alpha)
                
                soft_target_update(critic, target_critic, args.tau)

            if done:
                recent_rewards.append(score)

        if episode % args.log_interval == 0:
            print('{} episode | score_avg: {:.2f}'.format(episode, np.mean(recent_rewards)))
            writer.add_scalar('log/score', float(score), episode)

        if np.mean(recent_rewards) > args.goal_score:
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)

            ckpt_path = args.save_path + 'model.pth.tar'
            torch.save(actor.state_dict(), ckpt_path)
            print('Recent rewards exceed -300. So end')
            break  

if __name__ == '__main__':
    main()