import os
import gym
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
parser.add_argument('--lamda', type=float, default=0.98)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--actor_lr', type=float, default=1e-3)
parser.add_argument('--critic_lr', type=float, default=1e-3)
parser.add_argument('--model_update_num', type=int, default=10)
parser.add_argument('--clip_param', type=float, default=0.2)
parser.add_argument('--max_iter_num', type=int, default=500)
parser.add_argument('--total_sample_size', type=int, default=2048)
parser.add_argument('--log_interval', type=int, default=5)
parser.add_argument('--goal_score', type=int, default=-300)
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

def train_model(actor, critic, actor_optimizer, critic_optimizer, 
                trajectories, state_size, action_size):
    trajectories = np.array(trajectories)
    states = np.vstack(trajectories[:, 0])
    actions = list(trajectories[:, 1])
    rewards = list(trajectories[:, 2])
    masks = list(trajectories[:, 3])

    actions = torch.Tensor(actions).squeeze(1)
    rewards = torch.Tensor(rewards).squeeze(1)
    masks = torch.Tensor(masks)

    old_values = critic(torch.Tensor(states))
    returns, advantages = get_gae(rewards, masks, old_values, args)

    mu, std = actor(torch.Tensor(states))
    old_policy = get_log_prob(actions, mu, std)
    
    criterion = torch.nn.MSELoss()

    n = len(states)
    arr = np.arange(n)

    for _ in range(args.model_update_num):
        np.random.shuffle(arr)

        for i in range(n // args.batch_size):
            mini_batch_index = arr[args.batch_size * i : args.batch_size * (i + 1)]
            mini_batch_index = torch.LongTensor(mini_batch_index)
            
            states_samples = torch.Tensor(states)[mini_batch_index]
            actions_samples = torch.Tensor(actions)[mini_batch_index]
            returns_samples = returns.unsqueeze(1)[mini_batch_index]
            advantages_samples = advantages.unsqueeze(1)[mini_batch_index]
            old_values_samples = old_values[mini_batch_index].detach()
            
            # get critic loss
            values_samples = critic(states_samples)
            clipped_values_samples = old_values_samples + \
                                    torch.clamp(values_samples - old_values_samples,
                                                -args.clip_param, 
                                                args.clip_param)
            
            critic_loss = criterion(values_samples, returns_samples)
            clipped_critic_loss = criterion(clipped_values_samples, returns_samples)
            
            critic_loss = torch.max(critic_loss, clipped_critic_loss)

            # get actor loss
            actor_loss, ratio = surrogate_loss(actor, advantages_samples, states_samples,
                                                old_policy.detach(), actions_samples,
                                                mini_batch_index)

            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_actor_loss = clipped_ratio * advantages_samples
            
            actor_loss = -torch.min(actor_loss, clipped_actor_loss).mean()

            # update actor & critic 
            loss = actor_loss + 0.5 * critic_loss

            critic_optimizer.zero_grad()
            loss.backward(retain_graph=True) 
            critic_optimizer.step()

            actor_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()
            

def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    print('state size:', state_size)
    print('action size:', action_size)
    
    actor = Actor(state_size, action_size, args)
    critic = Critic(state_size, args)
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=args.critic_lr)

    writer = SummaryWriter(args.logdir)

    recent_rewards = deque(maxlen=100)
    episodes = 0

    for iter in range(args.max_iter_num):
        trajectories = deque()
        steps = 0

        while steps < args.total_sample_size: 
            done = False
            score = 0
            episodes += 1

            state = env.reset()
            state = np.reshape(state, [1, state_size])

            while not done:
                if args.render:
                    env.render()

                steps += 1

                mu, std = actor(torch.Tensor(state))
                action = get_action(mu, std)

                next_state, reward, done, _ = env.step(action)
                
                mask = 0 if done else 1

                trajectories.append((state, action, reward, mask))

                next_state = np.reshape(next_state, [1, state_size])
                state = next_state
                score += reward

                if done:
                    recent_rewards.append(score)

        actor.train(), critic.train()
        train_model(actor, critic, actor_optimizer, critic_optimizer, 
                    trajectories, state_size, action_size)
        
        writer.add_scalar('log/score', float(score), episodes)
        
        if iter % args.log_interval == 0:
            print('{} iter | {} episode | score_avg: {:.2f}'.format(iter, episodes, np.mean(recent_rewards)))

        if np.mean(recent_rewards) > args.goal_score:
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            
            ckpt_path = args.save_path + 'model.pth'
            torch.save(actor.state_dict(), ckpt_path)
            print('Recent rewards exceed -300. So end')
            break  

if __name__ == '__main__':
    main()