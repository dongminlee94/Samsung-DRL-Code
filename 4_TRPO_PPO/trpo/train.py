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
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--critic_lr', type=float, default=1e-3)
parser.add_argument('--max_kl', type=float, default=1e-2)
parser.add_argument('--max_iter_num', type=int, default=500)
parser.add_argument('--total_sample_size', type=int, default=2048)
parser.add_argument('--log_interval', type=int, default=5)
parser.add_argument('--goal_score', type=int, default=-300)
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

def train_model(actor, critic, critic_optimizer, 
                trajectories, state_size, action_size):
    trajectories = np.array(trajectories)
    states = np.vstack(trajectories[:, 0])
    actions = list(trajectories[:, 1])
    rewards = list(trajectories[:, 2])
    masks = list(trajectories[:, 3])

    actions = torch.Tensor(actions).squeeze(1)
    rewards = torch.Tensor(rewards).squeeze(1)
    masks = torch.Tensor(masks)

    # ----------------------------
    # step 1: get returns
    returns = get_returns(rewards, masks, args.gamma)

    # ----------------------------
    # step 2: update ciritic
    criterion = torch.nn.MSELoss()

    values = critic(torch.Tensor(states))
    targets = returns.unsqueeze(1)

    critic_loss = criterion(values, targets)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # ----------------------------
    # step 3: get gradient of actor loss through surrogate loss
    mu, std = actor(torch.Tensor(states))
    old_policy = get_log_prob(actions, mu, std)
    actor_loss = surrogate_loss(actor, values, targets, states, old_policy.detach(), actions)
    
    actor_loss_grad = torch.autograd.grad(actor_loss, actor.parameters())
    actor_loss_grad = flat_grad(actor_loss_grad)
    
    # ----------------------------
    # step 4: get search direction through conjugate gradient method
    search_dir = conjugate_gradient(actor, states, actor_loss_grad.data, nsteps=10)
    
    # ----------------------------
    # step 5: get step size and maximal step
    gHg = (hessian_vector_product(actor, states, search_dir) * search_dir).sum(0, keepdim=True)
    step_size = torch.sqrt(2 * args.max_kl / gHg)[0]
    maximal_step = step_size * search_dir

    # ----------------------------    
    # step 6: perform backtracking line search and update actor in trust region
    params = flat_params(actor)
    
    old_actor = Actor(state_size, action_size, args)
    update_model(old_actor, params)
    
    backtracking_line_search(old_actor, actor, actor_loss, actor_loss_grad, 
                             old_policy, params, maximal_step, args.max_kl,
                             values, targets, states, actions)


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
        train_model(actor, critic, critic_optimizer, 
                    trajectories, state_size, action_size)

        writer.add_scalar('log/score', float(score), episodes)
        
        if iter % args.log_interval == 0:
            print('{} iter | {} episode | score_avg: {:.2f}'.format(iter, episodes, np.mean(recent_rewards)))    

        if np.mean(recent_rewards) > args.goal_score:
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            
            ckpt_path = args.save_path + 'model.pth.tar'
            torch.save(actor.state_dict(), ckpt_path)
            print('Recent rewards exceed -300. So end')
            break  

if __name__ == '__main__':
    main()