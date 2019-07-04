import os
import gym
import random
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim

from model import QNet
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="CartPole-v1")
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--save_path', default='./save_model/', help='')
parser.add_argument('--render', action="store_true", default=False)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--initial_exploration', type=int, default=1000)
parser.add_argument('--epsilon', type=float, default=1.0)
parser.add_argument('--epsilon_decay', type=float, default=0.00005)
parser.add_argument('--update_target', type=int, default=100)
parser.add_argument('--max_iter_num', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--goal_score', type=int, default=400)
parser.add_argument('--logdir', type=str, default='./logs',
                    help='tensorboardx logs directory')
args = parser.parse_args()

def train_model(q_net, target_q_net, optimizer, mini_batch):
    mini_batch = np.array(mini_batch)
    states = np.vstack(mini_batch[:, 0])
    actions = list(mini_batch[:, 1]) 
    rewards = list(mini_batch[:, 2]) 
    next_states = np.vstack(mini_batch[:, 3])
    masks = list(mini_batch[:, 4]) 

    actions = torch.LongTensor(actions)
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    
    criterion = torch.nn.MSELoss()

    # get Q-value
    q_values = q_net(torch.Tensor(states))
    q_value = q_values.gather(1, actions.unsqueeze(1)).view(-1)

    # get target
    target_next_q_values = target_q_net(torch.Tensor(next_states))
    target = rewards + masks * args.gamma * target_next_q_values.max(1)[0]
    
    loss = criterion(q_value, target.detach())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def get_action(q_values, action_size, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    else:
        _, action = torch.max(q_values, 1)
        return action.numpy()[0]

def update_target_model(q_net, target_q_net):
    target_q_net.load_state_dict(q_net.state_dict())


def main():
    env = gym.make(args.env_name)
    env.seed(500)
    torch.manual_seed(500)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print('state size:', state_size) 
    print('action size:', action_size)

    q_net = QNet(state_size, action_size, args)
    target_q_net = QNet(state_size, action_size, args)
    optimizer = optim.Adam(q_net.parameters(), lr=0.001)

    update_target_model(q_net, target_q_net)

    writer = SummaryWriter(args.logdir)
    
    replay_buffer = deque(maxlen=10000)
    running_score = 0
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

            q_values = q_net(torch.Tensor(state))
            action = get_action(q_values, action_size, args.epsilon)

            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done or score == 499 else -1
            mask = 0 if done else 1

            replay_buffer.append((state, action, reward, next_state, mask))

            state = next_state
            score += reward

            if steps > args.initial_exploration:
                args.epsilon -= args.epsilon_decay
                args.epsilon = max(args.epsilon, 0.1)

                mini_batch = random.sample(replay_buffer, args.batch_size)
                
                q_net.train(), target_q_net.train()
                train_model(q_net, target_q_net, optimizer, mini_batch)

                if steps % args.update_target:
                    update_target_model(q_net, target_q_net)

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score

        if episode % args.log_interval == 0:
            print('{} episode | running_score: {:.2f} | epsilon: {:.2f}'.format(
                episode, running_score, args.epsilon))
            writer.add_scalar('log/score', float(score), episode)

        if running_score > args.goal_score:
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)

            ckpt_path = args.save_path + 'model.pth'
            torch.save(q_net.state_dict(), ckpt_path)
            print('Running score exceeds 400. So end')
            break  

if __name__ == '__main__':
    main()