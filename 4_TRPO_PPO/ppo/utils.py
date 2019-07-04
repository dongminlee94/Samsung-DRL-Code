import math
import torch
from torch.distributions import Normal

def get_action(mu, std):
    normal = Normal(mu, std)
    action = normal.sample()
    
    return action.data.numpy()

def get_returns(rewards, masks, gamma):
    returns = torch.zeros_like(rewards)
    running_returns = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + masks[t] * gamma * running_returns 
        returns[t] = running_returns

    return returns

def get_log_prob(actions, mu, std):
    normal = Normal(mu, std)
    log_prob = normal.log_prob(actions)

    return log_prob

def surrogate_loss(actor, values, targets, states, old_policy, actions, batch_index):
    mu, std = actor(torch.Tensor(states))
    new_policy = get_log_prob(actions, mu, std)
    
    old_policy = old_policy[batch_index]
    ratio = torch.exp(new_policy - old_policy)

    advantages = targets - values

    surrogate_loss = ratio * advantages

    return surrogate_loss, ratio, advantages