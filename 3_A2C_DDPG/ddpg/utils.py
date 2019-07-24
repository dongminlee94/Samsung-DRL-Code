import torch
import numpy as np

class OUNoise:
    def __init__(self, action_size, theta, mu, sigma):
        self.action_size = action_size
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.X = np.zeros(self.action_size) 

    def sample(self):
        dx = self.theta * (self.mu - self.X) + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        
        return self.X

def get_action(policy, ou_noise, env): 
    action = policy.detach().numpy() + ou_noise.sample() 
    action = np.clip(action, env.action_space.low, env.action_space.high)

    return action

def hard_target_update(actor, critic, target_actor, target_critic):
    target_critic.load_state_dict(critic.state_dict())
    target_actor.load_state_dict(actor.state_dict())

def soft_target_update(actor, critic, target_actor, target_critic, tau):
    soft_update(critic, target_critic, tau)
    soft_update(actor, target_actor, tau)

def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)