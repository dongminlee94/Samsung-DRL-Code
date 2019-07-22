# Deep Reinforcement Learning, Summer 2019 (Samsung)

This repository contains codes for **Deep Reinforcement Learning (DRL) algorithms** with PyTorch (v0.4.1). It also provides lecture slides that explain codes in detail.

The agents with the DRL algorithms have been implemented and trained using classic control environments in OpenAI Gym.

- [CartPole](https://gym.openai.com/envs/CartPole-v1/)
- [Pendulum](https://gym.openai.com/envs/Pendulum-v0/)

## Table of Contents

### 00. Prerequisite

1. [Install from Anaconda to OpenAI Gym (Window Ver. & MacOS Ver.)](https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/0_Prerequisite/01_Install)
2. [Numpy](https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/0_Prerequisite/02_Numpy)

### 01. Deep Learning with PyTorch

- [Slide](https://github.com/dongminlee94/Samsung-DRL-Code/blob/master/1_DL_Pytorch/DL_PyTorch.pdf)
- [Code](https://github.com/dongminlee94/Samsung-DRL-Code/blob/master/1_DL_Pytorch/PyTorch.py)

### 02. Deep Q-Network (DQN) & Double DQN (DDQN)

- [Slide](https://github.com/dongminlee94/Samsung-DRL-Code/blob/master/2_DQN_DDQN/DDQN.pdf)
- [DQN Code](https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/2_DQN_DDQN/dqn)
- [DDQN Code](https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/2_DQN_DDQN/ddqn)

### 03. Advantage Actor-Critic (A2C) & Deep Deterministic Policy Gradient (DDPG)

1. A2C
   - [Slide](https://github.com/dongminlee94/Samsung-DRL-Code/blob/master/3_A2C_DDPG/A2C.pdf)
   - [Code](https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/3_A2C_DDPG/a2c)

2. DDPG
   - [Slide](https://github.com/dongminlee94/Samsung-DRL-Code/blob/master/3_A2C_DDPG/DDPG.pdf)
   - [Code](https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/3_A2C_DDPG/ddpg)

### 04. Trust Region Policy Optimization (TRPO) & Proximal Policy Optimization (PPO)

1. TRPO
   - [Slide](https://github.com/dongminlee94/Samsung-DRL-Code/blob/master/4_TRPO_PPO/TRPO.pdf)
   - [Code](https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/4_TRPO_PPO/trpo)

2. TRPO + GAE
   - [Slide](https://github.com/dongminlee94/Samsung-DRL-Code/blob/master/4_TRPO_PPO/GAE.pdf)
   - [Code](https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/4_TRPO_PPO/trpo_gae)

3. PPO
   - [Code](https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/4_TRPO_PPO/ppo)

4. PPO + GAE
   - [Slide](https://github.com/dongminlee94/Samsung-DRL-Code/blob/master/4_TRPO_PPO/PPO.pdf)
   - [Code](https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/4_TRPO_PPO/ppo_gae)

### 05. Soft Actor-Critic (SAC)

- [Slide](https://github.com/dongminlee94/Samsung-DRL-Code/blob/master/5_SAC/SAC.pdf)
- [Code](https://github.com/dongminlee94/Samsung-DRL-Code/tree/master/5_SAC/sac)

## Learning curve

### CartPole

<img src="img/cartpole.png" width="500"/>

### Pendulum

<img src="img/pendulum.png" width="500"/>

## Paper

- [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
- [Double DQN (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)
- [Advantage Actor-Critic (A2C)](http://incompleteideas.net/book/RLbook2018.pdf)
- [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/pdf/1602.01783.pdf)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Trust Region Policy Optimization (TRPO)](https://arxiv.org/pdf/1502.05477.pdf)
- [Generalized Advantage Estimator (GAE)](https://arxiv.org/pdf/1506.02438.pdf)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)

## Reference

- [Minimal and Clean Reinforcement Learning Examples in PyTorch](https://github.com/reinforcement-learning-kr/reinforcement-learning-pytorch)
- [Pytorch implementation for Policy Gradient algorithms (REINFORCE, NPG, TRPO, PPO)](https://github.com/reinforcement-learning-kr/pg_travel)
- [Pytorch implementation of SAC1](https://github.com/vitchyr/rlkit/tree/master/rlkit/torch/sac)
- [Pytorch implementation of SAC2](https://github.com/pranz24/pytorch-soft-actor-critic)
