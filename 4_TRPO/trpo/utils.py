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

    returns = (returns - returns.mean()) / returns.std()

    return returns

def get_log_prob(actions, mu, std):
    normal = Normal(mu, std)
    log_prob = normal.log_prob(actions)

    return log_prob

def surrogate_loss(actor, values, targets, states, old_policy, actions):
    mu, std = actor(torch.Tensor(states))
    new_policy = get_log_prob(actions, mu, std)

    advantages = targets - values

    surrogate_loss = torch.exp(new_policy - old_policy) * advantages
    surrogate_loss = surrogate_loss.mean()

    return surrogate_loss


# from openai baseline code
# https://github.com/openai/baselines/blob/master/baselines/common/cg.py
def conjugate_gradient(actor, states, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for i in range(nsteps): # nsteps = 10
        Ap = hessian_vector_product(actor, states, p, cg_damping=1e-1)
        alpha = rdotr / torch.dot(p, Ap)
        
        x += alpha * p
        r -= alpha * Ap
        
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        
        p = r + betta * p
        rdotr = new_rdotr
        
        if rdotr < residual_tol: # residual_tol = 0.0000000001
            break
    return x

def hessian_vector_product(actor, states, p, cg_damping=1e-1):
    p.detach() 
    kl = kl_divergence(new_actor=actor, old_actor=actor, states=states)
    kl = kl.mean()
    
    kl_grad = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
    kl_grad = flat_grad(kl_grad)

    kl_grad_p = (kl_grad * p).sum() 
    kl_hessian = torch.autograd.grad(kl_grad_p, actor.parameters())
    kl_hessian = flat_hessian(kl_hessian)

    return kl_hessian + p * cg_damping # cg_damping = 0.1

def kl_divergence(new_actor, old_actor, states):
    mu, std = new_actor(torch.Tensor(states))
    
    mu_old, std_old = old_actor(torch.Tensor(states))
    mu_old = mu_old.detach()
    std_old = std_old.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu_old, std_old / pi_new -> mu, std
    # be careful of calculating KL-divergence. It is not symmetric metric.
    kl = torch.log(std / std_old) + (std_old.pow(2) + (mu_old - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)


def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten

def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten

def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index : index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


def backtracking_line_search(old_actor, actor, actor_loss, actor_loss_grad, 
                             old_policy, params, maximal_step, max_kl,
                             values, targets, states, actions):
    backtrac_coef = 1.0
    alpha = 0.5
    beta = 0.5
    flag = False

    expected_improve = (actor_loss_grad * maximal_step).sum(0, keepdim=True)
    expected_improve = expected_improve.data.numpy()

    for i in range(10):
        new_params = params + backtrac_coef * maximal_step
        update_model(actor, new_params)
        
        new_actor_loss = surrogate_loss(actor, values, targets, states, old_policy.detach(), actions)
        new_actor_loss = new_actor_loss.data.numpy()

        loss_improve = new_actor_loss - actor_loss
        expected_improve *= backtrac_coef
        improve_condition = loss_improve / expected_improve

        kl = kl_divergence(new_actor=actor, old_actor=old_actor, states=states)
        kl = kl.mean()

        if kl < max_kl and improve_condition > alpha:
            flag = True
            break

        backtrac_coef *= beta

    if not flag:
        params = flat_params(old_actor)
        update_model(actor, params)
        print('policy update does not impove the surrogate')