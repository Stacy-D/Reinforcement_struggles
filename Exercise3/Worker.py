import torch
import torch.nn as nn
import logging
from Networks import ValueNetwork
from Environment import HFOEnv
import random
from hfo import *
import numpy as np
epsilon_list = [0.2, 0.3, 0.4, 0.5, 0.8]


def train(idx, args, learning_network, target_network, optimizer, lock, counter):
    # init port & seed for the thread based on id val
    port = 8100 + 10 * idx  # init
    seed = idx*113 + 923
    torch.manual_seed(seed)
    logger = logging.getLogger(str(port)+__name__)
    logger.info('Init')

    worker_network = ValueNetwork(15, 4)
    worker_network.load_state_dict(learning_network.state_dict())
    # change init
    # init env
    hfo_env = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
    hfo_env.connectToServer()
    logger.info('Port {} connected'.format(port))
    episode_num = 0
    eps = random.sample(epsilon_list, 1)[0]
    worker_timestep = 0
    mse_loss = nn.MSELoss()
    max_worker_steps = args.max_steps / args.num_processes
    can_continue = True
    goal = 0
    to_goal = []
    while can_continue:
        # run episode
        obs_tensor = hfo_env.reset()
        done = False
        loss = 0
        reward_ep = 0
        ep_steps = 0
        upd_steps = 0
        while not done:
            # select action based on greedy policy
            action_idx = select_action(obs_tensor, worker_network, worker_timestep, max_worker_steps, args, eps)
            action = hfo_env.possibleActions[action_idx]
            # observe next
            next_obs_tensor, reward, done, status, info = hfo_env.step(action)
            y = computeTargets(reward, next_obs_tensor, args.discount, done, target_network)
            q_next = computePrediction(obs_tensor, action_idx, worker_network)
            # put new state
            obs_tensor = next_obs_tensor
            # update episode stats
            loss += mse_loss(y, q_next)
            reward_ep += reward
            upd_steps += 1
            ep_steps += 1
            worker_timestep += 1
            if status == 1:
                goal += 1
                to_goal.append(ep_steps)
            with lock:
                counter.value += 1
            # if terminal or time to update network
            if done or worker_timestep % args.val_net_update_freq == 0:
                worker_network.zero_grad()
                optimizer.zero_grad()
                # take mean loss
                loss /= upd_steps
                loss.backward()
                sync_grad(learning_network, worker_network)
                optimizer.step()
                worker_network.load_state_dict(learning_network.state_dict())
                loss = 0
                upd_steps = 0
            # perform update of target network
            if counter.value % args.tgt_net_update_freq == 0:
                target_network.load_state_dict(learning_network.state_dict())

            if counter.value % args.checkpoint_time == 0:
                saveModelNetwork(learning_network, args.checkpoint_dir + '_{}'.format(counter.value))
        episode_num += 1

        if episode_num % 500 == 0:
            logger.info('Global - {}, Local - {}, Episode - {}, Scored {}, Time avg {}'.format(counter.value,
                                                                                               worker_timestep,
                                                                                               episode_num,
                                                                                               goal,
                                                                                               np.mean(to_goal)))
            to_goal = []
            goal = 0
        # if time is exceeded -> break the loop
        can_continue = counter.value <= args.max_steps and worker_timestep <= max_worker_steps and status !=SERVER_DOWN
    # finish the game
    hfo_env.quitGame()
    # save the network it stopped with
    saveModelNetwork(learning_network, args.checkpoint_dir + '_{}_final'.format(counter.value))
    logger.info('Finished')



def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
    """
    :param reward: float type data representing the reward achieved by the agent.
    :param nextObservation: 2D pytorch Tensor of the next states' feature representation.
    :param discountFactor:  float type data representing the discounting factor used by the agent.
    :param done: boolean which indicates the end of the episode
    :param targetNetwork: pytorch model that will be used to compute the target values for the agents.
    :return:
    """
    if done:
        q_next = torch.FloatTensor([[0]])
    else:
        q_next, _ = targetNetwork(nextObservation).data.max(1)
    q_next = discountFactor * q_next + reward
    return q_next


# Implement the target value computation for Q-Learning inside
# this function should be usable for any target network architecture that you're using

def computePrediction(state, action, valueNetwork):
    """
    :param state: 2D pytorch Tensor of the current states' feature representation.
    :param action: integer between 0 and 3 that denotes the index of the actions that are taken.
    :param valueNetwork:  pytorch model that will be used to compute the values for the agents
    :return: # tensor with one value
    """
    k = valueNetwork(state)
    k = k.gather(1, torch.LongTensor([[action]]))
    return k.squeeze()


# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
    torch.save(model.state_dict(), strDirectory)


def sync_grad(target_net, worker_net):
    for shared_param, param in zip(target_net.parameters(), worker_net.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad.clone()


def update_network(target_network, value_network):
    for target_param, param in zip(target_network.parameters(), value_network.parameters()):
        target_param.data.copy_(param.data)

def select_action(states, value_network, current_steps, final_step, args, eps):
    sample = random.random()
    epsilon = eps - float(current_steps) / final_step * (eps - args.min_epsilon)
    if sample > epsilon:
        with torch.no_grad():
            _, action = torch.Tensor([computePrediction(states, act, value_network).item()
                                      for act in range(0, 4)]).data.max(0)
            return action.item()
    else:
        return random.randint(0, 3)