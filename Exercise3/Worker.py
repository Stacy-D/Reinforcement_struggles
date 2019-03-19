import torch
import torch.nn as nn
from Environment import HFOEnv
import random

epsilon_list = [0.1, 0.2, 0.3, 0.4, 0.5]

def train(idx, args, value_network, target_network, optimizer, lock, counter):
    port = 8000 + 40 * idx  # init
    seed = idx*100 + 123
    torch.manual_seed(seed)
    hfo_env = HFOEnv(numTeammates=0, numOpponents=1, port=port, seed=seed)
    """
    hfo_env.connectToServer()
    print('Port {} connected'.format(port))

    discount = 0.99
    episode_num = 0
    eps = random.sample(epsilon_list, 1)[0]
    t = 0
    mse = nn.MSELoss()
    while episode_num < args.num_episodes:
        obs_tensor = hfo_env.reset()
        done = False
        loss = 0
        reward_ep = 0
        while not done:
            action_idx = select_action(obs_tensor, value_network, t, 4000, args, eps)

            action = hfo_env.possibleActions[action_idx]
            next_obs_tensor, reward, done, status, info = hfo_env.step(action)
            reward_ep += 1
            y = computeTargets(reward, next_obs_tensor, discount, done, target_network)
            q_next = computePrediction(obs_tensor, action_idx, value_network)
            loss += mse(y, q_next)
            # put new state
            obs_tensor = next_obs_tensor
            # new local step
            t += 1
            # update global step
            with lock:
                counter.value += 1
            if t % args.tgt_net_update_freq == 0:
                update_network(target_network, value_network)
            if counter.value % args.checkpoint_time == 0:
                saveModelNetwork(value_network, args.checkpoint_dir+'_{}'.format(counter.value))
            # update online network
            if t % args.val_net_update_freq == 0 or done:
                value_network.zero_grad()
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()
                loss = 0
            if t % 500 == 0:  # end of episodes
                break

        episode_num += 1
        """

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


# Implement a single call for the forward computation of your Q-Network inside
# . Note that this should be agnostic to any Q-Network architecture that you're using.

# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
    torch.save(model.state_dict(), strDirectory)


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