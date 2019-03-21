#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
from collections import defaultdict
import random


class QLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(QLearningAgent, self).__init__()
        self.learning_rate = learningRate
        self.discount = discountFactor
        self.init_lr = learningRate
        self.init_ep = epsilon
        self.min_ep = 0.0001
        self.setEpsilon(epsilon)
        self.q = defaultdict(lambda: initVals)
        self.prev_act = None
        self.action = None
        self.prev_state = None
        self.cur_state = None
        self.status = None
        self.old_val = 0
        self.td_target = 0
        self.reward = 0
        self.decay_factor = (self.init_ep - self.min_ep) / (5000 * 500)

    def learn(self):
        # value after update subtracted by value before update
        td_update = self.learning_rate * (self.td_target - self.old_val)
        self.q[(self.prev_state, self.prev_act)] += td_update
        return td_update

    def act(self):
        #  return the action that should be taken by the agent at the current state
        sample = random.random()
        if sample > self.epsilon:
            return self.get_best_action(self.cur_state)
        else:
            return self.possibleActions[random.randint(0, 4)]

    def get_best_action(self, state):
        val = 0
        act = self.possibleActions[0]
        for a in self.possibleActions:
            appr = self.q[(self.cur_state, a)]
            val, act = (appr, a) if appr > val else (val, act)
        return act

    def toStateRepresentation(self, state):
        # change  the representation
        return state[0]

    def setState(self, state):
        self.cur_state = state

    def setExperience(self, state, action, reward, status, nextState):
        #  prepare your agent to learn using the SARSA update
        self.old_val = self.q[(state, action)]
        best_action = self.get_best_action(nextState)
        self.td_target = reward + self.discount * self.q[(nextState, best_action)] - self.old_val
        # update vars
        self.prev_state = state
        self.cur_state = nextState
        self.action = action
        self.reward = reward
        self.status = status

    def setLearningRate(self, learning_rate):
        #  set the learning rate
        self.learning_rate = learning_rate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def reset(self):
        # reset some states of an agent at the beginning of each episode.
        self.action = None
        self.prev_state = None
        self.cur_state = None
        self.status = None
        self.old_val = 0
        self.td_target = 0
        self.reward = 0

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        # tuple indicating the learning rate and epsilon used at a certain timestep
        k = 0.01
        lr = self.init_lr * 0.9 ** (numTakenActions / 5000)
        eps = max(self.min_ep, self.init_ep * 0.99 ** (numTakenActions / 500))
        print((lr, eps))
        print((episodeNumber, numTakenActions))
        return lr, eps


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)
    parser.add_argument('--port', type=int, default=6000)

    args = parser.parse_args()

    # Initialize connection with the HFO server
    hfoEnv = HFOAttackingPlayer(numOpponents=args.numOpponents,
                                numTeammates=args.numTeammates,
                                agentId=args.id,
                                port=args.port)
    hfoEnv.connectToServer()

    # Initialize a Q-Learning Agent
    agent = QLearningAgent(learningRate=0.1, discountFactor=0.99, epsilon=1.0)
    numEpisodes = args.numEpisodes

    # Run training using Q-Learning
    numTakenActions = 0
    for episode in range(numEpisodes):
        status = 0
        observation = hfoEnv.reset()

        while status == 0:
            learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            agent.setLearningRate(learningRate)

            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1

            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                                agent.toStateRepresentation(nextObservation))
            update = agent.learn()

            observation = nextObservation
