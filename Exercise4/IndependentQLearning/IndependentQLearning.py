#!/usr/bin/env python3
# encoding utf-8

import random
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import argparse
from collections import defaultdict
import numpy as np

class IndependentQLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(IndependentQLearningAgent, self).__init__()

        self.learning_rate = learningRate
        self.discount = discountFactor
        self.init_lr = learningRate
        self.init_ep = epsilon
        self.min_ep = 0.05
        self.setEpsilon(epsilon)
        self.q = defaultdict(lambda: initVals)
        self.action = None
        self.prev_state = None
        self.cur_state = None
        self.status = None
        self.update = 0
        self.decay_factor = (self.init_ep - self.min_ep) / (5000 * 100)

    def learn(self):
        # value after update subtracted by value before update
        td_update = self.learning_rate * self.update
        self.q[(self.prev_state, self.action)] += td_update
        return td_update

    def act(self):
        #  return the action that should be taken by the agent at the current state
        sample = random.random()
        if sample > self.epsilon:
            return self.get_best_action(self.cur_state)
        else:
            return self.possibleActions[random.randint(0, 4)]

    def get_best_action(self, state):
        val = float('-inf')
        act = self.possibleActions[0]
        for a in self.possibleActions:
            appr = self.q[(state, a)]
            val, act = (appr, a) if appr > val else (val, act)
        return act

    def toStateRepresentation(self, state):
        # change  the representation
        return str(state)

    def setState(self, state):
        self.cur_state = state

    def setExperience(self, state, action, reward, status, nextState):
        #  prepare your agent to learn
        old_val = self.q[(state, action)]
        best_action = self.get_best_action(nextState)
        self.update = reward + self.discount * self.q[(nextState, best_action)] - old_val
        # update vars
        self.prev_state = state
        self.cur_state = nextState
        self.action = action
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
        self.update = 0

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        # tuple indicating the learning rate and epsilon used at a certain timestep
        lr = max(0.02, self.init_lr * 0.93 ** (episodeNumber / 1700))
        eps = max(self.min_ep, self.init_ep * 0.9 ** (episodeNumber / 1050)) #2000
        return lr, eps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numAgents', type=int, default=2)
    parser.add_argument('--numEpisodes', type=int, default=50000)

    args = parser.parse_args()

    MARLEnv = DiscreteMARLEnvironment(numOpponents=args.numOpponents, numAgents=args.numAgents)
    agents = []
    for i in range(args.numAgents):
        agent = IndependentQLearningAgent(learningRate=0.1, discountFactor=0.99, epsilon=1) # 0.8
        agents.append(agent)

    numEpisodes = args.numEpisodes
    numTakenActions = 0
    goal_scored = 0
    goals = []
    for episode in range(numEpisodes):
        status = ["IN_GAME", "IN_GAME", "IN_GAME"]
        observation = MARLEnv.reset()
        totalReward = 0.0
        timeSteps = 0
        num_steps = 0
        while status[0] == "IN_GAME":
            for agent in agents:
                learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
                agent.setEpsilon(epsilon)
                agent.setLearningRate(learningRate)
            actions = []
            stateCopies = []
            for agentIdx in range(args.numAgents):
                obsCopy = deepcopy(observation[agentIdx])
                stateCopies.append(obsCopy)
                agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
                actions.append(agents[agentIdx].act())
            numTakenActions += 1
            num_steps += 1
            nextObservation, reward, done, status = MARLEnv.step(actions)

            for agentIdx in range(args.numAgents):
                agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx],
                                               reward[agentIdx],
                                               status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                agents[agentIdx].learn()

            observation = nextObservation

        if status[0] == 'GOAL':
            goal_scored += 1
            goals.append(num_steps)
        if (episode+1) % 500 == 0:
            print((learningRate, epsilon))
            print('Episode {} scored {}, accuracy {}, steps to goal {}'.format(episode+1,
                                                                               goal_scored, goal_scored*100/500,
                                                                               np.mean(goals)))
            goal_scored = 0
            goals = []
