#!/usr/bin/env python3
# encoding utf-8

import random
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import argparse
from collections import defaultdict
import numpy as np

class JointQLearningAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, numTeammates, initVals=0.0):
        super(JointQLearningAgent, self).__init__()

        self.learning_rate = learningRate
        self.discount = discountFactor
        self.team = numTeammates
        self.init_lr = learningRate
        self.init_ep = epsilon
        self.min_ep = 0.01
        self.setEpsilon(epsilon)
        self.q = defaultdict(lambda: initVals)
        self.action = None
        self.prev_state = None
        self.cur_state = None
        self.status = None
        self.update = 0
        self.team_mates = defaultdict(lambda: defaultdict(float))
        self.n_s = defaultdict(float)

    def setExperience(self, state, action, oppoActions, reward, status, nextState):
        self.opponent = oppoActions[0]
        old_val = self.q[(state, (action, self.opponent))]
        #  prepare your agent to learn
        self.update = reward + self.discount*self.get_best_action(nextState)[1] - old_val
        # update vars
        self.prev_state = state
        self.cur_state = nextState
        self.action = (action, self.opponent)
        self.status = status

    def learn(self):
        # value after update subtracted by value before update
        td_update = self.learning_rate * self.update
        self.q[(self.prev_state, self.action)] += td_update
        self.team_mates[self.prev_state][self.opponent] += 1
        self.n_s[self.prev_state] += 1
        return td_update

    def act(self):
        #  return the action that should be taken by the agent at the current state
        sample = random.random()
        if sample > self.epsilon:
            return self.get_best_action(self.cur_state)[0]
        else:
            return self.possibleActions[random.randint(0, 4)]

    def get_teammate(self, state, action):
        cnt = self.n_s[state]
        if cnt:
            return self.team_mates.get(state)[action] / cnt
        else:
            return 1 / len(self.possibleActions)

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def get_best_action(self, state):
        val = float('-inf')
        act = self.possibleActions[0]
        for a in self.possibleActions:
            appr = sum([self.get_teammate(state, action)*self.q[(state, (a, action))]
                        for action in self.possibleActions])
            val, act = (appr, a) if appr > val else (val, act)
        return act, val

    def setLearningRate(self, learningRate):
        self.learning_rate = learningRate

    def setState(self, state):
        self.cur_state = state

    def toStateRepresentation(self, rawState):
        return str(rawState)

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        # tuple indicating the learning rate and epsilon used at a certain timestep
        lr = max(0.01, self.init_lr * 0.95 ** (episodeNumber / 1700))
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
    numAgents = args.numAgents
    numEpisodes = args.numEpisodes
    for i in range(numAgents):
        agent = JointQLearningAgent(learningRate=0.1, discountFactor=0.99, epsilon=1, numTeammates=args.numAgents - 1)
        agents.append(agent)

    numEpisodes = numEpisodes
    numTakenActions = 0
    goals = []
    goal_scored = 0
    for episode in range(numEpisodes):
        status = ["IN_GAME", "IN_GAME", "IN_GAME"]
        observation = MARLEnv.reset()
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
                agents[agentIdx].setState(agents[agentIdx].toStateRepresentation(obsCopy))
                actions.append(agents[agentIdx].act())

            nextObservation, reward, done, status = MARLEnv.step(actions)
            numTakenActions += 1
            num_steps += 1
            for agentIdx in range(args.numAgents):
                oppoActions = actions.copy()
                del oppoActions[agentIdx]
                agents[agentIdx].setExperience(agents[agentIdx].toStateRepresentation(stateCopies[agentIdx]),
                                               actions[agentIdx], oppoActions,
                                               reward[agentIdx], status[agentIdx],
                                               agent.toStateRepresentation(nextObservation[agentIdx]))
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