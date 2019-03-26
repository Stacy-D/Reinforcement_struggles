#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
from collections import defaultdict
import random


class SARSAAgent(Agent):
    def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
        super(SARSAAgent, self).__init__()
        self.learning_rate = learningRate
        self.discount = discountFactor
        self.init_lr = learningRate
        self.init_ep = epsilon
        self.min_ep = 0.15
        self.setEpsilon(epsilon)
        self.q = defaultdict(lambda: initVals)
        self.prev_act = None
        self.action = None
        self.prev_state = None
        self.cur_state = None
        self.status = None
        self.error = 0
        self.reward = 0

    def learn(self):
        # value after update subtracted by value before update
        td_update = self.learning_rate*self.update
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
        val = float('-inf')
        act = self.possibleActions[0]
        for a in self.possibleActions:
            appr = self.q[(state, a)]
            val, act = (appr, a) if appr > val else (val, act)
        return act

    def setState(self, state):
        # no output
        # provide the agents you're controlling with the current state information
        self.cur_state = state

    def setExperience(self, state, action, reward, status, nextState):
        #  prepare your agent to learn using the SARSA update
        if self.action is None:
            old_val = 0
        else:
            old_val = self.q[(state, self.action)]
        self.update = self.reward + self.discount * self.q[(nextState, action)] - old_val
        # update vars
        self.prev_state = state
        self.cur_state = nextState
        self.prev_act = self.action
        self.action = action
        self.status = status
        self.reward = reward


    def computeHyperparameters(self, numTakenActions, episodeNumber):
        # tuple indicating the learning rate and epsilon used at a certain timestep
        k = 0.01
        lr = self.init_lr * 0.99**(numTakenActions/1000)
        eps = max(self.min_ep, self.init_ep * 0.99**(numTakenActions/(50 * (episodeNumber + 1))))
        return lr, eps

    def toStateRepresentation(self, state):
        # change  the representation
        return state[0]

    def reset(self):
        # reset some states of an agent at the beginning of each episode.
        self.prev_act = None
        self.action = None
        self.prev_state = None
        self.cur_state = None
        self.status = None
        self.update = 0
        self.reward = 0

    def setLearningRate(self, learning_rate):
        #  set the learning rate
        self.learning_rate = learning_rate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)
    parser.add_argument('--port', type=int, default=6000)

    args = parser.parse_args()

    numEpisodes = args.numEpisodes
    # Initialize connection to the HFO environment using HFOAttackingPlayer
    hfoEnv = HFOAttackingPlayer(numOpponents=args.numOpponents, numTeammates=args.numTeammates, agentId=args.id, port=args.port)
    hfoEnv.connectToServer()

    # Initialize a SARSA Agent
    agent = SARSAAgent(0.2, 0.95, 0.8)

    # Run training using SARSA
    numTakenActions = 0
    goal_scored = 0
    goals = []
    for episode in range(numEpisodes):
        agent.reset()
        status = 0

        observation = hfoEnv.reset()
        nextObservation = None
        epsStart = True
        num_steps = 0
        while status == 0:
            learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            agent.setLearningRate(learningRate)

            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1
            num_steps += 1
            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                                agent.toStateRepresentation(nextObservation))

            if not epsStart:
                agent.learn()
            else:
                epsStart = False

            observation = nextObservation

        agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
        agent.learn()

        if status == 1:
            goal_scored += 1
            goals.append(num_steps)
        if (episode+1) % 100 == 0:
            print((learningRate, epsilon))
            print('Episode {} scored {}, accuracy {}, steps to goal {}'.format(episode+1,
                                                                               goal_scored, goal_scored*100/episode+1,
                                                                               np.mean(goals)))
