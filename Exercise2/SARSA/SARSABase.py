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
        self.setEpsilon(epsilon)
        self.q = defaultdict(lambda: initVals)
        self.prev_act = None
        self.action = None
        self.state = None
        self.prev_state = None
        self.cur_state = None
        self.status = None
        self.old_val = 0
        self.td_target = 0
        self.reward = 0

    def learn(self):
        # value after update subtracted by value before update
        td_update = self.learning_rate*(self.td_target - self.old_val)
        self.q[(self.prev_state, self.prev_act)] += td_update
        return td_update

    def act(self):
        #  return the action that should be taken by the agent at the current state
        sample = random.random()
        if sample > self.epsilon:
            val = 0
            act = self.possibleActions[0]
            for a in self.possibleActions:
                appr = self.q[(self.cur_state, a)]
                val, act = (appr, a) if appr > val else (val, act)
            return act
        else:
            return self.possibleActions[random.randint(0, 4)]

    def setState(self, state):
        # no output
        # provide the agents you're controlling with the current state information
        self.cur_state = state

    def setExperience(self, state, action, reward, status, nextState):
        #  prepare your agent to learn using the SARSA update
        if self.action is None:
            self.old_val = 0
        else:
            self.old_val = self.q[(state, self.action)]
        self.td_target = self.reward + self.discount*self.q[(nextState, action)] - self.old_val
        # update vars
        self.prev_state = state
        self.cur_state = nextState
        self.prev_act = self.action
        self.action = action
        self.reward = reward
        self.status = status


    def computeHyperparameters(self, numTakenActions, episodeNumber):
        # tuple indicating the learning rate and epsilon used at a certain timestep
        k = 0.95
        lr = self.init_lr * np.exp(-k*numTakenActions)
        eps = self.init_ep * np.exp(-0.3 * numTakenActions)
        return lr, eps

    def toStateRepresentation(self, state):
        # change  the representation
        return state[0]

    def reset(self):
        # reset some states of an agent at the beginning of each episode.
        self.prev_act = None
        self.action = None
        self.state = None
        self.prev_state = None
        self.cur_state = None
        self.status = None
        self.old_val = 0
        self.td_target = 0
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
    agent = SARSAAgent(0.1, 0.99, 1)

    # Run training using SARSA
    numTakenActions = 0
    for episode in range(numEpisodes):
        agent.reset()
        status = 0

        observation = hfoEnv.reset()
        nextObservation = None
        epsStart = True

        while status == 0:
            learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            agent.setLearningRate(learningRate)

            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1

            nextObservation, reward, done, status = hfoEnv.step(action)
            print(obsCopy, action, reward, nextObservation)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                                agent.toStateRepresentation(nextObservation))

            if not epsStart:
                agent.learn()
            else:
                epsStart = False

            observation = nextObservation

        agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
        agent.learn()
