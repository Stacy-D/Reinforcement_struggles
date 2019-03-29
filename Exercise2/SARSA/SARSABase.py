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
        self.setEpsilon(epsilon)
        self.q = defaultdict(lambda: initVals)
        self.prev_state = None
        self.cur_state = None
        self.next_state = None
        self.prev_reward = None
        self.reward = None
        self.prev_act = None
        self.cur_act = None
        self.state = None
        self.min_ep = 0.001
        self.min_lr = 0.01
        self.init_ep = 0.85
        self.init_lr = 0.2

    def learn(self):
        # value after update subtracted by value before update
        td_update = self.learning_rate*(self.prev_reward +
                                        self.discount*self.q[(self.cur_state, self.cur_act)] -
                                        self.q[(self.prev_state, self.prev_act)])
        self.q[(self.prev_state, self.prev_act)] += td_update
        return td_update

    def act(self):
        #  return the action that should be taken by the agent at the current state
        sample = random.random()
        if sample > self.epsilon:
            return self.get_best_action(self.state)
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
        self.state = state

    def setExperience(self, state, action, reward, status, nextState):
        #  prepare your agent to learn using the SARSA update
        self.prev_state = self.cur_state
        self.cur_state = state
        self.next_state = nextState
        self.prev_reward = self.reward
        self.reward = reward
        self.prev_act = self.cur_act
        self.cur_act = action


    def computeHyperparameters(self, numTakenActions, episodeNumber):
        # tuple indicating the learning rate and epsilon used at a certain timestep
        lr = max(self.min_lr, self.init_lr * 0.95 ** (episodeNumber / 110))
        eps = max(self.min_ep, self.init_ep * 0.85 ** (episodeNumber / 130))
        return lr, eps

    def toStateRepresentation(self, state):
        # change  the representation
        return str(state)

    def reset(self):
        # reset some states of an agent at the beginning of each episode.
        self.prev_state = None
        self.cur_state = None
        self.next_state = None
        self.prev_reward = None
        self.reward = None
        self.prev_act = None
        self.cur_act = None
        self.state = None

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
    agent = SARSAAgent(0.2, 0.95, 0.85)

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
        if (episode+1) % 500 == 0:
            print((learningRate, epsilon))
            print('Episode {} scored {}, accuracy {}, steps to goal {}'.format(episode+1,
                                                                               goal_scored, goal_scored*100/500,
                                                                               np.mean(goals)))
            goal_scored = 0
            goals = []
