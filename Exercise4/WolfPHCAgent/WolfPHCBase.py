#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import numpy as np
from collections import defaultdict


class WolfPHCAgent(Agent):
    def __init__(self, learningRate, discountFactor, winDelta=0.01, loseDelta=0.1, initVals=0.0):
        super(WolfPHCAgent, self).__init__()
        self.q = defaultdict(lambda: initVals)
        # init policy with uniform
        self.policy = defaultdict(lambda: 1 / len(self.possibleActions))
        self.avg_policy = defaultdict(float)
        self.c = defaultdict(float)
        self.init_lr = learningRate
        self.init_win = winDelta
        self.init_lose = loseDelta
        self.discount = discountFactor
        self.setLearningRate(learningRate)
        self.setLoseDelta(loseDelta)
        self.setWinDelta(winDelta)
        self.sub_opt_act = None
        self.delta = 0
        self.opt_act = None

    def setExperience(self, state, action, reward, status, nextState):
        self.update = reward + self.discount * self.get_q_max(nextState) - self.q[(state, action)]
        self.cur_state = state
        self.action = action
        self.status = status


    def get_q_max(self, state):
        return np.max([self.q[(state, action)] for action in self.possibleActions])

    def learn(self):
        td_update = self.learning_rate * self.update
        self.q[(self.cur_state, self.action)] += td_update
        return td_update

    def act(self):
        probs = [self.policy[(self.cur_state, action)] for action in self.possibleActions]
        return np.random.choice(self.possibleActions, p=probs)

    def calculateAveragePolicyUpdate(self):
        self.c[self.cur_state] += 1
        for action in self.possibleActions:
            self.avg_policy[(self.cur_state, action)] += (self.policy[(self.cur_state, action)]
                                                          - self.avg_policy[(self.cur_state, action)]) \
                                                         / self.c[self.cur_state]
        return [self.avg_policy[(self.cur_state, action)] for action in self.possibleActions]

    def calculatePolicyUpdate(self):
        self.compute_optimals()
        moved_prob = 0
        factor = self.delta / len(self.sub_opt_act) if len(self.sub_opt_act) else 0
        # update of suboptimal actions
        for action in self.sub_opt_act:
            moved_prob += min(factor, self.policy[(self.cur_state, action)])
            self.policy[(self.cur_state, action)] -= min(factor, self.policy[(self.cur_state, action)])
        # update of optimal
        opt_factor = moved_prob / (len(self.possibleActions) - len(self.sub_opt_act))
        for action in self.opt_act:
            self.policy[(self.cur_state, action)] += opt_factor
        return [self.policy[(self.cur_state, action)] for action in self.possibleActions]

    def compute_optimals(self):
        self.sub_opt_act = []
        self.delta = 0
        self.opt_act = []
        policy_val = 0
        avg_policy_val = 0
        max_val = self.get_q_max(self.cur_state)
        for action in self.possibleActions:
            state_val = self.q[(self.cur_state, action)]
            if state_val != max_val:
                self.sub_opt_act.append(action)
            elif state_val == max_val:
                self.opt_act.append(action)
            policy_val += self.policy[(self.cur_state, action)] * state_val
            avg_policy_val += self.avg_policy[(self.cur_state, action)] * state_val

        self.delta = self.win_delta if policy_val >= avg_policy_val else self.lose_delta


    def toStateRepresentation(self, state):
        return str(state)

    def setState(self, state):
        self.cur_state = state

    def setLearningRate(self, lr):
        self.learning_rate = lr

    def setWinDelta(self, winDelta):
        self.win_delta = winDelta

    def setLoseDelta(self, loseDelta):
        self.lose_delta = loseDelta

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        lr = max(0.01, self.init_lr * 0.99 ** (episodeNumber / 30000))
        return self.lose_delta, self.win_delta, lr


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numAgents', type=int, default=2)
    parser.add_argument('--numEpisodes', type=int, default=50000)

    args = parser.parse_args()

    numOpponents = args.numOpponents
    numAgents = args.numAgents
    MARLEnv = DiscreteMARLEnvironment(numOpponents=numOpponents, numAgents=numAgents)

    agents = []
    for i in range(args.numAgents):
        agent = WolfPHCAgent(learningRate=0.2, discountFactor=0.99, winDelta=0.0025, loseDelta=0.01)
        agents.append(agent)

    numEpisodes = args.numEpisodes
    numTakenActions = 0
    goal_scored = 0
    goals = []
    for episode in range(numEpisodes):
        status = ["IN_GAME", "IN_GAME", "IN_GAME"]
        observation = MARLEnv.reset()
        num_steps = 0
        while status[0] == "IN_GAME":
            for agent in agents:
                loseDelta, winDelta, learningRate = agent.computeHyperparameters(numTakenActions, episode)
                agent.setLoseDelta(loseDelta)
                agent.setWinDelta(winDelta)
                agent.setLearningRate(learningRate)
            actions = []
            perAgentObs = []
            agentIdx = 0
            for agent in agents:
                obsCopy = deepcopy(observation[agentIdx])
                perAgentObs.append(obsCopy)
                agent.setState(agent.toStateRepresentation(obsCopy))
                actions.append(agent.act())
                agentIdx += 1
            nextObservation, reward, done, status = MARLEnv.step(actions)
            numTakenActions += 1
            num_steps += 1
            agentIdx = 0
            for agent in agents:
                agent.setExperience(agent.toStateRepresentation(perAgentObs[agentIdx]), actions[agentIdx],
                                    reward[agentIdx],
                                    status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
                agent.learn()
                agent.calculateAveragePolicyUpdate()
                agent.calculatePolicyUpdate()
                agentIdx += 1

            observation = nextObservation
        if status[0] == 'GOAL':
            goal_scored += 1
            goals.append(num_steps)
        if (episode+1) % 500 == 0:
            print(learningRate)
            print('Episode {} scored {}, accuracy {}, steps to goal {}'.format(episode+1,
                                                                               goal_scored, goal_scored*100/500,
                                                                               np.mean(goals)))
            goal_scored = 0
            goals = []
