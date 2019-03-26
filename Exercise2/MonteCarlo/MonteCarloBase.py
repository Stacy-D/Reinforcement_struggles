#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
from collections import defaultdict


class MonteCarloAgent(Agent):
    def __init__(self, discountFactor, epsilon, initVals=0.0):
        super(MonteCarloAgent, self).__init__()
        self.policy = self.get_epsilon_policy()
        self.discount = discountFactor
        self.epsilon = epsilon
        self.min_ep = 0.15
        self.init_ep = epsilon
        # init for sum and count
        self.returns = defaultdict(lambda: defaultdict(float))
        self.current_episode = []
        self.cur_state = None
        self.status = None
        self.q = defaultdict(lambda: initVals)


    def learn(self):

        # complete Q-value table of all states
        # Q-value estimate after update of the states you've encountered in the episode ordered by their first time appearance in the episode.
        lookup = set()
        filtered_pairs = [((x[0], x[1]), index) for index, x in enumerate(self.current_episode)
                          if (x[0], x[1]) not in lookup and lookup.add((x[0], x[1])) is None]
        update = []
        for pair, finx in filtered_pairs:
            G = sum([x[2] * (self.discount ** i) for i, x in enumerate(self.current_episode[finx:])])
            self.returns[pair]['G'] += G
            self.returns[pair]['cnt'] += 1
            upd = self.returns[pair]['G'] / self.returns[pair]['cnt']
            self.q[pair] = upd
            update.append(upd)
        return self.q, update

    def toStateRepresentation(self, state):
        return str(state)

    def setExperience(self, state, action, reward, status, nextState):
        self.current_episode.append((state, action, reward))
        self.cur_state = nextState
        self.status = status

    def setState(self, state):
        self.cur_state = state

    def reset(self):
        self.current_episode = []
        self.cur_state = None

    def act(self):
        act_probs = self.policy(self.cur_state)
        return np.random.choice(self.possibleActions, p=act_probs)

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        eps = max(self.min_ep, self.init_ep * 0.95 ** (numTakenActions / 1000))
        return eps

    def get_best_action(self, state):
        val = float('-inf')
        index = 0
        for idx, a in enumerate(self.possibleActions):
            appr = self.q[(state, a)]
            val, index = (appr, idx) if appr > val else (val, index)
        return index

    def get_epsilon_policy(self):
        def get_policy(state):
            # select max action
            best_action = self.get_best_action(state)
            # init all with eps/|A| probs
            actions = np.ones(len(self.possibleActions), dtype=float) * self.epsilon / len(self.possibleActions)
            # increase best action prob
            actions[best_action] += (1.0 - epsilon)
            return actions
        return get_policy


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)
    parser.add_argument('--port', type=int, default=6000)

    args = parser.parse_args()

    # Init Connections to HFO Server
    hfoEnv = HFOAttackingPlayer(numOpponents=args.numOpponents,
                                numTeammates=args.numTeammates,
                                agentId=args.id,
                                port=args.port)
    hfoEnv.connectToServer()

    # Initialize a Monte-Carlo Agent
    agent = MonteCarloAgent(discountFactor=0.99, epsilon=1.0)
    numEpisodes = args.numEpisodes
    numTakenActions = 0
    goal_scored = 0
    goals = []
    # Run training Monte Carlo Method
    for episode in range(numEpisodes):
        agent.reset()
        observation = hfoEnv.reset()
        status = 0
        num_steps = 0
        while status == 0:
            epsilon = agent.computeHyperparameters(numTakenActions, episode)
            agent.setEpsilon(epsilon)
            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1
            num_steps += 1
            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status,
                                agent.toStateRepresentation(nextObservation))
            observation = nextObservation

        agent.learn()

        if status == 1:
            goal_scored += 1
            goals.append(num_steps)
        if (episode+1) % 100 == 0:
            print(epsilon)
            print('Episode {} scored {}, accuracy {}, steps to goal {}'.format(episode+1,
                                                                               goal_scored, goal_scored*100/episode+1,
                                                                               np.mean(goals)))
