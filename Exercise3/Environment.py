#!/usr/bin/env python3
# encoding utf-8

from hfo import *
from copy import copy, deepcopy
import math
import random
import os
import time

import torch

class HFOEnv(object):

    def __init__(self, config_dir='../../../bin/teams/base/config/formations-dt',
                 port=6000, server_addr='localhost', team_name='base_left', play_goalie=False,
                 numOpponents=0, numTeammates=0, seed=123):

        self.config_dir = config_dir
        self.port = port
        self.server_addr = server_addr
        self.team_name = team_name
        self.play_goalie = play_goalie

        self.curState = None
        self.possibleActions = ['MOVE', 'SHOOT', 'DRIBBLE', 'GO_TO_BALL']
        self.numOpponents = numOpponents
        self.numTeammates = numTeammates
        self.seed = seed
        self.startEnv()
        self.hfo = HFOEnvironment()

    # Method to initialize the server for HFO environment
    def startEnv(self):
        if self.numTeammates == 0:
            os.system(
                "./../../../bin/HFO  --headless --seed {} --defense-npcs=0 --defense-agents={} --offense-agents=1 --trials 16000 --untouched-time 500 --frames-per-trial 500 --port {}  --fullstate & ".format(
                    str(self.seed),
                    str(self.numOpponents), str(self.port)))
            print(self.port)
        else:
            os.system(
                "./../../../bin/HFO --seed {} --defense-agents={} --defense-npcs=0 --offense-npcs={} --offense-agents=1 --trials 16000 --untouched-time 500 --frames-per-trial 500  --headless --port {} --fullstate &".format(
                    str(self.seed), str(self.numOpponents), str(self.numTeammates), str(self.port)))
        time.sleep(10)

    # Reset the episode and returns a new initial state for the next episode
    # You might also reset important values for reward calculations
    # in this function
    def reset(self):
        processedStatus = self.preprocessState(self.hfo.getState())
        self.curState = processedStatus

        return self.curState

    # Connect the custom weaker goalkeeper to the server and
    # establish agent's connection with HFO server
    def connectToServer(self):
        time.sleep(5)
        os.system("./Goalkeeper.py --numEpisodes=16000 --port={} &".format(str(self.port)))
        time.sleep(2)
        self.hfo.connectToServer(LOW_LEVEL_FEATURE_SET, self.config_dir, self.port, self.server_addr, self.team_name,
                                 self.play_goalie)
        print('{} connected'.format(self.port))

    # This method computes the resulting status and states after an agent decides to take an action
    def act(self, actionString):

        if actionString == 'MOVE':
            self.hfo.act(MOVE)
        elif actionString == 'SHOOT':
            self.hfo.act(SHOOT)
        elif actionString == 'DRIBBLE':
            self.hfo.act(DRIBBLE)
        elif actionString == 'GO_TO_BALL':
            self.hfo.act(GO_TO_BALL)
        else:
            raise Exception('INVALID ACTION!')

        status = self.hfo.step()
        currentState = self.hfo.getState()
        processedStatus = self.preprocessState(currentState)
        self.curState = processedStatus

        return status, self.curState

    # Define the rewards you use in this function
    # You might also give extra information on the name of each rewards
    # for monitoring purposes.

    def get_reward(self, status, nextState):
        # Implement your reward calculations inside get_reward(status, nextState)
        """
        File to establish connections with HFO, define rewards for agents and preprocess state
         representations gathered from the HFO domain. Rewards and state representations can be derived
         from HFO's LOW_LEVEL_FEATURE_SET or HIGH_LEVEL_FEATURE_SET. You are allowed to choose any of them to
         base your rewards and states from. Just ensure that you've put the correct choice in line 56 of Environment.py.
        """
        if status == GOAL:
            reward = 1.0
        elif status in [CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME]:
            reward = -1.0
        else:
            reward = 0

        info = {}

        return reward, info

    # Method that serves as an interface between a script controlling the agent
    # and the environment. Method returns the nextState, reward, flag indicating
    # end of episode, and current status of the episode

    def step(self, action_params):
        status, nextState = self.act(action_params)
        done = (status != IN_GAME)
        reward, info = self.get_reward(status, nextState)
        return nextState, reward, done, status, info

    # This method enables agents to quit the game and the connection with the server
    # will be lost as a result
    def quitGame(self):
        self.hfo.act(QUIT)

    # Preprocess the state representation in this function
    # Implement your state preprocessing function inside preprocessState(state)
    def preprocessState(self, state):
        return torch.Tensor(state).unsqueeze(0)
