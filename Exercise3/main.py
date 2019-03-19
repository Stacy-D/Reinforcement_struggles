#!/usr/bin/env python3
# encoding utf-8
import  multiprocessing as mp
from Worker import train
from Networks import ValueNetwork
from SharedAdam import SharedAdam
import argparse
import torch
import time
parser = argparse.ArgumentParser(description='One step Q-learning')
parser.add_argument('--num-processes', default=4, type=int, help='Number of processes to train the agents in')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-episodes', type=int, default=4000, metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--init-epsilon', type=float, default=0.5, metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--min_epsilon', type=float, default=0.000001, metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--tgt-net-update-freq', type=int, default=400, metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--val-net-update-freq', type=int, default=50, metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--checkpoint-time', type=int, default=100000, metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', metavar='N',
                    help='Number of episodes per agent')
# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.
"""
This script should contain the necessary processes to run your asynchronous agent.
 We've provided an code snippet on how to asynchronously call multiple instances of the training function train()
  in Worker.py in the __main__ function.
"""
if __name__ == "__main__":

    # Example on how to initialize global locks for processes
    # and counters.

    # TODO should create models here   model.share_memory() # gradients are allocated lazily, so they are not shared here
    args = parser.parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)
    #TODO figure out what this thing does
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    value_network = ValueNetwork(15,4)
    target_network = ValueNetwork(15, 4)
    optimizer = SharedAdam(value_network.parameters(), lr=args.lr)
    processes = []
    for idx in range(0, args.num_processes):
        trainingArgs = (idx, args, value_network, target_network, optimizer, lock, counter)
        p = mp.Process(target=train, args=(trainingArgs))
        #train the model across `num_processes` processes
        print('start {}'.format(idx))
        p.start()
        print('end {}'.format(idx))

        processes.append(p)
    for p in processes:
        p.join()
