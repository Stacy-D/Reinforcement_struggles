#!/usr/bin/env python3
# encoding utf-8
import  torch.multiprocessing as mp
from Worker import train
from Networks import ValueNetwork
from SharedAdam import SharedAdam
import argparse
import torch
import os
import time
parser = argparse.ArgumentParser(description='One step Q-learning')
parser.add_argument('--num-processes', default=8, type=int, help='Number of processes to train the agents in')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-episodes', type=int, default=8000, metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--init-epsilon', type=float, default=0.5, metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--min_epsilon', type=float, default=1e-1, metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--tgt-net-update-freq', type=int, default=400, metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--val-net-update-freq', type=int, default=40, metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--checkpoint-time', type=int, default=int(1e6), metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--max-steps', type=int, default=int(32e6), metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--per-episode', type=int, default=500, metavar='N',
                    help='Number of episodes per agent')
parser.add_argument('--discount', type=float, default=0.99, metavar='N',
                    help='Number of episodes per agent')
if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    args = parser.parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dataloader_kwargs = {'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)
    counter = mp.Value('i', 0)
    lock = mp.Lock()
    # num features in HIGH FEATURE SET is 15 for this task
    # num actions is 4
    worker_network = ValueNetwork(15, 4)
    worker_network.share_memory()
    target_network = ValueNetwork(15, 4)
    target_network.load_state_dict(worker_network.state_dict())
    target_network.share_memory()

    optimizer = SharedAdam(worker_network.parameters(), lr=args.lr)
    optimizer.share_memory()

    processes = []
    for idx in range(0, args.num_processes):
        trainingArgs = (idx, args, worker_network,  target_network, optimizer, lock, counter)
        p = mp.Process(target=train, args=(trainingArgs))
        #train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
