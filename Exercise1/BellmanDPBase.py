from MDP import MDP


class BellmanDPSolver(object):
    def __init__(self, discountRate=1):
        self.MDP = MDP()
        self.gamma = discountRate
        self.initVs()

    def initVs(self):
        self.state_values = {pair: 0 for pair in self.MDP.S}

    def BellmanUpdate(self):
        prev_version = self.state_values.copy()
        for state in self.MDP.S:
            total_val = dict()
            for action in self.MDP.A:
                sub_total = 0
                for next_state, prob in self.MDP.probNextStates(state, action).items():
                    sub_total += prob*(self.MDP.getRewards(state, action, next_state)
                                       + self.gamma * prev_version.get(next_state))
                total_val[action] = sub_total
            self.state_values[state] = max(total_val.values())
        return self.state_values, self.compute_greedy_policy()

    def compute_greedy_policy(self):
        policy = dict()
        for state in self.MDP.S:
            q_sa = dict()
            for action in self.MDP.A:
                q_sa[action] = sum(prob*(self.MDP.getRewards(state, action, next_state) +
                                         self.gamma * self.state_values[next_state])
                                   for next_state, prob in self.MDP.probNextStates(state, action).items())
            policy[state] = [action for action in self.MDP.A if q_sa[action] == max(q_sa.values())]
        return policy


if __name__ == '__main__':
    solution = BellmanDPSolver()
    solution.initVs()
    for i in range(20000):
        values, policy = solution.BellmanUpdate()
    print("Values : ", values)
    print("Policy : ", policy)
    states = [(x, y) for x in range(5) for y in range(5)]
    print("\nState Values")
    for counter, (y, x) in enumerate(states):
        print("{:+.4f}  ".format(values[(x, y)]), end='')

        if ((counter + 1) % 5 == 0 and counter != 0):
            print("")
    print("\n State Policies")
    for counter, (y, x) in enumerate(states):
        print("{:25} ".format(', '.join(policy[(x, y)])), end='')
        if ((counter + 1) % 5 == 0 and counter != 0):
            print("")
