import numpy as np

class QualityLaddersGame():
    def __init__(self,
                 type="unbounded",
                 state_type="pair",
                 num_actions=5,
                 cost=3,
                 q=1.1,
                 Delta=0.5,
                 max_quality=5,
                 max_diff=5,
                 change_in_cost_with_quality="constant",
                 profit_type="constant",
                 include_business_actions=False
                ):
        self.increment = q
        action_probs = np.linspace(0,1,num_actions)

        self.action_map = {
            i: action_probs[i] for i in range(num_actions)
        }
        self.num_actions = num_actions   
        self.cost = cost
        self.type = type
        self.Delta = Delta
        self.state_type = state_type
        self.max_quality = max_quality
        self.max_diff = max_diff
        self.change_in_cost_with_quality = change_in_cost_with_quality
        self.baseline_profit = 1
        self.profit_type = profit_type
        self.include_business_actions = include_business_actions
    
    def step(self, state, action1, action2):
        if self.change_in_cost_with_quality == "constant":
            p1 = action1 / self.cost
            p2 = action2 / self.cost
        elif self.change_in_cost_with_quality == "increasing":
            if self.state_type == "pair":
                p1 = action1 / (self.cost * self.increment ** state[0])
                p2 = action2 / (self.cost * self.increment ** state[1])
            elif self.state_type == "diff":
                p1 = action1 / (self.cost * self.increment ** max(0, state))
                p2 = action2 / (self.cost * self.increment ** max(0, -state))
        elif self.change_in_cost_with_quality == "decreasing":
            if self.state_type == "pair":
                p1 = action1 / (self.cost) * (self.increment ** state[0])
                p2 = action2 / (self.cost) * (self.increment ** state[1])
            elif self.state_type == "diff":
                p1 = action1 / (self.cost) * (self.increment ** max(0, state))
                p2 = action2 / (self.cost) * (self.increment ** max(0, -state))

        if self.state_type == "pair" and self.type == "only_outsiders":
            new_state = [0, 0]
            if state[1] > state[0]:
                if np.random.uniform() < p1:
                    new_state[0] = state[1] + 1
                    new_state[1] = state[1]
                else:
                    new_state[0] = state[0]
                    new_state[1] = state[1]
            elif state[0] > state[1]:
                if np.random.uniform() < p2:
                    new_state[1] = state[0] + 1
                    new_state[0] = state[0]
                else:
                    new_state[0] = state[0]
                    new_state[1] = state[1]
            else:
                assert False, "States are equal, which shouldn't happen"
        elif self.state_type == "diff" and self.type == "enforce_diff_by_one":
            new_state = 0
            agent1_innovates = (np.random.uniform() < p1)
            agent2_innovates = (np.random.uniform() < p2)
            new_state += 1 if agent1_innovates else 0
            new_state -= 1 if agent2_innovates else 0
            new_state = max(min(new_state, 1), -1)
            
        # rewards 
        if self.state_type == "pair":
            if new_state[0] > new_state[1]:
                profit = [self.increment ** new_state[0], 0]
            elif new_state[0] == new_state[1]:
                profit = [self.increment ** new_state[0] * (1 - self.Delta), self.increment ** new_state[0] * (1 - self.Delta)]
            else:
                profit = [0, self.increment ** new_state[1]]
        elif self.state_type == "diff":
            if new_state > 0:
                profit = [self.increment ** new_state, 0]
            elif new_state == 0:
                profit = [self.increment ** new_state * (1 - self.Delta), self.increment ** new_state * (1 - self.Delta)]
            else:
                profit = [0, self.increment ** (-new_state)]

        r1 = profit[0] - self.action_map[action1]
        r2 = profit[1] - self.action_map[action2]

        if self.state_type == "pair":
            return tuple(new_state), r1, r2
        elif self.state_type == "diff":
            return new_state, r1, r2
        assert False
    
    def isGoal(self, state):
        # for computational feasibility, stop game if someone is over certain level of quality
        if self.state_type == "pair":
            return state[0] >= self.max_quality or state[1] >= self.max_quality
        elif self.state_type == "diff":
            return abs(state) >= self.max_diff
        else:
            raise Exception("Invalid state type")
    