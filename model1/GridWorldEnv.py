import gymnasium as gym
from gym import spaces
import pygame
import numpy as np

class GridWorldEnv():
    def __init__(self,
                 num_product_ladders, 
                 num_business_ladders, 
                 num_actions_each,
                 increasing_invention_difficulty,
                 zeta_p=0.02,
                 zeta_b=0.2,
                 alpha=0.2):
        self.num_states = (num_product_ladders) * (num_business_ladders)
        self.num_product_ladders = num_product_ladders
        self.num_business_ladders = num_business_ladders
        self.num_actions_each = num_actions_each
        self.action_space = []
        # cannot spend more than a total of 1
        self.possible_spending = np.linspace(0, 1, num_actions_each)
        for i in range(num_actions_each):
            for j in range(num_actions_each):
                if self.possible_spending[i] + self.possible_spending[j] < 1.01 and self.possible_spending[i] + self.possible_spending[j] > 0:
                    self.action_space.append((self.possible_spending[i], self.possible_spending[j]))

        self.num_actions = len(self.action_space)
        self.increasing_invention_difficulty = increasing_invention_difficulty
        self.product_probability_coeff = zeta_p
        self.business_probability_coeff = zeta_b
        self.fixed_cost = 1
        self.alpha = alpha

        self.s = 0

        self.P = self.set_P()

    def set_P(self):
        P = { state_id: { action_id: [] for action_id in range(self.num_actions) } for state_id in range(self.num_states) }

        for state_id in range(self.num_states):
            for action_id in range(self.num_actions):
                product_level, business_level = self.state_id_to_state(state_id)
                product_spending, business_spending = self.action_id_to_action(action_id)

                p_product = self.get_product_improvement_probability(product_spending, business_level)
                p_business = self.get_business_improvement_probability(business_spending)

                p_product = max(min(p_product, 1), 0)
                p_business = max(min(p_business, 1), 0)

                product_improvement_done = (product_level + 1 >= self.num_product_ladders - 1)
                product_same_done = False

                reward_product_same = -product_spending - business_spending - self.fixed_cost
                reward_product_improvement = reward_product_same
                reward_product_improvement += 100 if product_improvement_done else 0
                
                # product and business both have room to grow
                if product_level < self.num_product_ladders - 1 and business_level < self.num_business_ladders - 1:
                    P[state_id][action_id] = [
                        (p_product * p_business, self.state_to_state_id((product_level + 1, business_level + 1)), reward_product_improvement, product_improvement_done), # both product and business improve
                        (p_product * (1 - p_business), self.state_to_state_id((product_level + 1, business_level)), reward_product_improvement, product_improvement_done), # product improves but not business
                        (p_business * (1 - p_product), self.state_to_state_id((product_level, business_level+1)), reward_product_same, product_same_done), # business improves but not product
                        ((1 - p_product) * (1 - p_business), self.state_to_state_id((product_level, business_level)), reward_product_same, product_same_done) # neither product nor business improve
                    ]

                # business is at highest level but not product
                elif product_level < self.num_product_ladders - 1:
                    P[state_id][action_id] = [
                        (p_product, self.state_to_state_id((product_level + 1, business_level)), reward_product_improvement, product_improvement_done), # product improves but not business
                        ((1 - p_product), self.state_to_state_id((product_level, business_level)), reward_product_same, product_same_done) # neither product nor business improve
                    ]

        return P


    def state_id_to_state(self, state_id):
        return (state_id // self.num_business_ladders, state_id % self.num_business_ladders)
    
    def state_to_state_id(self, state):
        return state[0] * self.num_business_ladders + state[1]

    def action_id_to_action(self, action_id):
        return self.action_space[action_id]

    def business_factor(self, curr_business):
        return (curr_business+1)
    
    def effective_spending(self, spending):
        if self.increasing_invention_difficulty:
            return spending ** self.alpha
        return spending
    
    def get_product_improvement_probability(self, product_spending, curr_business):
        return self.product_probability_coeff * self.effective_spending(product_spending) * self.business_factor(curr_business)
    
    def get_business_improvement_probability(self, business_spending):
        return self.business_probability_coeff * self.effective_spending(business_spending)

    def step(self, action_id):
        # needs to return curr_state, reward, finished, truncated, info
        curr_product, curr_business = self.state_id_to_state(self.s)
        product_spending, business_spending = self.action_id_to_action(action_id)
        p_product = self.get_product_improvement_probability(product_spending, curr_business)
        p_product = max(min(p_product, 1), 0)
        if np.random.uniform() < p_product:
            curr_product += 1
        p_business = self.get_business_improvement_probability(business_spending)
        p_business = max(min(p_business, 1), 0)
        if np.random.uniform() < p_business and curr_business < self.num_business_ladders - 1:
            curr_business += 1
        self.s = self.state_to_state_id((curr_product, curr_business))
        finished = (curr_product >= self.num_product_ladders - 1)

        # assuming cost is purely based off of spending
        reward = -product_spending - business_spending - self.fixed_cost
        reward += 100 if finished else 0

        return self.s, reward, finished, False, None
    
    def reset(self):
        self.s = 0
