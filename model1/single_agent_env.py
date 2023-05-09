from os import system
from time import sleep
import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from GridWorldEnv import GridWorldEnv


State = int
Action = int


# NOTE: When testing locally with the `display=True` option
# use `cls` if you are on a Windows machine or `clear` if you are on an Apple machine.

class MDPAgent:
    def __init__(self, env, gamma):
        self.env = env
        self.gamma = gamma
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.policy = self.create_initial_policy()

    def create_initial_policy(self):
        '''
        A policy is a numpy array of length self.num_states where
        self.policy[state] = action

        You are welcome to modify this function to test out the performance of different policies.
        '''
        # policy is num_states array (deterministic)
        policy = np.zeros(self.num_states, dtype=int)
        return policy

    def play_game(self, display=False):
        '''
        Play through one episode of the game under the current policy
        display=True results in displaying the current policy performed on a randomly generated environment in the terminal.
        '''
        self.env.reset()
        episodes = []
        finished = False

        curr_state = self.env.s
        total_reward = 0

        while not finished:
            # display current state
            if display:
                system('cls')
                self.env.render()
                sleep(0.1)

            # find next state
            action = self.policy[curr_state]
            curr_state, reward, finished, truncated, info = self.env.step(action)
            total_reward += reward
            episodes.append([curr_state, action, reward])

        # display end result
        if display:
            system('cls')
            self.env.render()

        print(f"Total Reward from this run: {total_reward}")
        return episodes

    def compute_episode_rewards(self, num_episodes=100, step_limit=1000):
        '''
        Computes the mean, variance, and maximum of episode reward over num_episodes episodes
        '''
        total_rewards = np.zeros(num_episodes)
        for episode in range(num_episodes):
            self.env.reset()
            finished = False
            num_steps = 0
            curr_state = self.env.s
            while not finished and num_steps < step_limit:
                action = self.policy[curr_state]
                curr_state, reward, finished, truncated, info = self.env.step(action)
                total_rewards[episode] += reward
                num_steps += 1

        return np.mean(total_rewards), np.var(total_rewards), np.max(total_rewards)

    def compute_episode_states(self, num_episodes=100, step_limit=1000):
        states = [[] for _ in range(num_episodes)]
        for episode in range(num_episodes):
            self.env.reset()
            finished = False
            num_steps = 0
            curr_state = self.env.s
            states[episode].append(curr_state)
            while not finished and num_steps < step_limit:
                action = self.policy[curr_state]
                curr_state, reward, finished, truncated, info = self.env.step(action)
                num_steps += 1
                states[episode].append(curr_state)

        return states

    def print_rewards_info(self, num_episodes=100, step_limit=1000):
        '''
        Prints information from compute_episode_rewards
        '''
        mean, var, best = self.compute_episode_rewards(num_episodes=num_episodes, step_limit=step_limit)
        print(f"Mean of Episode Rewards: {mean}, Variance of Episode Rewards: {var}, Best Episode Reward: {best}")


class DynamicProgramming(MDPAgent):
    """
    Write your algorithms for Policy Iteration and Value Iteration in the appropriate functions below.
    """
    def __init__(self, env, gamma=0.95, epsilon=0.001):
        '''
        Initialize policy, environment, value table (V), policy (policy), and transition matrix (P)
        '''
        super(DynamicProgramming, self).__init__(env=env, gamma=gamma)
        self.V = np.zeros(self.num_states) # value function
        self.P = self.env.P # two dimensional array of transition probabilities based on state and action
        self.epsilon = epsilon

    def updated_action_values(self, state: State) -> np.ndarray:
        """
        This is a useful helper function for implementing value_iteration.
        Given a state (given by index), returns a numpy array

            [Q[s, a_1], Q[s, a_2], ..., Q[s, a_n]]

        based on current value function self.V.
        """
        action_values = np.zeros(self.num_actions)
        for action in range(self.num_actions):
            for prob, next_state, reward, end in self.env.P[state][action]:
                action_values[action] += prob * (reward + self.gamma * self.V[next_state])
        return action_values

    def value_iteration(self):
        """
        Perform value iteration to compute the value of every state under the optimal policy.
        This method does not return anything. After calling this method, self.V should contain the
        correct values for each state. Additionally, self.policy should be an array that contains
        the optimal policy, where policies are encoded as indicated in the `create_initial_policy` docstring.
        """
        while True:
            delta = 0
            for state in range(self.num_states):
                action_values = self.updated_action_values(state)
                best_action_value = np.max(action_values)
                delta = max(delta, np.abs(self.V[state] - best_action_value))
                self.V[state] = best_action_value

            # convergence of values
            if delta < self.epsilon * (1 - self.gamma) / self.gamma:
                break

        # find deterministic best policy
        for state in range(self.num_states):
            action_values = self.updated_action_values(state)
            best_action = np.argmax(action_values)
            self.policy[state] = best_action


class QLearning(MDPAgent):
    """
    Write you algorithm for active model-free Q-learning in the appropriate functions below.
    """

    def __init__(self, env, gamma=0.95, epsilon=0.01):
        """
        Initialize policy, environment, and Q table (Q)
        """
        super(QLearning, self).__init__(env=env, gamma=gamma)
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.state_action_counter = np.zeros((self.num_states, self.num_actions))   # keeps track of k_sa
        self.epsilon = epsilon

    def choose_action(self, state: State) -> Action:
        """
        Returns action based on Q-values using the epsilon-greedy exploration strategy
        """
        if np.random.uniform() < self.epsilon:
            return np.random.choice(np.arange(self.num_actions))
        else:
            best_actions = np.argwhere(self.Q[state] == np.amax(self.Q[state])).flatten()
            return np.random.choice(best_actions)

    def q_learning(self, num_episodes=10000, interval=1000, display=False, step_limit=10000):
        """
        Implement the tabular update for the table of Q-values, stored in self.Q
        Note that unlike value iteration, Q-learning is done online (i.e. we learn from direct experience with the
        environment instead of needing access to the transition probabilities.
        Boilerplate code of running several episodes and retrieving the (s, a, r, s') transitions has already been done
        for you.
        Just as before, make sure to update `self.policy` to match the Q-values.
        """
        mean_returns = []
        for e in range(1, num_episodes+1):
            self.env.reset()
            finished = False

            curr_state = self.env.s
            num_steps = 0

            while not finished and num_steps < step_limit:
                # display current state
                if display:
                    system('cls')
                    self.env.render()
                    sleep(1)

                action = self.choose_action(curr_state)
                self.state_action_counter[curr_state][action] += 1
                next_state, reward, finished, truncated, info = self.env.step(action)

                # update Q values
                alpha = min(0.1, 10 / self.state_action_counter[curr_state][action] ** 0.8)
                self.Q[curr_state][action] = self.Q[curr_state][action] + alpha * (
                        reward + self.gamma * np.max(self.Q[next_state]) - self.Q[curr_state][action]
                )

                num_steps += 1
                curr_state = next_state
                self.policy = np.argmax(self.Q, axis=1)

            # run tests every interval episodes
            if e % interval == 0:
                mean, var, best = self.compute_episode_rewards(num_episodes=100)
                mean_returns.append(mean)


def print_stats(arr, arr_name):
    print(arr_name)
    print(np.round(np.mean(arr), 4), np.round(np.median(arr), 4), np.round(np.std(arr), 3))

def get_stats(arr):
    return np.mean(arr, axis=0), np.median(arr, axis=0), np.std(arr, axis=0)

if __name__ == "__main__":
    num_product_ladders = 5
    num_business_ladders = 15
    num_actions_each = 11
    increasing_invention_difficulty = True

    # list_of_num_actions_each = np.arange(3, 12)
    # df_results = {"L": [], "Mean Product Innovation Rate": [], "Mean Business Process Innovation Rate": []}

    # # iterate across multiple parameters
    # for num_actions_each in list_of_num_actions_each:

    #     env = GridWorldEnv(num_product_ladders,
    #                     num_business_ladders,
    #                     num_actions_each,
    #                     increasing_invention_difficulty=increasing_invention_difficulty)
        
    #     env.reset()

    #     print("Testing Value Iteration...")
    #     my_policy = DynamicProgramming(env, gamma=0.9, epsilon=0.0001)
    #     my_policy.value_iteration()
    #     readable_policy = [env.action_id_to_action(action) for action in my_policy.policy]

    #     num_episodes = 1000
    #     states = my_policy.compute_episode_states(num_episodes=num_episodes)

    #     product_innovation_rates = []
    #     business_innovation_rates = []
    #     for episode in range(num_episodes):
    #         iterations_required = len(states[episode])
    #         product_innovation_rate = env.state_id_to_state(states[episode][-1])[0] / iterations_required
    #         business_innovation_rate = env.state_id_to_state(states[episode][-1])[1] / iterations_required
    #         product_innovation_rates.append(product_innovation_rate)
    #         business_innovation_rates.append(business_innovation_rate)

    #     product_innovation_rates = np.array(product_innovation_rates).reshape(100, -1)
    #     business_innovation_rates = np.array(business_innovation_rates).reshape(100, -1)

    #     pmeans, pmedians, pstds = get_stats(product_innovation_rates)
    #     bmeans, bmedians, bstds = get_stats(business_innovation_rates)

    #     df_results["L"] += [num_actions_each]*10
    #     df_results["Mean Product Innovation Rate"] += list(pmeans)
    #     df_results["Mean Business Process Innovation Rate"] += list(bmeans)

    # df_results = pd.DataFrame(df_results)
    # sns.lineplot(
    #     data=df_results,
    #     x="L", y="Mean Product Innovation Rate", label="Product Innovation Rate"
    # )
    # sns.lineplot(
    #     data=df_results,
    #     x="L", y="Mean Business Process Innovation Rate", label="Business Process Innovation Rate"
    # )
    # plt.ylabel("Innovation Rate")
    # plt.title("Dependence of Innovation Rates on L")
    # plt.legend()
    # plt.savefig("L_Dependence")
    # # plt.show()
    # plt.close()

    # dependence on zeta_p

    zeta_ps = np.arange(0.01, 0.101, 0.01)
    df_results = {"Zeta_p": [], "Mean Product Innovation Rate": [], "Mean Business Process Innovation Rate": []}

    # iterate across multiple parameters
    for zeta_p in zeta_ps:
        env = GridWorldEnv(num_product_ladders,
                        num_business_ladders,
                        num_actions_each,
                        zeta_p=zeta_p,
                        increasing_invention_difficulty=increasing_invention_difficulty)
        
        env.reset()

        print("Testing Value Iteration...")
        my_policy = DynamicProgramming(env, gamma=0.9, epsilon=0.0001)
        my_policy.value_iteration()
        readable_policy = [env.action_id_to_action(action) for action in my_policy.policy]

        num_episodes = 1000
        states = my_policy.compute_episode_states(num_episodes=num_episodes)

        product_innovation_rates = []
        business_innovation_rates = []
        for episode in range(num_episodes):
            iterations_required = len(states[episode])
            product_innovation_rate = env.state_id_to_state(states[episode][-1])[0] / iterations_required
            business_innovation_rate = env.state_id_to_state(states[episode][-1])[1] / iterations_required
            product_innovation_rates.append(product_innovation_rate)
            business_innovation_rates.append(business_innovation_rate)

        product_innovation_rates = np.array(product_innovation_rates).reshape(100, -1)
        business_innovation_rates = np.array(business_innovation_rates).reshape(100, -1)

        pmeans, pmedians, pstds = get_stats(product_innovation_rates)
        bmeans, bmedians, bstds = get_stats(business_innovation_rates)

        df_results["Zeta_p"] += [1/zeta_p]*10
        df_results["Mean Product Innovation Rate"] += list(pmeans)
        df_results["Mean Business Process Innovation Rate"] += list(bmeans)

    df_results = pd.DataFrame(df_results)
    sns.lineplot(
        data=df_results,
        x="Zeta_p", y="Mean Product Innovation Rate", label="Product Innovation Rate"
    )
    sns.lineplot(
        data=df_results,
        x="Zeta_p", y="Mean Business Process Innovation Rate", label="Business Process Innovation Rate"
    )
    plt.ylabel("Innovation Rate")
    plt.title("Dependence of Innovation Rates on Zeta_p")
    plt.legend()
    plt.savefig("Zeta_p_Dependence")
    # plt.show()
    plt.close()




    zeta_bs = np.arange(0.01, 0.211, 0.02)
    df_results = {"Zeta_b": [], "Mean Product Innovation Rate": [], "Mean Business Process Innovation Rate": []}

    # iterate across multiple parameters
    for zeta_b in zeta_bs:
        env = GridWorldEnv(num_product_ladders,
                        num_business_ladders,
                        num_actions_each,
                        zeta_b=zeta_b,
                        increasing_invention_difficulty=increasing_invention_difficulty)
        
        env.reset()

        print("Testing Value Iteration...")
        my_policy = DynamicProgramming(env, gamma=0.9, epsilon=0.0001)
        my_policy.value_iteration()
        readable_policy = [env.action_id_to_action(action) for action in my_policy.policy]

        num_episodes = 1000
        states = my_policy.compute_episode_states(num_episodes=num_episodes)

        product_innovation_rates = []
        business_innovation_rates = []
        for episode in range(num_episodes):
            iterations_required = len(states[episode])
            product_innovation_rate = env.state_id_to_state(states[episode][-1])[0] / iterations_required
            business_innovation_rate = env.state_id_to_state(states[episode][-1])[1] / iterations_required
            product_innovation_rates.append(product_innovation_rate)
            business_innovation_rates.append(business_innovation_rate)

        product_innovation_rates = np.array(product_innovation_rates).reshape(100, -1)
        business_innovation_rates = np.array(business_innovation_rates).reshape(100, -1)

        pmeans, pmedians, pstds = get_stats(product_innovation_rates)
        bmeans, bmedians, bstds = get_stats(business_innovation_rates)

        df_results["Zeta_b"] += [1/zeta_b]*10
        df_results["Mean Product Innovation Rate"] += list(pmeans)
        df_results["Mean Business Process Innovation Rate"] += list(bmeans)

    df_results = pd.DataFrame(df_results)
    sns.lineplot(
        data=df_results,
        x="Zeta_b", y="Mean Product Innovation Rate", label="Product Innovation Rate"
    )
    sns.lineplot(
        data=df_results,
        x="Zeta_b", y="Mean Business Process Innovation Rate", label="Business Process Innovation Rate"
    )
    plt.ylabel("Innovation Rate")
    plt.title("Dependence of Innovation Rates on Zeta_b")
    plt.legend()
    plt.savefig("Zeta_b_Dependence")
    # plt.show()
    plt.close()




    # alphas = np.arange(0.1, 1.0, 0.1)
    # df_results = {"Alpha": [], "Mean Product Innovation Rate": [], "Mean Business Process Innovation Rate": []}

    # # iterate across multiple parameters
    # for alpha in alphas:
    #     env = GridWorldEnv(num_product_ladders,
    #                     num_business_ladders,
    #                     num_actions_each,
    #                     alpha=alpha,
    #                     increasing_invention_difficulty=increasing_invention_difficulty)
        
    #     env.reset()

    #     print("Testing Value Iteration...")
    #     my_policy = DynamicProgramming(env, gamma=0.9, epsilon=0.0001)
    #     my_policy.value_iteration()
    #     readable_policy = [env.action_id_to_action(action) for action in my_policy.policy]

    #     num_episodes = 1000
    #     states = my_policy.compute_episode_states(num_episodes=num_episodes)

    #     product_innovation_rates = []
    #     business_innovation_rates = []
    #     for episode in range(num_episodes):
    #         iterations_required = len(states[episode])
    #         product_innovation_rate = env.state_id_to_state(states[episode][-1])[0] / iterations_required
    #         business_innovation_rate = env.state_id_to_state(states[episode][-1])[1] / iterations_required
    #         product_innovation_rates.append(product_innovation_rate)
    #         business_innovation_rates.append(business_innovation_rate)

    #     product_innovation_rates = np.array(product_innovation_rates).reshape(100, -1)
    #     business_innovation_rates = np.array(business_innovation_rates).reshape(100, -1)

    #     pmeans, pmedians, pstds = get_stats(product_innovation_rates)
    #     bmeans, bmedians, bstds = get_stats(business_innovation_rates)

    #     df_results["Alpha"] += [alpha]*10
    #     df_results["Mean Product Innovation Rate"] += list(pmeans)
    #     df_results["Mean Business Process Innovation Rate"] += list(bmeans)

    # df_results = pd.DataFrame(df_results)
    # sns.lineplot(
    #     data=df_results,
    #     x="Alpha", y="Mean Product Innovation Rate", label="Product Innovation Rate"
    # )
    # sns.lineplot(
    #     data=df_results,
    #     x="Alpha", y="Mean Business Process Innovation Rate", label="Business Process Innovation Rate"
    # )
    # plt.ylabel("Innovation Rate")
    # plt.title("Dependence of Innovation Rates on Alpha")
    # plt.legend()
    # plt.savefig("Alpha_Dependence")
    # plt.close()