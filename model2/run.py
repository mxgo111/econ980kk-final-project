import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from minimax_q_learner import MiniMaxQLearner 
from policy import EpsGreedyQPolicy
from QualityLaddersGame import QualityLaddersGame
from tqdm import tqdm

import os, datetime, pickle

date_time_str = "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
experiment_folder = date_time_str
experiment_folder = "enforce_diff_by_one_statetype_diff"

if not os.path.exists(experiment_folder):
    os.makedirs(experiment_folder)

games_save_file = os.path.join(experiment_folder, "games.pkl")
agents1_save_file = os.path.join(experiment_folder, "agents1.pkl")
agents2_save_file = os.path.join(experiment_folder, "agents2.pkl")
results_save_file = os.path.join(experiment_folder, "results.npy")
viz_save_folder = os.path.join(experiment_folder, "viz")

TRAIN_AGENTS = False
EVALUATE_AGENTS = False
VISUALIZE = True

nb_sectors = 5
nb_episodes = 100
nb_test_episodes = 50
nb_iterations = 10
MAX_DIFF = 5
MAX_QUALITY = 5
include_business_actions = True

TYPES = ["unbounded", 
         "leader_outsider",
         "enforce_diff_by_one", 
         "only_outsiders"]

# define parameters for the game in each sector
epsilon = 0.5
decay_rate = 1
alpha = 0.2
nb_ini_reps = 10
ini_state = 0

# randomize initial state for only_outsiders
def get_ini_state_only_outsiders_statetype_pair():
    return (0, 1) if np.random.uniform() < 0.5 else (1, 0)

# str_type = "only_outsiders"
str_type = "enforce_diff_by_one"
# state_type = "pair"
state_type = "diff"

if state_type == "pair":
    assert type(ini_state) == tuple
elif state_type == "diff":
    assert type(ini_state) == int
type = str_type

if (type == "unbounded" or type == "leader_outsider") and state_type == "pair":
    all_states = [(i, j) for i in range(MAX_QUALITY) for j in range(MAX_QUALITY)]
elif (type == "unbounded" or type == "leader_outsider") and state_type == "diff":
    all_states = np.arange(-MAX_DIFF+1, MAX_DIFF)
elif type == "enforce_diff_by_one" and state_type == "pair":
    all_states = [(i, j) for i in range(MAX_QUALITY) for j in range(i-1, i+2) if (j >= 0 and j < MAX_QUALITY)]
elif type == "enforce_diff_by_one" and state_type == "diff":
    all_states = np.arange(-1, 2)
elif type == "only_outsiders" and state_type == "pair":
    all_states = [(i, j) for i in range(MAX_QUALITY) for j in range(MAX_QUALITY) if i != j]
elif type == "only_outsiders" and state_type == "diff":
    all_states = np.concatenate((np.arange(-MAX_DIFF+1, 0)), np.arange(1, MAX_DIFF))

# define games if they don't exist
if os.path.exists(games_save_file):
    with open(games_save_file, 'rb') as f:
        games = pickle.load(f)
else:
    num_actions = 5
    cost=3,
    q=1.1
    Delta=0.5
    change_in_cost_with_quality = "constant"
    # profit_type = "constant"
    profit_type = "increasing"

    games = [
        QualityLaddersGame(type=type,
                           state_type=state_type,
                        num_actions=num_actions,
                        cost=cost,
                        q=q,
                        Delta=Delta,
                        max_diff=MAX_DIFF,
                        max_quality=MAX_QUALITY,
                        change_in_cost_with_quality=change_in_cost_with_quality,
                        profit_type = "constant")
        for _ in range(nb_sectors)
    ]

    with open(games_save_file, 'wb') as f:
        pickle.dump(games, f)

# learning phase
if TRAIN_AGENTS:
    agents1 = [
        MiniMaxQLearner(aid=0, 
                        alpha=alpha, 
                        policy=EpsGreedyQPolicy(epsilon=epsilon, decay_rate=decay_rate), 
                        ini_state=ini_state,
                        actions=np.arange(num_actions))
        for _ in range(nb_sectors)
    ]

    agents2 = [
        MiniMaxQLearner(aid=1, 
                        alpha=alpha, 
                        policy=EpsGreedyQPolicy(epsilon=epsilon, decay_rate=decay_rate), 
                        ini_state=ini_state,
                        actions=np.arange(num_actions))
        for _ in range(nb_sectors)
    ]
    for sector in tqdm(range(nb_sectors), desc=" (training) sector", position=0):
        agent1 = agents1[sector]
        agent2 = agents2[sector]
        game = games[sector]

        for ini_reps in range(nb_ini_reps):
            for ini_state in all_states:
                state = ini_state
                if state_type == "pair":
                    state_history = np.zeros((nb_iterations, 2))
                elif state_type == "diff":
                    state_history = np.zeros(nb_iterations)
                for iteration in range(nb_iterations):
                    state_history[iteration] = state
                    
                    action1 = agent1.act()
                    action2 = agent2.act()

                    next_state, r1, r2 = game.step(state, action1, action2)

                    agent1.observe(reward=r1, state=state, next_state=next_state, opponent_action=agent2.previous_action)
                    agent2.observe(reward=r2, state=state, next_state=next_state, opponent_action=agent1.previous_action)

                    state = next_state

                    if game.isGoal(state):
                        break

        for episode in tqdm(range(nb_episodes), desc=" episode", position=1, leave=False):
            if state_type == "pair" and type == "only_outsiders":
                state = get_ini_state_only_outsiders_statetype_pair()
            else:
                assert False
            if state_type == "pair":
                state_history = np.zeros((nb_iterations, 2))
            elif state_type == "diff":
                state_history = np.zeros(nb_iterations)
            for iteration in range(nb_iterations):
                state_history[iteration] = state
                
                action1 = agent1.act()
                action2 = agent2.act()

                next_state, r1, r2 = game.step(state, action1, action2)

                agent1.observe(reward=r1, state=state, next_state=next_state, opponent_action=agent2.previous_action)
                agent2.observe(reward=r2, state=state, next_state=next_state, opponent_action=agent1.previous_action)

                state = next_state

                if game.isGoal(state):
                    break

    # save agents
    with open(agents1_save_file, 'wb') as f:
        pickle.dump(agents1, f)
    with open(agents2_save_file, 'wb') as f:
        pickle.dump(agents2, f)

else:
    with open(agents1_save_file, 'rb') as f:
        agents1 = pickle.load(f)
    with open(agents2_save_file, 'rb') as f:
        agents2 = pickle.load(f)


# evaluation phase
if EVALUATE_AGENTS:
    action_history_all = np.zeros((nb_sectors, nb_test_episodes, nb_iterations, 2))
    if state_type == "pair":
        start_state_history_all = np.zeros((nb_sectors, nb_test_episodes, nb_iterations, 2))
        end_state_history_all = np.zeros((nb_sectors, nb_test_episodes, nb_iterations, 2))
    elif state_type == "diff":
        start_state_history_all = np.zeros((nb_sectors, nb_test_episodes, nb_iterations))
        end_state_history_all = np.zeros((nb_sectors, nb_test_episodes, nb_iterations))

    for sector in tqdm(range(nb_sectors), desc=" (evaluation) sector", position=0):
        agent1 = agents1[sector]
        agent2 = agents2[sector]

        action_history_per_sector = np.zeros((nb_test_episodes, nb_iterations, 2))
        if state_type == "pair":
            start_state_history_per_sector = np.zeros((nb_test_episodes, nb_iterations, 2))
            end_state_history_per_sector = np.zeros((nb_test_episodes, nb_iterations, 2))
        elif state_type == "diff":
            start_state_history_per_sector = np.zeros((nb_test_episodes, nb_iterations))
            end_state_history_per_sector = np.zeros((nb_test_episodes, nb_iterations))

        for episode in tqdm(range(nb_test_episodes), desc=" episode", position=1, leave=False):
            if state_type == "pair" and type == "only_outsiders":
                state = get_ini_state_only_outsiders_statetype_pair()
            else:
                assert False
            action_history = np.zeros((nb_iterations, 2))
            if state_type == "pair":
                start_state_history = np.zeros((nb_iterations, 2))
                end_state_history = np.zeros((nb_iterations, 2))
            elif state_type == "diff":
                start_state_history = np.zeros(nb_iterations)
                end_state_history = np.zeros(nb_iterations)
            
            for iteration in range(nb_iterations):
                start_state_history[iteration] = state

                action1 = agent1.act(training=False)
                action2 = agent2.act(training=False)

                next_state, r1, r2 = game.step(state, action1, action2)

                agent1.observe(reward=r1, state=state, next_state=next_state, opponent_action=agent2.previous_action)
                agent2.observe(reward=r2, state=state, next_state=next_state, opponent_action=agent1.previous_action)

                action_history[iteration][0], action_history[iteration][1] = action1, action2
                end_state_history[iteration] = next_state

                state = next_state

            start_state_history_per_sector[episode] = start_state_history
            action_history_per_sector[episode] = action_history
            end_state_history_per_sector[episode] = end_state_history

        start_state_history_all[sector] = start_state_history_per_sector
        action_history_all[sector] = action_history_per_sector
        end_state_history_all[sector] = end_state_history_per_sector

    # save matrices
    with open(results_save_file, 'wb') as f:
        np.save(f, start_state_history_all)
        np.save(f, action_history_all)
        np.save(f, end_state_history_all)

else:
    with open(results_save_file, 'rb') as f:
        start_state_history_all = np.load(f)
        action_history_all = np.load(f)
        end_state_history_all = np.load(f)

if VISUALIZE:
    if not os.path.exists(viz_save_folder):
        os.makedirs(viz_save_folder)

    sectors = np.arange(nb_sectors)

    # (a) histogram of average frontier quality among different sectors
    plot_a_save_file = os.path.join(viz_save_folder, "average_frontier_quality_by_sector")
    
    if state_type == "pair":
        average_frontier_quality_by_sector = np.mean(np.amax(end_state_history_all, axis=-1)[:,:,-1], axis=-1)
    # elif state_type == "diff":
    #     average_frontier_quality_by_sector = np.mean(np.abs(end_state_history_all)[:,:,-1], axis=-1)
    # on second thought I don't think it makes sense to plot this for difference in sectors
        df = pd.DataFrame({"Average Frontier Quality by Sector": average_frontier_quality_by_sector})
        sns.histplot(data=df, x="Average Frontier Quality by Sector", kde=True)
        plt.title("Histogram for Average Frontier Quality by Sector")
        plt.savefig(plot_a_save_file)
        plt.close()

    # (b) aggregate all episodes over all different sectors and plot histogram of frontier quality
    # I think this only makes sense to plot when the sectors are all the same (i.e. no change in Delta, for example)
    if state_type == "pair":
        plot_b_save_file = os.path.join(viz_save_folder, "aggregated_frontier_quality_by_sector")
        aggregated_frontier_quality_by_sector = np.amax(end_state_history_all, axis=-1)[:,:,-1].flatten()
        df = pd.DataFrame({"Aggregated Frontier Quality by Sector": aggregated_frontier_quality_by_sector})
        sns.histplot(data=df, x="Aggregated Frontier Quality by Sector", kde=True)
        plt.title("Histogram for Aggregated Frontier Quality by Sector")
        plt.savefig(plot_b_save_file)
        plt.close()
    # plot the fraction of time spent in each of the difference states
    if state_type == "diff":
        plot_b_save_file = os.path.join(viz_save_folder, "proportion_time_spent_in_each_state")
        
        for sector in sectors:
            proportion_time_spent_in_each_state_by_sector = end_state_history_all[sector, :, :].flatten()
            df = pd.DataFrame({f"Proportion of Time Spent in Each State, Sector {sector}": proportion_time_spent_in_each_state_by_sector})
            sns.histplot(data=df, x=f"Proportion of Time Spent in Each State, Sector {sector}", kde=True)
            plt.title(f"Proportion of Time Spent in Each State, Sector {sector}")
            plt.savefig(plot_b_save_file)
            plt.close()

    # (c) for a specific sector (?) plot all action taken (y-axis) vs. difference in sector quality (x-axis)
    
    # sectors = [0]
    for sector in sectors:
        plot_c_save_file = os.path.join(viz_save_folder, f"sector_{sector}_action_taken_vs_difference_in_sector_quality")

        if state_type == "pair":
            differences = start_state_history_all[sector, :, :, 0] - start_state_history_all[sector, :, :, 1]
            differences = differences.flatten()
            agent1_actions = action_history_all[sector, :, :, 0].flatten()
            agent2_actions = action_history_all[sector, :, :, 1].flatten()

            total_differences = np.concatenate((differences, -differences))
            total_actions = np.concatenate((agent1_actions, agent2_actions))

            df = pd.DataFrame({"Action Taken by Agents": total_actions, "Quality Gap": total_differences})
        elif state_type == "diff":
            differences = start_state_history_all[sector, :, :]
            differences = differences.flatten()
            agent1_actions = action_history_all[sector, :, :, 0].flatten()
            agent2_actions = action_history_all[sector, :, :, 1].flatten()

            total_differences = np.concatenate((differences, -differences))
            total_actions = np.concatenate((agent1_actions, agent2_actions))

            df = pd.DataFrame({"Action Taken by Agents": total_actions, "Quality Gap": total_differences})
        
        sns.scatterplot(data=df, x="Quality Gap", y="Action Taken by Agents")
        plt.title("Action Taken by Agents vs. Quality Gap")
        plt.savefig(plot_c_save_file)
        plt.close()

    # (d) aggregate (c) across all sectors? (might be a lot of points?)
    # do later?

    # (e) plot the quality frontier over time for different sectors)
    plot_e_save_file = os.path.join(viz_save_folder, "frontier_quality_over_time_by_sector")

    if state_type == "pair":
        average_frontier_quality_over_time = np.mean(np.amax(end_state_history_all, axis=-1), axis=1)
        df = pd.DataFrame({
            f"Sector {i}": average_frontier_quality_over_time[i]
            for i in range(nb_sectors)
        })
        sns.lineplot(data=df)
        plt.title("Average Frontier Quality over Time")
        plt.savefig(plot_e_save_file)
        plt.close()

    # questions to answer include:
    # what does the frontier of knowledge look like? distribution of frontier quality
    # can we mimic a poisson distribution and/or other distribution?
    # can we endogenize the distribution of probabilities of business and product innovations?