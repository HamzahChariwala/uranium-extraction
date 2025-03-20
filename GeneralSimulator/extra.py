import numpy as np
import matplotlib.pyplot as plt
from tooling import OligopolyGame, Agent
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def general_cost(q, fixed, variable, exponent):
    cost = fixed + (variable*q) ** exponent
    return cost

def plot_best_response_stream(agent1, agent2, inverse_demand_function, q_range=(0, 100), resolution=20):

    q1_vals = np.linspace(q_range[0], q_range[1], resolution)
    q2_vals = np.linspace(q_range[0], q_range[1], resolution)
    
    Q1, Q2 = np.meshgrid(q1_vals, q2_vals)
    
    dQ1 = np.zeros_like(Q1)
    dQ2 = np.zeros_like(Q2)

    for i in range(resolution):
        for j in range(resolution):
            q1, q2 = Q1[i, j], Q2[i, j]
            
            agent1.set_quantity(q1)
            agent2.set_quantity(q2)
            
            best_q1 = agent1.best_response(q2, inverse_demand_function, q_range)
            best_q2 = agent2.best_response(q1, inverse_demand_function, q_range)
            
            dQ1[i, j] = best_q1 - q1
            dQ2[i, j] = best_q2 - q2
    
    plt.figure(figsize=(8, 6))
    plt.streamplot(Q1, Q2, dQ1, dQ2, color=np.hypot(dQ1, dQ2), cmap='magma', linewidth=1)
    plt.xlabel(r'Firm 1 Quantity ($q_1$)')
    plt.ylabel(r'Firm 2 Quantity ($q_2$)')
    plt.title('Best Response Dynamics in Cournot')
    plt.colorbar(label='Response Magnitude')
    plt.grid(True)
    plt.show()


def explore_exponent_vs_quantity(inverse_demand_function, exp_range, num_seeds=10, q_range=(0, 100)):

    total_quantities = []
    exponents = []

    for exponent in exp_range:
        for seed in range(num_seeds):

            agents = [
                Agent('Firm1', general_cost, [0, 0.1, exponent]),
                Agent('Firm2', general_cost, [0, 0.1, exponent])
            ]
            game = OligopolyGame(agents, inverse_demand_function, q_range)
            game.random_initialization(seed=seed)

            game.simulate(max_iter=1000, tolerance=1e-5)

            total_q = game.total_quantity()
            total_quantities.append(total_q)
            exponents.append(exponent)

    norm = mcolors.Normalize(vmin=min(total_quantities), vmax=max(total_quantities))
    colormap = cm.magma
    dot_colors = colormap(norm(total_quantities))  # Assign colors based on total quantity values

    plt.figure(figsize=(8, 6))
    plt.scatter(exponents, total_quantities, c=dot_colors, cmap='magma', s=50, alpha=1)
    plt.xlabel("Cost Function Exponent")
    plt.ylabel("Total Quantity at Equilibrium")
    plt.title("Effect of Cost Exponent on Equilibrium Quantity")
    plt.grid(True)
    plt.show()


# exp_values = np.linspace(0.1, 20, 150) 
# explore_exponent_vs_quantity(inverse_demand_partial, exp_values, num_seeds=1)


def plot_game_results_vs_agents(base_agents, inverse_demand_function, max_agents=10):
    total_quantities = []
    prices = []
    agent_counts = range(2, max_agents + 1)

    for n_agents in agent_counts:
        agents = [Agent(f'Firm{i}', general_cost, [0, 0.5, 2]) for i in range(n_agents)]
        game = OligopolyGame(agents, inverse_demand_function)
        game.random_initialization()
        game.simulate()
        report = game.equilibrium_report()
        total_quantities.append(report['Total Quantity'])
        prices.append(report['Equilibrium Price'])

    plt.figure(figsize=(8, 6))
    plt.plot(agent_counts, total_quantities, label='Total Quantity', marker='o', color='darkorange')
    plt.plot(agent_counts, prices, label='Price', marker='s', color='red')
    plt.xlabel("Number of Agents")
    plt.ylabel("Market Results")
    plt.legend()
    plt.grid(True)
    plt.title("Market Outcomes vs Number of Agents")
    plt.show()


# def plot_heatmap(model_class, base_params, param_name1, param_values1, param_name2, param_values2, outcome_extractor, xlabel, ylabel, title):
#     results = np.zeros((len(param_values1), len(param_values2)))
    
#     for i, val1 in enumerate(param_values1):
#         for j, val2 in enumerate(param_values2):
#             agents = [
#                 Agent('Firm1', general_cost, [0, val1, 2]),
#                 Agent('Firm2', general_cost, [0, val2, 2])
#             ]
#             game = OligopolyGame(agents, inverse_demand_partial_3)
#             game.random_initialization()
#             game.simulate()
#             eq = game.equilibrium_report()
#             results[i, j] = outcome_extractor(eq)
    
#     plt.figure(figsize=(8, 6))
#     plt.imshow(results, aspect='auto', origin='lower', cmap='magma', extent=[param_values2[0], param_values2[-1], param_values1[0], param_values1[-1]])
#     plt.colorbar(label='Outcome Value')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.show()



def plot_stackelberg_vs_n_followers(base_agents, inverse_demand_function, max_followers=10):
    total_quantities = []
    prices = []
    follower_counts = range(1, max_followers + 1)

    for n_followers in follower_counts:
        leader = Agent('Leader', general_cost, [0, 0.5, 2], committed_quantity=50)
        followers = [Agent(f'Follower{i}', general_cost, [0, 0.5, 2]) for i in range(n_followers)]
        all_agents = [leader] + followers
        game = OligopolyGame(all_agents, inverse_demand_function)
        game.random_initialization()
        game.simulate()
        report = game.equilibrium_report()
        total_quantities.append(report['Total Quantity'])
        prices.append(report['Equilibrium Price'])

    plt.figure(figsize=(8, 6))
    plt.plot(follower_counts, total_quantities, label='Total Quantity', marker='o', color='darkorange')
    plt.plot(follower_counts, prices, label='Price', marker='s', color='red')
    plt.xlabel("Number of Followers")
    plt.ylabel("Market Outcomes")
    plt.legend()
    plt.grid(True)
    plt.title("Market Outcomes with Stackelberg Leader vs Number of Followers")
    plt.show()



def plot_leader_commitment_vs_market(base_agents, inverse_demand_function, commitment_values):
    total_quantities = []
    prices = []

    for commitment in commitment_values:
        leader = Agent('Leader', general_cost, [0, 0.5, 2], committed_quantity=commitment)
        followers = [Agent(f'Follower{i}', general_cost, [0, 0.5, 2]) for i in range(5)]
        all_agents = [leader] + followers
        game = OligopolyGame(all_agents, inverse_demand_function)
        game.random_initialization()
        game.simulate()
        report = game.equilibrium_report()
        total_quantities.append(report['Total Quantity'])
        prices.append(report['Equilibrium Price'])

    plt.figure(figsize=(8, 6))
    plt.plot(commitment_values, total_quantities, label='Total Quantity', marker='o', color='darkorange')
    plt.plot(commitment_values, prices, label='Price', marker='s', color='red')
    plt.xlabel("Leader's Committed Quantity")
    plt.ylabel("Market Outcomes")
    plt.legend()
    plt.grid(True)
    plt.title("Market Outcomes vs Leader's Commitment")
    plt.show()


# plot_game_results_vs_agents(agents, inverse_demand_partial, max_agents=50)

# param_values1 = np.linspace(0.1, 10, 50)
# param_values2 = np.linspace(0.1, 10, 50)
# plot_heatmap(OligopolyGame, {}, 'beta_1', param_values1, 'beta_2', param_values2, lambda eq: eq['Total Quantity'], 'Beta 1', 'Beta 2', 'Total Quantity Heatmap')

# plot_stackelberg_vs_n_followers(agents, inverse_demand_partial, max_followers=50)

# commitment_values = np.linspace(10, 100, 10)
# plot_leader_commitment_vs_market(agents, inverse_demand_partial, commitment_values)


def plot_agent_profits(game):
    report = game.equilibrium_report()
    agent_names = list(report['Individual Profits'].keys())
    profits = list(report['Individual Profits'].values())
    
    plt.figure(figsize=(8, 6))
    plt.bar(agent_names, profits, color=cm.magma(np.linspace(0.2, 0.8, len(agent_names))))
    plt.xlabel("Agents")
    plt.ylabel("Profit (Millions GBP)")
    plt.title("Equilibrium Profits of Agents")
    plt.xticks(rotation=20, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_cost_functions(agents):
    q_values = np.linspace(0, 100, 100)
    plt.figure(figsize=(8, 6))
    for i, agent in enumerate(agents):
        costs = [agent.cost(q) for q in q_values]
        plt.plot(q_values, costs, label=agent.id, color=cm.magma(i / len(agents)))
    
    plt.xlabel("Quantity (Millions Kg)")
    plt.ylabel("Variable Cost (Millions GBP)")
    plt.title("Variable Cost Functions of Firms")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_leader_commitment_vs_follower_profits(agents, inverse_demand_function, commitment_values):
    follower_profits = {agent.id: [] for agent in agents}  # Track profits for followers only

    for commitment in commitment_values:
        leader = Agent('Leader', general_cost, [0, 0.5, 2], committed_quantity=commitment)
        followers = [Agent(agent.id, agent.cost_function, agent.cost_params) for agent in agents]
        all_agents = [leader] + followers

        game = OligopolyGame(all_agents, inverse_demand_function)
        game.random_initialization()
        game.simulate()
        report = game.equilibrium_report()

        for agent in followers:
            if agent.id in report['Individual Profits']:
                follower_profits[agent.id].append(report['Individual Profits'][agent.id])
            else:
                follower_profits[agent.id].append(0)  

    plt.figure(figsize=(8, 6))
    color_map = cm.magma(np.linspace(0.1, 0.9, len(follower_profits)))
    for i, (agent_id, profits) in enumerate(follower_profits.items()):
        if len(profits) == len(commitment_values): 
            plt.plot(commitment_values, profits, label=agent_id, color=color_map[i])
    
    plt.xlabel("Leader's Committed Quantity")
    plt.ylabel("Follower Profit")
    plt.legend()
    plt.grid(True)
    plt.title("Follower Profits vs Leader's Commitment")
    plt.show()
