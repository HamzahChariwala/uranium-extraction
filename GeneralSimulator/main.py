from tooling import OligopolyGame, Agent
from functools import partial
from plotting import OligopolyPlotter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import extra


def inverse_demand_1(Q, intercept, slope):
    return max(0, intercept - slope * Q)

def inverse_demand_2(Q, alpha, beta):
    return max(0, (alpha - Q) / beta)

inverse_demand_partial = partial(inverse_demand_2, alpha=100, beta=0.5)

def inverse_demand_elasticity(Q, alpha, epsilon):
    return max(0, alpha * Q ** (-epsilon))

inverse_demand_partial_2 = partial(inverse_demand_elasticity, alpha=100, epsilon=0.1)

def inverse_demand_quadratic(Q, alpha, beta, gamma):
    return max(0, alpha - beta * Q - gamma * Q**2)

inverse_demand_partial_3 = partial(inverse_demand_quadratic, alpha=200, beta=0, gamma=0.1)

uranium_partial = partial(inverse_demand_1, intercept=41.44, slope=0.0346)


def general_cost(q, fixed, variable, exponent):
    cost = fixed + (variable*q) ** exponent
    return cost

def uranium_cost(q, multiplier):
    cost = 0 + (multiplier * 0.99) * q**3.19
    return cost

# agents = [
#     Agent('Firm1', general_cost, [0,0.5,2]),
#     Agent('Firm2', general_cost, [0,0.1,2])
# ]


agents = [
    Agent('Government', uranium_cost, [1], committed_quantity=52.214),
    Agent('Cameco', uranium_cost, [1]),
    Agent('CGN', uranium_cost, [1.154]),
    Agent('Kazatomprom', uranium_cost, [0.841]),
    Agent('BHP Billiton', uranium_cost, [1.018]),
    Agent('Orano', uranium_cost, [1.066])
]

# Create game instance
game = OligopolyGame(agents, uranium_partial)

# Random initialization
game.random_initialization(seed=123)

# Simulate equilibrium
game.simulate(max_iter=1000, tolerance=1e-5)

# Generate equilibrium report
report = game.equilibrium_report()
for key, val in report.items():
    if key != 'Quantity Histories':
        print(f"{key}: {val}")

plotter = OligopolyPlotter(game)
# plotter.plot_trajectory(firm_ids=['Firm1', 'Firm2'], num_trials=5)

# intercept_values = np.linspace(50, 150, 10)  # Example range 
# plotter.plot_heatmap(intercept_values, param_name="inverse_demand_function.alpha", metric="Total Profit")


# extra.plot_best_response_stream(agents[0], agents[1], inverse_demand_partial, q_range=(0, 50), resolution=150)


# commitment_values = np.linspace(10, 100, 10)  
# extra.plot_leader_commitment_vs_follower_profits(agents, inverse_demand_partial, commitment_values)

previous_profits = {'Cameco': 90.4, 'CGN': 84.7, 'Kazatomprom': 97.8, 'BHP Billiton': 89.6, 'Orano': 87.8}
extra.plot_profit_difference_bar_chart(report, previous_profits)
