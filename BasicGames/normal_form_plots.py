from Oligopoly import CournotModel, StackelbergModel, QuadraticCostCournotModel
from Plotting import ModelPlotter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    base_params = {'alpha_d': 100, 'beta_d': 0.5, 'beta_1': 10, 'beta_2': 0.1}
    
    print("Testing CournotModel with ModelPlotter:")
    cournot_plotter = ModelPlotter(CournotModel, base_params)

    beta_1_values = cournot_plotter.generate_param_values(0.1, 10, 10)
    beta_2_values = cournot_plotter.generate_param_values(0.1, 10, 0.1)
    alpha_d_values = cournot_plotter.generate_param_values(100, 200, 1)
    beta_d_values = cournot_plotter.generate_param_values(1, 5, 0.1)

    # cournot_plotter.plot_heat_map(
    #     param_name1='beta_1',
    #     param_values1=beta_1_values,
    #     param_name2='beta_2',
    #     param_values2=beta_2_values,
    #     outcome_extractor=lambda eq: eq['pi/Q'],
    #     xlabel='beta_1',
    #     ylabel='beta_2',
    #     title='Total Profit:Quantity Ratio (Cournot with q^2 Cost)'
    # )

cournot_quad = QuadraticCostCournotModel(alpha_d=100, beta_d=0.5, beta_1=0.5, beta_2=0.1, mu_1=0, mu_2=0)
cournot_eq = cournot_quad.compute_equilibrium()
print("Cournot Equilibrium:", cournot_eq)


# cournot_plotter.plot_two_models(CournotModel,
#                                 StackelbergModel,
#                                 'alpha_d',
#                                 alpha_d_values,
#                                 lambda eq: eq['pi/Q'],
#                                 'Cournot',
#                                 'Stackelberg',
#                                 'Alpha_d',
#                                 '(Pi_1 + Pi_2)/Q',
#                                 'Profit per unit according to alpha_d')

def plot_best_response_cournot_vs_stackelberg(alpha_d, beta_d, beta_1, beta_2):
    cournot = CournotModel(alpha_d, beta_d, beta_1, beta_2)
    stackelberg = StackelbergModel(alpha_d, beta_d, beta_1, beta_2)

    cournot_eq = cournot.compute_equilibrium()
    q1_cournot, q2_cournot = cournot_eq['q1'], cournot_eq['q2']

    stackelberg_eq = stackelberg.compute_equilibrium()
    q1_stackelberg, q2_stackelberg = stackelberg_eq['q1'], stackelberg_eq['q2']

    q2_values = np.linspace(0, alpha_d, 100)
    q1_best_response = [cournot.best_response_firm1(q2) for q2 in q2_values]

    q1_values = np.linspace(0, alpha_d, 100)
    q2_best_response = [cournot.best_response_firm2(q1) for q1 in q1_values]

    plt.figure(figsize=(8, 6))
    plt.plot(q1_best_response, q2_values, label="Firm 1 Best Response", color="black", linestyle='dashed')
    plt.plot(q1_values, q2_best_response, label="Firm 2 Best Response", color="black")
    plt.scatter(q1_cournot, q2_cournot, color="purple", s=100, label="Cournot Equilibrium")
    plt.scatter(q1_stackelberg, q2_stackelberg, color="red", s=100, label="Stackelberg Equilibrium")
    plt.xlabel("$q_1$")
    plt.ylabel("$q_2$")
    plt.title("Best Response Functions: Cournot vs Stackelberg")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_output_vs_price(alpha_d, beta_d, beta_1, beta_2):
    cournot = CournotModel(alpha_d, beta_d, beta_1, beta_2)
    stackelberg = StackelbergModel(alpha_d, beta_d, beta_1, beta_2)

    cournot_eq = cournot.compute_equilibrium()
    Q_cournot, P_cournot = cournot_eq['total_q'], cournot_eq['price']

    stackelberg_eq = stackelberg.compute_equilibrium()
    Q_stackelberg, P_stackelberg = stackelberg_eq['total_q'], stackelberg_eq['price']

    MC_perfect_competition = min(beta_1, beta_2)
    Q_perfect_competition = alpha_d / beta_d  

    plt.figure(figsize=(8, 6))
    plt.scatter(Q_cournot, P_cournot, color="purple", s=100, label="Cournot Equilibrium")
    plt.scatter(Q_stackelberg, P_stackelberg, color="red", s=100, label="Stackelberg Equilibrium")
    plt.scatter(Q_perfect_competition, MC_perfect_competition, color="orange", s=100, label="Perfect Competition")
    plt.xlabel("Total Output ($Q$)")
    plt.ylabel("Equilibrium Price ($P$)")
    plt.xlim(0)
    plt.title("Equilibrium Price vs Total Output")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "Cournot": {"Q": Q_cournot, "P": P_cournot},
        "Stackelberg": {"Q": Q_stackelberg, "P": P_stackelberg},
        "Perfect Competition": {"Q": Q_perfect_competition, "P": MC_perfect_competition}
    }


equilibrium_values = plot_output_vs_price(alpha_d=100, beta_d=0.5, beta_1=0.5, beta_2=0.1)
# plot_best_response_cournot_vs_stackelberg(alpha_d=100, beta_d=2, beta_1=1.0, beta_2=1.0)
