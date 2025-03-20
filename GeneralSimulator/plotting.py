import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class OligopolyPlotter:
    def __init__(self, game):
        self.game = game

    def plot_heatmap(self, param_values, param_name, metric='Total Profit'):
 
        results = []
        
        for value in param_values:
            setattr(self.game, param_name, value)
            self.game.random_initialization()
            self.game.simulate()
            report = self.game.equilibrium_report()
            results.append(report[metric])
        
        results = np.array(results).reshape(len(param_values), -1)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(results, xticklabels=param_values, yticklabels=param_values, cmap='magma', annot=True)
        plt.xlabel(param_name)
        plt.ylabel(param_name)
        plt.title(f'Heatmap of {metric} over {param_name}')
        plt.show()

    def plot_line_graph(self, param_values, param_name, metric='Total Profit', models=None):

        if models is None:
            models = {'Base Model': self.game}

        plt.figure(figsize=(10, 6))
        
        for label, model in models.items():
            results = []
            
            for value in param_values:
                setattr(model, param_name, value)
                model.random_initialization()
                model.simulate()
                report = model.equilibrium_report()
                results.append(report[metric])
            
            plt.plot(param_values, results, label=label, marker='o', linestyle='-', color='black')
        
        plt.xlabel(param_name)
        plt.ylabel(metric)
        plt.title(f'Effect of {param_name} on {metric}')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_trajectory(self, firm_ids, num_trials=1):

        assert len(firm_ids) == 2, "This function only works for two firms."
        
        plt.figure(figsize=(8, 6))
        
        for _ in range(num_trials):
            self.game.random_initialization()
            self.game.simulate()
            report = self.game.equilibrium_report()
            
            q1_history = report['Quantity Histories'][firm_ids[0]]
            q2_history = report['Quantity Histories'][firm_ids[1]]
            
            plt.plot(q1_history, q2_history, marker='o', linestyle='-', color='darkred', alpha=0.6)
        
        plt.xlabel(f'Quantity of Firm {firm_ids[0]}')
        plt.ylabel(f'Quantity of Firm {firm_ids[1]}')
        plt.title('Trajectory of Quantities in 2D Space')
        plt.grid()
        plt.show()