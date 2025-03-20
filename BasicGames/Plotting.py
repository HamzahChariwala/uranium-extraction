import numpy as np
import matplotlib.pyplot as plt

class ModelPlotter:
    def __init__(self, model_class, base_params):
        self.model_class = model_class
        self.base_params = base_params.copy()
        
    @staticmethod
    def generate_param_values(min_val, max_val, step_size):
        return np.arange(min_val, max_val + step_size, step_size)
    
    def _sweep_parameter(self, param_name, param_values, outcome_extractor):
        outcomes = []
        for val in param_values:
            params = self.base_params.copy()
            params[param_name] = val
            model_instance = self.model_class(**params)
            eq_data = model_instance.compute_equilibrium()
            outcomes.append(outcome_extractor(eq_data))
        return param_values, outcomes
    
    def plot_line(self, param_name, param_values, outcome_extractor, xlabel=None, ylabel=None, title=None):

        x_vals, y_vals = self._sweep_parameter(param_name, param_values, outcome_extractor)
        
        plt.figure(figsize=(8,6))
        plt.plot(x_vals, y_vals, marker=None)
        plt.xlabel(xlabel if xlabel else param_name)
        plt.ylabel(ylabel if ylabel else 'Outcome')
        if title:
            plt.title(title)
        plt.grid(True)
        plt.show()
        
        return x_vals, y_vals
    
    def plot_heat_map(self, param_name1, param_values1, param_name2, param_values2, outcome_extractor, xlabel=None, ylabel=None, title=None):

        heat_data = []
        for val2 in param_values2:
            row = []
            for val1 in param_values1:
                params = self.base_params.copy()
                params[param_name1] = val1
                params[param_name2] = val2
                model_instance = self.model_class(**params)
                eq_data = model_instance.compute_equilibrium()
                row.append(outcome_extractor(eq_data))
            heat_data.append(row)
        
        heat_data = np.array(heat_data)
        
        plt.figure(figsize=(8,6))
        im = plt.imshow(heat_data, aspect='auto', origin='lower', cmap='magma',
                   extent=[min(param_values1), max(param_values1), min(param_values2), max(param_values2)])
        plt.contour(heat_data, 
                    extent=[min(param_values1), max(param_values1), min(param_values2), max(param_values2)],
                    colors='white',alpha=0.5,linewdiths=0.5)
        plt.colorbar(im, label='Outcome')
        plt.xlabel(xlabel if xlabel else param_name1)
        plt.ylabel(ylabel if ylabel else param_name2)
        if title:
            plt.title(title)
        plt.show()
        
        return param_values1, param_values2, heat_data
    
    def plot_two_models(self, model_class_1, model_class_2, param_name, param_values, outcome_extractor,
                    label_1='Model 1', label_2='Model 2', xlabel=None, ylabel=None, title=None):

        def sweep_model(model_cls):
            outcomes = []
            for val in param_values:
                params = self.base_params.copy()
                params[param_name] = val
                model_instance = model_cls(**params)
                eq_data = model_instance.compute_equilibrium()
                outcomes.append(outcome_extractor(eq_data))
            return outcomes

        outcomes_1 = sweep_model(model_class_1)
        outcomes_2 = sweep_model(model_class_2)

        plt.figure(figsize=(8, 6))
        plt.plot(param_values, outcomes_1, marker=None, label=label_1, color='purple')
        plt.plot(param_values, outcomes_2, marker=None, label=label_2, color='red')
        plt.xlabel(xlabel if xlabel else param_name)
        plt.ylabel(ylabel if ylabel else 'Outcome')
        if title:
            plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.show()

        return param_values, outcomes_1, outcomes_2

    def plot_outcome_space(self, param_name, param_values, x_outcome_extractor, y_outcome_extractor,
                       xlabel='Outcome X', ylabel='Outcome Y', title=None):

        x_outcomes = []
        y_outcomes = []

        for val in param_values:
            params = self.base_params.copy()
            params[param_name] = val
            model_instance = self.model_class(**params)
            eq_data = model_instance.compute_equilibrium()
            x_outcomes.append(x_outcome_extractor(eq_data))
            y_outcomes.append(y_outcome_extractor(eq_data))

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(x_outcomes, y_outcomes, c=param_values, cmap='magma', edgecolor='k')
        cbar = plt.colorbar(scatter)
        cbar.set_label(param_name)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)
        plt.grid(True)
        plt.show()

        return x_outcomes, y_outcomes, param_values
