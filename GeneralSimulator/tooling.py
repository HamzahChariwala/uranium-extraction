import numpy as np
from scipy.optimize import minimize_scalar


class Agent:
    def __init__(self, id, cost_function, cost_params=None, committed_quantity=0):
        self.id = id
        self.cost_function = cost_function
        self.cost_params = cost_params if cost_params else []
        self.committed_quantity = committed_quantity
        self.current_quantity = committed_quantity
        self.history_quantities = [committed_quantity] 

    def set_quantity(self, q):
        if self.committed_quantity == 0:
            self.current_quantity = q
        self.history_quantities.append(self.current_quantity)

    def cost(self, q):
        return self.cost_function(q, *self.cost_params)

    def profit(self, total_quantity, inverse_demand_function):
        price = inverse_demand_function(total_quantity)
        revenue = price * self.current_quantity
        cost = self.cost(self.current_quantity)
        return revenue - cost

    def best_response(self, other_quantities, inverse_demand_function, q_range):

        def negative_profit(q):
            total_q = q + other_quantities
            price = inverse_demand_function(total_q)
            return -(price * q - self.cost(q))

        res = minimize_scalar(negative_profit, bounds=q_range, method='bounded')
        return res.x


class OligopolyGame:
    def __init__(self, agents, inverse_demand_function, q_range=(0, 100)):
        self.agents = agents
        self.inverse_demand_function = inverse_demand_function
        self.q_range = q_range
    
    def total_quantity(self):
        return sum(agent.current_quantity for agent in self.agents)

    def random_initialization(self, seed=None):
        np.random.seed(seed)
        for agent in self.agents:
            if agent.committed_quantity == 0:
                random_q = np.random.uniform(*self.q_range)
                agent.set_quantity(random_q)
                agent.history_quantities = [random_q]
            else:
                agent.set_quantity(agent.committed_quantity)
                agent.history_quantities = [agent.committed_quantity]

    def simulate(self, max_iter=1000, tolerance=1e-6):
        for iteration in range(max_iter):
            quantities_prev = [agent.current_quantity for agent in self.agents]
            
            for agent in self.agents:
                if agent.committed_quantity == 0:
                    other_quantities = self.total_quantity() - agent.current_quantity
                    best_q = agent.best_response(other_quantities, self.inverse_demand_function, self.q_range)
                    agent.set_quantity(best_q)
            
            quantities_new = [agent.current_quantity for agent in self.agents]

            # Check convergence
            if max(abs(q_new - q_old) for q_new, q_old in zip(quantities_new, quantities_prev)) < tolerance:
                print(f"Equilibrium found after {iteration+1} iterations")
                break
        else:
            print("Equilibrium not reached within iteration limit")
        
        return {agent.id: agent.current_quantity for agent in self.agents}

    def equilibrium_report(self):
        total_q = sum(agent.current_quantity for agent in self.agents)
        price = self.inverse_demand_function(total_q)
        total_revenue = price * total_q
        individual_profits = {agent.id: agent.profit(total_q, self.inverse_demand_function)
                            for agent in self.agents}
        total_profit = sum(individual_profits.values())
        profit_shares = {id_: profit / total_profit for id_, profit in individual_profits.items()}

        report = {
            'Individual Quantities': {agent.id: agent.current_quantity for agent in self.agents},
            'Total Quantity': total_q,
            'Equilibrium Price': price,
            'Individual Profits': individual_profits,
            'Total Profit': total_profit,
            'Total Revenue': total_revenue,
            'Profit Shares': profit_shares,
            'Quantity Histories': {agent.id: agent.history_quantities for agent in self.agents}
        }
        return report
