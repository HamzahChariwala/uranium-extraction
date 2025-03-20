class OligopolyModel:

    def __init__(self, alpha_d, beta_d, beta_1, beta_2):
        self.alpha_d = alpha_d
        self.beta_d = beta_d
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def compute_equilibrium(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def total_output(self):
        eq = self.compute_equilibrium()
        return eq['q1'] + eq['q2']

    def equilibrium_price(self):
        eq = self.compute_equilibrium()
        return eq['price']


class CournotModel(OligopolyModel):

    def best_response_firm1(self, q2):
        return 0.5 * (self.alpha_d - q2 - self.beta_1 * self.beta_d)
    
    def best_response_firm2(self, q1):
        return 0.5 * (self.alpha_d - q1 - self.beta_2 * self.beta_d)
    
    def compute_equilibrium(self):
        q1_star = (1/3) * (self.alpha_d - 2 * self.beta_1 * self.beta_d + self.beta_2 * self.beta_d)
        q2_star = (1/3) * (self.alpha_d + self.beta_1 * self.beta_d - 2 * self.beta_2 * self.beta_d)
        total_q = q1_star + q2_star 
        price = (self.alpha_d / (3 * self.beta_d)) + ((self.beta_1 + self.beta_2) / 3)
        profit1 = (1 / (9 * self.beta_d)) * (self.alpha_d - 2 * self.beta_1 * self.beta_d + self.beta_2 * self.beta_d) ** 2
        profit2 = (1 / (9 * self.beta_d)) * (self.alpha_d + self.beta_1 * self.beta_d - 2 * self.beta_2 * self.beta_d) ** 2
        total_profit = profit1 + profit2
        total_revenue = total_q * price
        firm1_profit_share = profit1*100/total_profit
        firm2_profit_share = 100-firm1_profit_share
        profit_per_Q = total_profit / total_q
        
        return {
            'q1': q1_star,
            'q2': q2_star,
            'total_q': total_q,
            'price': price,
            'profit1': profit1,
            'profit2': profit2,
            'total_pi': total_profit,
            'total_r': total_revenue,
            'pi1_share': firm1_profit_share,
            'pi2_share': firm2_profit_share,
            'pi/Q': profit_per_Q
        }


class StackelbergModel(OligopolyModel):

    def follower_response(self, q1):
        return 0.5 * (self.alpha_d - q1 - self.beta_2 * self.beta_d)
    
    def compute_equilibrium(self):
        q1_star = 0.5 * (self.alpha_d + self.beta_2 * self.beta_d - 2 * self.beta_1 * self.beta_d)
        q2_star = 0.25 * (self.alpha_d + 2 * self.beta_1 * self.beta_d - 3 * self.beta_2 * self.beta_d)
        total_q = q1_star + q2_star 
        price = (self.alpha_d / (4 * self.beta_d)) + ((2 * self.beta_1 + self.beta_2) / 4)
        profit1 = (1 / (8 * self.beta_d)) * (self.alpha_d - 2 * self.beta_1 * self.beta_d + self.beta_2 * self.beta_d) ** 2
        profit2 = (1 / (16 * self.beta_d)) * (self.alpha_d + 2 * self.beta_1 * self.beta_d - 3 * self.beta_2 * self.beta_d) ** 2
        total_profit = profit1 + profit2
        total_revenue = total_q * price
        firm1_profit_share = profit1*100/total_profit
        firm2_profit_share = 100-firm1_profit_share
        profit_per_Q = total_profit / total_q
        
        return {
            'q1': q1_star,
            'q2': q2_star,
            'total_q': total_q,
            'price': price,
            'profit1': profit1,
            'profit2': profit2,
            'total_pi': total_profit,
            'total_r': total_revenue,
            'pi1_share': firm1_profit_share,
            'pi2_share': firm2_profit_share,
            'pi/Q': profit_per_Q
        }

# cournot = CournotModel(alpha_d=100, beta_d=2, beta_1=1, beta_2=1)
# cournot_eq = cournot.compute_equilibrium()
# print("Cournot Equilibrium:", cournot_eq)

# stackelberg = StackelbergModel(alpha_d=100, beta_d=0.5, beta_1=0.5, beta_2=0.1)
# stackelberg_eq = stackelberg.compute_equilibrium()
# print("Stackelberg Equilibrium:", stackelberg_eq)

# for beta_1 in [0.8, 1.0, 1.2]:
#     model = CournotModel(alpha_d=100, beta_d=2, beta_1=beta_1, beta_2=1.2)
#     eq = model.compute_equilibrium()
#     print(f"beta_1 = {beta_1}, Equilibrium: {eq}")


class QuadraticCostCournotModel(OligopolyModel):
    def __init__(self, alpha_d, beta_d, beta_1, beta_2, mu_1, mu_2):
        super().__init__(alpha_d, beta_d, beta_1, beta_2)
        self.mu_1 = mu_1
        self.mu_2 = mu_2

    def compute_equilibrium(self):
        theta = (4 * self.beta_d**2 * self.beta_1**2 * self.beta_2**2 + 
                 4 * self.beta_d * (self.beta_1**2 + self.beta_2**2) + 3)
        
        q1_star = (1/theta) * (2 * self.alpha_d * self.beta_d * self.beta_2**2 + self.alpha_d)
        q2_star = (1/theta) * (2 * self.alpha_d * self.beta_d * self.beta_1**2 + self.alpha_d)
        total_q = (1/theta) * (2 * self.alpha_d * (self.beta_d * self.beta_1**2 + self.beta_d * self.beta_2**2 + 1))
        price = (self.alpha_d / (self.beta_d * theta)) * (theta - 2 * (self.beta_d * (self.beta_1**2 + self.beta_2**2) + 1))
        
        profit1 = (1/theta**2) * (self.alpha_d**2 * (self.beta_d * self.beta_1**2 + 1) * (2 * self.beta_d * self.beta_2**2 + 1)**2) - self.mu_1
        profit2 = (1/theta**2) * (self.alpha_d**2 * (2 * self.beta_d * self.beta_1**2 + 1)**2 * (self.beta_d * self.beta_2**2 + 1)) - self.mu_2
        total_profit = profit1 + profit2
        total_revenue = total_q * price
        firm1_profit_share = profit1 * 100 / total_profit
        firm2_profit_share = 100 - firm1_profit_share
        profit_per_Q = total_profit / total_q
        
        return {
            'q1': q1_star,
            'q2': q2_star,
            'total_q': total_q,
            'price': price,
            'profit1': profit1,
            'profit2': profit2,
            'total_pi': total_profit,
            'total_r': total_revenue,
            'pi1_share': firm1_profit_share,
            'pi2_share': firm2_profit_share,
            'pi/Q': profit_per_Q
        }

