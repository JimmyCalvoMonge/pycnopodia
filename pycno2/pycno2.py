from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PycnoSIX2():

    def __init__(self, x00,
                 mu_I, mu_S,
                 delta, k,
                 lambda_J, lambda_A,
                 r_b, gamma_r,
                 alpha_b, gamma_alpha,
                 beta_b, gamma_beta,
                 ensi_file_path,
                 **kwargs):

        # Disease parameters
        self.mu_I = mu_I
        self.mu_S = mu_S
        self.delta = delta
        self.k = k
        self.lambda_J = lambda_J
        self.lambda_A = lambda_A
        self.r_b = r_b
        self.gamma_r = gamma_r
        self.alpha_b = alpha_b
        self.gamma_alpha = gamma_alpha
        self.beta_b = beta_b
        self.gamma_beta = gamma_beta

        col_name = kwargs.get('col_name', 'x')

        self.ensi_file_path = ensi_file_path
        data = pd.read_csv(self.ensi_file_path)
        self.MEI = data[col_name].tolist()

        # Simulation parameters
        self.x00 = x00
        self.t_max = data.shape[0]

    def state_odes_system(self, x, t):

        JS, JI, AS, AI = x[0], x[1], x[2], x[3]
        N = JS + JI + AS + AI
        du = [0]*4
        t = int(t)

        r = self.r_b + self.gamma_r*self.MEI[t]
        alpha = self.alpha_b + self.gamma_alpha*self.MEI[t]
        beta = self.beta_b + self.gamma_beta*self.MEI[t]

        du[0] = self.lambda_J + r*(1-N/self.k)*AS - beta*JS*(JI+AI)/N - alpha*JS - self.mu_S*JS
        du[1] = beta*JS*(JI+AI)/N - self.mu_I*JI - self.delta*JI
        du[2] = self.lambda_A + alpha*JS - beta*AS*(JI+AI)/N - self.mu_S*AS
        du[3] = beta*AS*(JI+AI)/N - self.mu_I*AI - self.delta*AI
    
        return [du[0], du[1], du[2], du[3]]

    def solve_odes_system_odeint(self):
        """
        Solve the classical system with initial conditions
        """
        t = list(range(0, self.t_max))
        self.time = t
        x = odeint(func=self.state_odes_system,
                    y0=self.x00, t=t, full_output=True)
        
        self.results = x

        self.JS = list(x[0][:, 0])
        self.JI = list(x[0][:, 1])
        self.AS = list(x[0][:, 2])
        self.AI = list(x[0][:, 3])

    def plot_ode_solution(self, **kwargs):
        
        title = kwargs.get('title', 'Plot')
    
        plt.plot(self.time, self.JS, label="Susceptible Juveniles")
        plt.plot(self.time, self.JI, label="Infected Juveniles")
        plt.plot(self.time, self.AS, label="Susceptible Adults")
        plt.plot(self.time, self.AI, label="Infected Adults")

        plt.title(title)
        plt.xlabel("Time (t)")
        plt.ylabel("Number of individuals")
        plt.legend(loc="upper right")
        plt.show()

    def plot_dN_dt(self):

        dNdt = [0]*self.t_max
        for t in range(0, self.t_max):
            N = sum([self.JS[t], self.JI[t], self.AS[t], self.AI[t]])
            r = self.r_b + self.gamma_r*self.MEI[t]
            dNdt[t] = self.lambda_J + self.lambda_A + r*(1-N/self.k)*self.AS[t] - \
            self.mu_S*(self.JS[t]+self.AS[t]) - (self.mu_I + self.delta)*(self.JI[t]+self.AI[t])

        plt.plot(self.time, dNdt, label=r"$\frac{dN}{dt}$")
        plt.title("Growth over time")
        plt.xlabel("Time (t)")
        plt.ylabel("Change in number of individuals")
        plt.legend(loc="upper right")
        plt.show()
        