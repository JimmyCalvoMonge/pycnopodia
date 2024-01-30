from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


class PycnoSIX():

    def __init__(self, mu_I, mu_S, alpha, beta, k, p, r, x00, t_max, **kwargs):

        # Initial parameters
        self.mu_I = mu_I
        self.mu_S = mu_S
        self.beta = beta
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.p = p
        self.r = r

        # Simulation parameters
        self.t_max = t_max
        self.x00 = x00

        # For odeint
        self.steps = kwargs.get('steps', 1000)

    def state_odes_system(self, x, t):

        J_s1 = x[0]
        J_i1 = x[1]
        A_s1 = x[2]
        A_i1 = x[3]
        N_1 = sum(x)

        dJs1dt = self.p*self.r*(1-N_1/self.k)*A_s1 - self.beta*J_s1*(J_i1+A_i1)/N_1 - self.alpha*J_s1 - self.mu_S*J_s1
        dJi1dt = self.beta*J_s1*(J_i1+A_i1)/N_1 - self.mu_I*J_i1
        dAs1dt = self.alpha*J_s1 - self.beta*A_s1*(J_i1+A_i1)/N_1 - self.mu_S*A_s1
        dAi1dt = self.beta*A_s1*(J_i1+A_i1)/N_1 - self.mu_I*A_i1

        return [dJs1dt, dJi1dt, dAs1dt, dAi1dt]

    def solve_odes_system_odeint(self):
        """
        Solve the classical system with initial conditions
        """
        t = np.linspace(0, 0 + self.t_max, self.steps)
        self.time = t
        x = odeint(func=self.state_odes_system,
                    y0=self.x00, t=t, full_output=True)

        self.Js1 = x[0][:, 0]
        self.Ji1 = x[0][:, 1]
        self.As1 = x[0][:, 2]
        self.Ai1 = x[0][:, 3]

    def plot_ode_solution(self, **kwargs):
        title = kwargs.get('title', 'Plot')
        plt.plot(self.time, self.Js1, label="Susceptible Juveniles")
        plt.plot(self.time, self.Ji1, label="Infected Juveniles")
        plt.plot(self.time, self.As1, label="Susceptible Adults")
        plt.plot(self.time, self.Ai1, label="Infected Adults")
        plt.title(title)
        plt.xlabel("Time (t)")
        plt.ylabel("Number of individuals")
        plt.legend(loc="upper right")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.show()
