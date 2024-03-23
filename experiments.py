from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


class PycnoSIX():

    def __init__(self, mu_I, mu_S, alpha, beta_1, beta_2, k, p, r, x00, t_max, **kwargs):

        # Initial parameters
        self.mu_I = mu_I
        self.mu_S = mu_S
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.alpha = alpha
        self.k = k
        self.p = p
        self.r = r

        # Simulation parameters
        self.t_max = t_max
        self.x00 = x00

        # For odeint
        self.steps = kwargs.get('steps', 1000)

    def state_odes_system(self, x, t):

        JS1, JI1, AS1, AI1 = x[0], x[1], x[2], x[3]
        JS2, JI2, AS2, AI2 = x[4], x[5], x[6], x[7]
        
        N1 = JS1 + JI1 + AS1 + AI1
        N2 = JS2 + JI2 + AS2 + AI2
        du = [0]*8

        du[0] = self.p*self.r*(1-N1/self.k)*AS1-self.beta_1*JS1*(JI1+AI1)/N1-self.alpha*JS1-self.mu_S*JS1
        du[1] = self.beta_1*JS1*(JI1+AI1)/N1-self.mu_I*JI1
        du[2] = self.alpha*JS1-self.beta_1*AS1*(JI1+AI1)/N1-self.mu_S*AS1
        du[3] = self.beta_1*AS1*(JI1+AI1)/N1-self.mu_I*AI1

        du[4] = (1-self.p)*self.r*(1-N2/self.k)*AS1 + self.p*self.r*(1-N2/self.k)*AS2-self.beta_2*JS2*(JI2+AI2)/N2-self.alpha*JS2-self.mu_S*JS2
        du[5] = self.beta_2*JS2*(JI2+AI2)/N2-self.mu_I*JI2
        du[6] = self.alpha*JS2-self.beta_2*AS2*(JI2+AI2)/N2-self.mu_S*AS2
        du[7] = self.beta_2*AS2*(JI2+AI2)/N2-self.mu_I*AI2
    
        return [du[0], du[1], du[2], du[3], du[4], du[5], du[6], du[7]]

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
        
        self.Js2 = x[0][:, 4]
        self.Ji2 = x[0][:, 5]
        self.As2 = x[0][:, 6]
        self.Ai2 = x[0][:, 7]

    def plot_ode_solution(self, **kwargs):
        
        patch = kwargs.get('patch', 1)
        title = kwargs.get('title', f'Plot Patch(s) {patch}')
        
        if patch == 1:

            plt.plot(self.time, self.Js1, label="Susceptible Juveniles")
            plt.plot(self.time, self.Ji1, label="Infected Juveniles")
            plt.plot(self.time, self.As1, label="Susceptible Adults")
            plt.plot(self.time, self.Ai1, label="Infected Adults")
            
        elif patch == 2:
            
            plt.plot(self.time, self.Js2, label="Susceptible Juveniles")
            plt.plot(self.time, self.Ji2, label="Infected Juveniles")
            plt.plot(self.time, self.As2, label="Susceptible Adults")
            plt.plot(self.time, self.Ai2, label="Infected Adults")
            
        else:
            plt.plot(self.time, self.Js1, label="Susceptible Juveniles")
            plt.plot(self.time, self.Ji1, label="Infected Juveniles")
            plt.plot(self.time, self.As1, label="Susceptible Adults")
            plt.plot(self.time, self.Ai1, label="Infected Adults")
            plt.plot(self.time, self.Js2, label="Susceptible Juveniles")
            plt.plot(self.time, self.Ji2, label="Infected Juveniles")
            plt.plot(self.time, self.As2, label="Susceptible Adults")
            plt.plot(self.time, self.Ai2, label="Infected Adults")

        plt.title(title)
        plt.xlabel("Time (t)")
        plt.ylabel("Number of individuals")
        plt.legend(loc="upper right")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.show()
