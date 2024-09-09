import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def sir_model(y, time_point, beta, gamma):
    S, I, R = y
    population = S + I + R # Total Population (constant)
    dSdt = -beta * S * I / population # Susceptible to Infected
    dIdt = beta * S * I / population - gamma * I # Infected to Recovered
    dRdt = gamma * I # Recovered
    return dSdt, dIdt, dRdt

# Initial Parameters

population = 1000
I0 = 1
R0 = 0
S0 = population - I0 - R0

beta = 0.3 # Infection Rate
gamma = 0.1 # Recovery Rate

y0 = S0, I0, R0

time_point = np.linspace(0, 160, 160)

ret = odeint(sir_model, y0, time_point, args=(beta, gamma))
S, I, R = ret.T

plt.figure(figsize=(10, 6))
plt.plot(time_point, S, 'b', label='Susceptible')
plt.plot(time_point, I, 'r', label='Infected')
plt.plot(time_point, R, 'g', label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Devices')
plt.title('SIR Model Simulation of Computer Virus Spread')
plt.legend()
plt.show()
