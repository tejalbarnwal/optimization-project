import numpy as np
import random
import pandas as pd
import math
from matplotlib import pyplot as plt
import time


class Optimization:
    def __init__(self):
        self.history_obj = np.array([])
        self.history_temp = np.array([])
        self.history_x = np.array([])

    # HIGHLIGHT
    def objective_function(self, x):
        f = 10 + np.power(x, 2) - (10 * np.cos(2 * (np.pi) * x))
        return f

    # HIGHTLIGHT
    def find_neighbour(self, x, i):
        # random.seed(i)
        x_new = x + random.gauss(0, 1)
        while (x_new < -3) or (x_new > 3):
            x_new = x + random.gauss(0, 1)
        return x_new

    def store_results(self, x, objective_value, temperature):
        self.history_obj = np.append(self.history_obj, objective_value)
        self.history_temp = np.append(self.history_temp, temperature)
        self.history_x = np.append(self.history_x, x)

    def simulated_annealing(self, n_iterations, temperature, initial_guess, cooling_rate):
        # HIGHLIGHT
        # TUNING PARAMS: n_iterations, temperature, cooling schedule
        COOLING_RATE = cooling_rate
        CONSTANT_C = 1

        current_x = initial_guess
        current_objective_value = self.objective_function(current_x)

        print("Intial x:", current_x)
        print("Initial objective value:", current_objective_value)

        current_temperature = temperature

        self.store_results(current_x, 
                            current_objective_value, 
                            current_temperature)
        
        new_x = current_x

        for i in range(n_iterations):
            new_x = self.find_neighbour(current_x, i)
            new_objective_value = self.objective_function(new_x)

            # HIGHLIGHT
            if new_objective_value <= current_objective_value:
                print("condition 1: move ahead")
                current_x = new_x
                current_objective_value = new_objective_value

            elif new_objective_value > current_objective_value:
                print("__find probability__")
                diff_obj_value = (new_objective_value-current_objective_value)
                prob = np.exp(- CONSTANT_C * diff_obj_value / current_temperature)
                print("probability: ", prob)

                if prob >= random.uniform(0, 1):
                    print("condition 2: accepting solution with probabity")
                    current_x = new_x
                    current_objective_value = new_objective_value

                else:
                    pass
            else:
                pass

            current_temperature = temperature * np.power(COOLING_RATE, i)

            print("---------------")
            print("iteration number: ", i)
            print("current x: ", current_x)
            print("current objective value: ", current_objective_value)
            print("current_temperature: ", current_temperature, "\n")

            self.store_results(current_x, 
                            current_objective_value, 
                            current_temperature)


        return current_x


random.seed(1)
obj = Optimization()

## plot rastridge function 
# x = np.arange(-3, 3, 0.1, dtype=float)
# y = obj.objective_function(x)
# print("y: ", y)
# print("x:", x)
# plt.plot(x, y)
# plt.show()

init_guess = 2.99
n_iterations = 150
temperature = 10000
cooling_rate = 0.9

obj.simulated_annealing(n_iterations, temperature, init_guess, cooling_rate)

print("##########")

# print("history x: ", obj.history_x)

# print("history obj:", obj.history_obj)

# print("history temp", obj.history_temp)

n = len(obj.history_x)

# time.sleep()

plt.subplot(3, 1, 1)
plt.plot(obj.history_x[0], obj.history_obj[0], "ro-")
plt.subplot(3, 1, 2)
plt.plot(0, obj.history_temp[0], "ro-")
plt.subplot(3, 1, 3)
plt.plot(0, obj.history_obj[0], "ro-")
plt.pause(0.1)

for i in range(n):
    plt.subplot(3, 1, 1)
    plt.plot(obj.history_x[i], obj.history_obj[i], "y.")
    plt.title("Space x explored")
    plt.subplot(3, 1, 2)
    plt.plot(i, obj.history_temp[i], "c.")
    plt.title("Temp V/S Iterations")
    plt.subplot(3, 1, 3)
    plt.plot(i, obj.history_obj[i], "g.")
    plt.title("Objective value V/S Iterations")
    plt.pause(0.1)

plt.subplot(3, 1, 1)
plt.plot(obj.history_x[n-1], obj.history_obj[n-1], "ko-")
plt.subplot(3, 1, 2)
plt.plot(n-1, obj.history_temp[n-1], "ko-")
plt.subplot(3, 1, 3)
plt.plot(n-1, obj.history_obj[n-1], "ko-")
plt.pause(0.1)


plt.show()


