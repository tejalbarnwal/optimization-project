import numpy as np
import random
import pandas as pd
import math
from matplotlib import pyplot as plt
from timeit import default_timer as timer


class Optimization:
    def __init__(self, df):
        KM_M_CONVERSION = 1
        self.num_cities = 38
        self.x_vector = df['X coord'].to_numpy().reshape(1, self.num_cities) * KM_M_CONVERSION
        self.y_vector = df['Y coord'].to_numpy().reshape(1, self.num_cities) * KM_M_CONVERSION

        self.speed = 1 #m/s

        self.travel_time_matrix = self.generate_travel_time_matrix(self.x_vector, self.y_vector, self.speed)
        
        # memory of results
        self.paths_visited_history = []
        self.objective_value_history = []
        self.tempertaure_history = []
        
        
    def generate_travel_time_matrix(self, x, y, speed):
        coordinates = np.concatenate((x, y), axis=0)
        # print("c:", coordinates.shape)
        a = coordinates.transpose()
        b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
        D = np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()   
        return D/speed
            
            
    def objective_function(self, new_path, travel_time_matrix):
        total_travel_time = 0.0
        for i in range(len(new_path)-1):
            total_travel_time += travel_time_matrix[new_path[i], new_path[i+1]] 
        return total_travel_time
    
    
    def create_new_path(self, new_path):
        # CONSIDERING 0 AS DEPOT
        N_CITIES = len(new_path) - 1
        positions_to_swap = random.sample(range(1, N_CITIES), 2) 
        temp = new_path[positions_to_swap[0]]
        new_path[positions_to_swap[0]] = new_path[positions_to_swap[1]]
        new_path[positions_to_swap[1]] = temp

        return new_path
    
    def create_new_path1(self, new_path):
        N_CITIES = len(new_path) - 2
        position = random.sample(range(1, N_CITIES), 1) 
        positions_to_swap = [position[0], position[0]+1]
        temp = new_path[positions_to_swap[0]]
        new_path[positions_to_swap[0]] = new_path[positions_to_swap[1]]
        new_path[positions_to_swap[1]] = temp
        
        return new_path

    def create_new_path2(self, new_path):
        N_CITIES = len(new_path) - 1
        i, j = random.sample(range(1, N_CITIES), 2) 
        if (i > j):
            i, j = j, i
        new_path[i:j+1] = np.flip(new_path[i:j+1])
        
        return new_path
    
    
    def store_visited_paths(self, path):
        self.paths_visited_history.append((path.tolist()).copy())
#         print("paths visited:", self.paths_visited_history)
#         print("type paths visited:", type(self.paths_visited_history))
        
    
    def store_results(self, objective_value, temperature):
        self.objective_value_history.append(objective_value)
        self.tempertaure_history.append(temperature)


    def simulated_annealing(self, n_iterations, temperature, initial_guess):
        COOLING_RATE = 0.998
        CONSTANT_C = 1
        
        # calculate a feasible initial path
        current_path = initial_guess.copy()
        
        # calculate objective function for obtained initial feasible path
        current_objective_value = self.objective_function(current_path, self.travel_time_matrix)
        print("current_objective_function:", current_objective_value)
        
        current_temperature = temperature
        
        self.store_visited_paths(current_path)
        self.store_results(current_objective_value, current_temperature)
        
        new_path = current_path.copy()
        j = 0
                
        for i in range(n_iterations):

            new_path = self.create_new_path2(current_path.copy())
            while new_path.tolist() in self.paths_visited_history:
                new_path = self.create_new_path2(current_path.copy())

            print("feasible new path:", new_path)

            new_objective_value = self.objective_function(new_path, self.travel_time_matrix)
            
            # compare with the current path if the new one is better
            if new_objective_value <= current_objective_value:
                print("condition 1: move ahead")
                current_path = new_path.copy()
                current_objective_value = new_objective_value
                
                # current_temperature = temperature
                # j = 0
                
            elif new_objective_value > current_objective_value:
                print("__find probability__")
                # accept the solution with some proabablity
                diff_obj_value = (new_objective_value-current_objective_value)
                probability = np.exp(- CONSTANT_C * diff_obj_value / current_temperature)
                print("probability: ", probability)

                if probability >= random.uniform(0, 1):
                    print("condition 2: accepting solution with probabity")
                    current_path = new_path.copy()
                    current_objective_value = new_objective_value
                     
                else:
                    pass  
            else:
                pass

            
            # current_temperature = temperature * np.power(COOLING_RATE, i)
            current_temperature = temperature / (4.0 * np.log(100 + i*100))
            # N = 5000
            # current_temperature = temperature * ((N - 1 - i)/ N)
            # current_temperature = temperature - (temperature * (i/ N))

            # j = j + 1

            print("---------------")
            print("iteration number: ", i)
            print("current path: ", current_path)
            print("current objective value: ", current_objective_value)
            print("current_temperature: ", current_temperature, "\n")
            print("-------------------------------------------")
            self.store_visited_paths(current_path)
            self.store_results(current_objective_value, current_temperature)

            # if round(current_temperature, 4) == 0.0:
            #     return current_path 
            if sum(self.objective_value_history[-100:])/100 == self.objective_value_history[-1]:
                return current_path

            
            
        return current_path    


# define file
file = '/home/devank/tejal/acads/optimization-project/datasets/level1_benchmark/level1_cities38 - Sheet1.csv'
df = pd.read_csv(file)  

# define objective
obj = Optimization(df)

# cities
n_cities = obj.num_cities

# initial guess
initial_guess = np.arange(0, n_cities)
initial_guess = np.append(initial_guess, 0)

# travel time matrix
print("travel_time_matrix: ")
print(obj.travel_time_matrix)

# iterations
n_iterations = 50000
temperature = 150

# time
start = timer()
solution = obj.simulated_annealing(n_iterations, temperature, initial_guess)
end = timer()
print("time used:", end-start)


n = len(obj.objective_value_history)
a = np.arange(0, n)


plt.subplot(2, 1, 1)
plt.plot(a, obj.tempertaure_history, "k-")
plt.title("Temperature V/S no of iterations")
plt.xlabel("no of iterations")
plt.ylabel("Temperature")
plt.subplot(2, 1, 2)
plt.plot(a, obj.objective_value_history, "k-")
plt.title("Objective Value V/S no of iterations")
plt.xlabel("no of iterations")
plt.ylabel("Objective Value")
plt.pause(0.0001)

plt.figure("2")
plt.scatter(obj.y_vector, obj.x_vector)
for i in range(len(solution)-1):
    j = solution[i]
    j_ = solution[i+1]
    x_list = [obj.x_vector[0, j], obj.x_vector[0, j_]]
    y_list = [obj.y_vector[0, j], obj.y_vector[0, j_]]
    # print(x_list, y_list)
    plt.plot(y_list, x_list, "ko-")
    # plt.text(x_list[0], y_list[0], str(j), fontsize="small", backgroundcolor="r")


plt.show()