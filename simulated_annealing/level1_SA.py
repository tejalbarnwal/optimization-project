import numpy as np
import random
import pandas as pd
import math
from matplotlib import pyplot as plt
from timeit import default_timer as timer


class Optimization:
    def __init__(self, df):
        KM_M_CONVERSION = 1
        self.num_cities = 194
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
            # while new_path.tolist() in self.paths_visited_history:
            #     new_path = self.create_new_path2(current_path.copy())

            # print("feasible new path:", new_path)

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
            current_temperature = temperature / (np.log(100 + i))
            # N = 5000
            # current_temperature = temperature * ((N - 1 - i)/ N)
            # current_temperature = temperature - (temperature * (i/ N))

            # j = j + 1

            print("---------------")
            print("iteration number: ", i)
            # print("current path: ", current_path)
            print("current objective value: ", current_objective_value)
            print("current_temperature: ", current_temperature, "\n")
            print("-------------------------------------------")
            self.store_visited_paths(current_path)
            self.store_results(current_objective_value, current_temperature)

            # if round(current_temperature, 4) == 0.0:
            #     return current_path 
            # if sum(self.objective_value_history[-100:])/100 == self.objective_value_history[-1]:
            #     return current_path

            
            
        return current_path    


# define file
file = '/home/devank/tejal/acads/optimization-project/datasets/level1_benchmark/level1_cities194 - Sheet1.csv'
df = pd.read_csv(file)  

# define objective
obj = Optimization(df)

# cities
n_cities = obj.num_cities

# initial guess
# initial_guess = np.arange(0, n_cities)
# initial_guess = np.append(initial_guess, 0)
# initial_guess = np.array([0, 154, 189, 50, 142, 122, 191, 179, 187, 78, 190, 110, 40, 147, 159, 161, 54, 33, 22, 62, 114, 173, 152, 139, 88, 102, 73, 178, 26, 15, 165, 69, 83, 75, 37, 38, 5, 29, 4, 18, 23, 115, 3, 1, 98, 20, 175, 105, 35, 180, 149, 68, 91, 87, 177, 119, 131, 24, 71, 167, 140, 130, 148, 118, 52, 45, 129, 63, 59, 84, 111, 181, 13, 112, 77, 172, 86, 30, 116, 183, 92, 96, 151, 143, 128, 164, 155, 95, 10, 8, 53, 156, 9, 170, 44, 11, 57, 171, 160, 28, 6, 90, 133, 125, 184, 137, 169, 186, 32, 107, 46, 150, 176, 55, 80, 174, 85, 93, 19, 146, 138, 182, 153, 135, 117, 103, 193, 66, 36, 14, 56, 76, 67, 2, 61, 72, 168, 101, 123, 7, 70, 79, 16, 49, 41, 43, 134, 109, 89, 60, 136, 166, 145, 120, 47, 104, 81, 12, 34, 39, 100, 48, 158, 188, 192, 124, 99, 82, 121, 126, 157, 185, 162, 144, 163, 94, 21, 51, 42, 31, 106, 108, 132, 127, 74, 65, 58, 64, 97, 141, 113, 17, 27, 25, 0])

a = [i for i in range(1, n_cities)]
random.shuffle(a)

initial_guess = np.array([0])
initial_guess = np.append(initial_guess, np.array(a))
initial_guess = np.append(initial_guess, 0)

# travel time matrix
# print("travel_time_matrix: ")
# print(obj.travel_time_matrix)

# iterations
n_iterations = 10000
temperature = 20

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