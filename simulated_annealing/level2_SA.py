import numpy as np
import random
import pandas as pd
import math
from matplotlib import pyplot as plt
import time
from timeit import default_timer as timer


class Optimization:
    def __init__(self, df):
        KM_M_CONVERSION = 1000
        self.x_vector = df['X coord'].to_numpy().reshape(1, 10) * KM_M_CONVERSION
        self.y_vector = df['Y coord'].to_numpy().reshape(1, 10) * KM_M_CONVERSION

        self.demand_vector = df['Demand(kg)'].to_numpy().reshape(1, 10)
        self.service_time_vector = df['Service Time(sec)'].to_numpy().reshape(1, 10)

        self.speed = 5 #m/s
        self.max_capacity = 2 #kg
        self.max_recharge_time = 180 #sec
        self.max_energy = 50000 #joules #50000
        self.uav_mass = 10 #kg

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
            
    
    def check_capacity_constraints(self, new_path, demand_vector, max_capacity):
        demand_vector = demand_vector[0, new_path]
        payload_vector = np.round(np.cumsum(demand_vector), 2)
        if (np.sum(np.where(payload_vector < 0)) > 0) or (np.sum(np.where(payload_vector > max_capacity)) > 0):
            return False, None
        else:
            return True, payload_vector
        
    def power_required(self, payload):
        uav_mass = self.uav_mass
        speed = self.speed
        g = 10
        mot_eff, prop_eff = 0.9, 0.9
        const = speed * g / (mot_eff * prop_eff)
        return (payload + uav_mass) * const
    
    
    def check_refueling_constraints(self, new_path, payload_vector, travel_time_matrix, max_energy, power_function):
        # refactor this!!
        RECHARGE_RATE = max_energy / self.max_recharge_time #180 seconds for full charge
        # noise model for wind disturbances
        LAMBDA = 1000.0
        MEAN = 0.0
        VAR = 1.0
        SAFETY_FACTOR = 1.15
        
        PATH_LENGTH = len(new_path)
        
        current_energy = max_energy
        recharge_time = np.zeros((1,PATH_LENGTH))
                          
        for i in range(PATH_LENGTH - 1):
            payload_ij = payload_vector[i]
            # print("payload ij:", payload_ij)
            time_ij = travel_time_matrix[new_path[i], new_path[i+1]]
            required_energy_ij = power_function(payload_ij) * time_ij
            
            ATM_UNCERTAINITY = 0.0 #LAMBDA * random.gauss(MEAN, VAR)
            
            if current_energy >= SAFETY_FACTOR * required_energy_ij:
                current_energy = current_energy - (required_energy_ij + ATM_UNCERTAINITY)
            else:
                recharge_time[0, i] = (max_energy - current_energy) / RECHARGE_RATE
                current_energy = max_energy - (required_energy_ij + ATM_UNCERTAINITY)

        return recharge_time
    
        
    def objective_function(self, new_path, travel_time_matrix, recharge_time_vector, service_time_vector):
        total_travel_time = 0.0
        for i in range(len(new_path)-1):
            # print(i)
            total_travel_time += travel_time_matrix[new_path[i], new_path[i+1]] 
        service_time_vector = np.append(service_time_vector, 0.0)
        service_time_vector = service_time_vector[new_path]
        # print("service time vector:", service_time_vector)
        # print("recharge time vector shape:", recharge_time_vector.shape)
        # print("service time vector shape:", service_time_vector.shape)
        additional_time = np.sum(np.abs(recharge_time_vector - service_time_vector))
        return total_travel_time + additional_time
    
    
    def create_new_path(self, new_path):
        # CONSIDERING 0 AS DEPOT
        N_CITIES = len(new_path) - 1
        # random.seed(0)
        positions_to_swap = random.sample(range(1, N_CITIES), 2) 
        temp = new_path[positions_to_swap[0]]
        new_path[positions_to_swap[0]] = new_path[positions_to_swap[1]]
        new_path[positions_to_swap[1]] = temp
        
        return new_path
    
    def create_new_path1(self, new_path, i):
        # CONSIDERING 0 AS DEPOT
        N_CITIES = len(new_path) - 2
        
        position = random.sample(range(1, N_CITIES), 1) 
        positions_to_swap = [position[0], position[0]+1]
        temp = new_path[positions_to_swap[0]]
        new_path[positions_to_swap[0]] = new_path[positions_to_swap[1]]
        new_path[positions_to_swap[1]] = temp
        
        return new_path
    
    
    def store_visited_paths(self, path):
        self.paths_visited_history.append(path.copy())
#         print("paths visited:", self.paths_visited_history)
#         print("type paths visited:", type(self.paths_visited_history))
        
    
    def store_results(self, objective_value, temperature):
        self.objective_value_history.append(objective_value)
        self.tempertaure_history.append(temperature)
    

    def simulated_annealing(self, n_iterations, temperature, initial_guess):
        COOLING_RATE = 0.999
        CONSTANT_C = 1
        # N_CITIES = len(initial_guess) - 1
        
        # calculate a feasible initial path
        current_path = initial_guess.copy()
        bool_result, payload_vector = self.check_capacity_constraints(current_path, 
                                                                      self.demand_vector, self.max_capacity)
        while(not bool_result):
            current_path = self.create_new_path1(current_path, 0)
            bool_result, payload_vector = self.check_capacity_constraints(current_path, 
                                                                      self.demand_vector, self.max_capacity)
            
        print("feasible initial guess:", current_path)
        
        # calculate recharge time for obtained initial feasible path
        recharge_time_vector = self.check_refueling_constraints(current_path, 
                                                         payload_vector, 
                                                         self.travel_time_matrix, 
                                                         self.max_energy, self.power_required)
        
        # calculate objective function for obtained initial feasible path
        current_objective_value = self.objective_function(current_path, 
                                                             self.travel_time_matrix, 
                                                             recharge_time_vector, 
                                                             self.service_time_vector)
        print("current_objective_function:", current_objective_value)
        
        current_temperature = temperature
        
        self.store_visited_paths(current_path)
        self.store_results(current_objective_value, current_temperature)
        
        new_path = current_path.copy()
        print("first new path:", new_path, " type:", type(new_path))
        print("type paths visited:", type(self.paths_visited_history))
        
        print("##############################################################")
        
        j = 0
        
        for i in range(n_iterations):
            new_path = self.create_new_path1(current_path.copy(), i)
            bool_result, payload_vector = self.check_capacity_constraints(new_path, 
                                                                            self.demand_vector, 
                                                                            self.max_capacity)
            while(not bool_result):
                # create new feasible path
                new_path = self.create_new_path1(current_path.copy(), i)
                bool_result, payload_vector = self.check_capacity_constraints(new_path, 
                                                                      self.demand_vector, self.max_capacity)
                # print("#####################################")
            
            print("feasible new path:", new_path)
            # print("current path:", current_path) 
            
            # calculate recharge time for obtained initial feasible path
            new_recharge_time_vector = self.check_refueling_constraints(new_path, 
                                                             payload_vector, 
                                                             self.travel_time_matrix, 
                                                             self.max_energy, self.power_required)

            # calculate objective function for obtained initial feasible path
            new_objective_value = self.objective_function(new_path, 
                                                                 self.travel_time_matrix, 
                                                                 new_recharge_time_vector, 
                                                                 self.service_time_vector)
            
            # compare with the current path if the new one is better
            if new_objective_value <= current_objective_value:
                print("condition 1: move ahead")
                current_path = new_path.copy()
                current_objective_value = new_objective_value.copy()
                
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
                    current_objective_value = new_objective_value.copy()
                     
                else:
                    pass  
                
                # print("2 cond=>j:", j)
                # current_temperature = temperature * pow(COOLING_RATE, j)
            else:
                pass
                # print("3 cond=>j: ")
                # current_temperature = temperature * pow(COOLING_RATE, j)
            # update temperature
            # j = j + 1

            current_temperature = temperature * np.power(COOLING_RATE, i)
            
            print("---------------")
            print("iteration number: ", i)
            print("current path: ", current_path)
            print("current objective value: ", current_objective_value)
            print("current_temperature: ", current_temperature, "\n")
            print("-------------------------------------------")
            self.store_visited_paths(current_path)
            self.store_results(current_objective_value, current_temperature)
            
            
        return current_path    


random.seed(1)
file = '/home/radiant/Acads/ae755_project/project_venv/optimization-project/datasets/level2_dataset - Sheet2.csv'
df = pd.read_csv(file)  

obj = Optimization(df)
initial_guess = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
n_iterations = 6000
temperature = 10000

start = timer()#time.time()
solution = obj.simulated_annealing(n_iterations, temperature, initial_guess)
end = timer()#time.time()


print("time used:", end-start)

n = len(obj.objective_value_history)
a = np.arange(0, n)

# for i in range(n):
#     plt.subplot(2, 1, 1)
#     plt.plot(i, obj.tempertaure_history[i], "c.")
#     plt.title("temp vs iterations")
#     plt.subplot(2, 1, 2)
#     plt.plot(i, obj.objective_value_history[i], "g.")
#     plt.title("objective value vs iterations")
#     plt.pause(0.00000001)

plt.subplot(2, 1, 1)
plt.plot(a, obj.tempertaure_history, "k-")
plt.title("Temperature V/S no of iterations")
plt.xlabel("no of iterations")
plt.ylabel("Temperature")
plt.subplot(2, 1, 2)
plt.plot(a, obj.objective_value_history, "k-")
plt.title("Temperature V/S no of iterations")
plt.xlabel("no of iterations")
plt.ylabel("Objective value")
# plt.pause(0.0001)


plt.figure("2")
plt.scatter(obj.x_vector, obj.y_vector)
for i in range(len(solution)-1):
    j = solution[i]
    j_ = solution[i+1]
    x_list = [obj.x_vector[0, j], obj.x_vector[0, j_]]
    y_list = [obj.y_vector[0, j], obj.y_vector[0, j_]]
    # print(x_list, y_list)
    plt.plot(x_list, y_list)

    plt.plot(x_list, y_list, "ko-")
    plt.text(x_list[0], y_list[0], str(j), fontsize="small", backgroundcolor="r")


plt.show()