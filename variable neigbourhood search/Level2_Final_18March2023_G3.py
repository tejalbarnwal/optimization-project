# Required Libraries
import pandas as pd
import numpy as np
import random
import copy
import math
from matplotlib import pyplot as plt 
import time

# Function: Tour Distance
def distance_calc(XData, city_tour):
    distance = 0
    for k in range(0, len(city_tour)-1):
        m = k + 1
        distance = distance + XData[city_tour[k-1], city_tour[m-1]]          
    return distance


# Function: Build Distance Matrix
def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   D = np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()
   np.savetxt("Distance_Matrix.txt", D)
   return D

    
# Function: Stochastic 2_opt
def stochastic_2_opt(XData, city_tour, Demand_vector):
    best_route = []
    Distance = 0
    Pay_load_vector = []
    pairs = []
    S = False
    city_tour_C = city_tour.copy()
    while S == False:
        i, j  = random.sample(range(1, len(city_tour_C)-1), 2) # range start changed from 0 to 1 to keep depot as starting point always
        # print(i, j)
        if [i, j] not in pairs and [j, i] not in pairs:
            pairs.append([i, j])
            # print(pairs)
        if (i > j):
            i, j = j, i
        city_tour_C[i:j+1] = np.flip(city_tour_C[i:j+1])
        city_tour_C[-1]  = city_tour_C[0]
        best_route.append(city_tour_C)
        if Capacity_Check(city_tour_C, Demand_vector)[0] == True:
            Pay_load_vector = Capacity_Check(city_tour_C, Demand_vector)[2]
            # Distance = distance_calc(XData, city_tour_C)
            S = True
        else:
            S = False
    return city_tour_C, Pay_load_vector

# Function: Local Search
def local_search(XData, city_tour, Demand_vector, max_attempts = 5, neighbourhood_size = 3):
    count = 0
    Distance = 0
    solution = city_tour.copy()
    # Distance_Sol = distance_calc(XData, solution)
    while (count < max_attempts):
        for i in range(0, neighbourhood_size):
            candidate = stochastic_2_opt(XData, solution, Demand_vector)[0]
            # Distance = distance_calc(XData, candidate)
            # if Distance < Distance_Sol:
            solution  = candidate.copy()
        count = count + 1                     
    return solution 

# Function: Variable Neighborhood Search
def variable_neighborhood_search(XData, Best_Route, Demand_vector, max_attempts = 5, neighbourhood_size = 3, iterations = 10):
    global TData, service_time_vector

    count = 0
    Distance = 0
    Distance_BSol = 0
    Pay_load_vector = []
    solution = Best_Route.copy()
    best_solution = Best_Route.copy()

    best_solution, payload_vector = stochastic_2_opt(XData, best_solution, Demand_vector)
    print("feasible initial guess: ", best_solution)
    recharge_time_vector = energy_req(best_solution, payload_vector, TData, total_energy=50000)
    # print("inital reharge time vector:", recharge_time_vector)
    Distance_BSol = objective_function(best_solution, TData, recharge_time_vector, service_time_vector)
    print("total time: ", Distance_BSol)

    time.sleep(2)
    print("##############################################################################")
    print()

    # Distance_BSol = distance_calc(XData, best_solution)
    while (count < iterations):
        for i in range(0, neighbourhood_size):
            for j in range(0, neighbourhood_size):
                solution, payload_vector = stochastic_2_opt(XData, best_solution, Demand_vector)
                print("vns new solution sto: ", solution)
                recharge_time_vector = energy_req(solution, payload_vector, TData, total_energy=50000)
                # print("recharge time vector:", recharge_time_vector)
                Distance = objective_function(solution, TData, recharge_time_vector, service_time_vector)
                print("total time: ", Distance)


                print("iteration number: ", count)
                if (Distance < Distance_BSol):
                    best_solution = solution.copy()
                    Distance_BSol = Distance
                    #break
                print("\n-----\n")
        count = count + 1
        # time.sleep(1)
        print("Iteration = ", count-1, "end ka obj value:", Distance_BSol, "end ka best sol:", best_solution)
        print("-----------------------------------------------")
    return best_solution

def plot_tour_coordinates (coordinates, city_tour, Cust_ID):
    xy = np.zeros((len(city_tour), 2))
    for i in range(0, len(city_tour)):
        if (i < len(city_tour)):
            xy[i, 0] = coordinates[city_tour[i]-1, 0]
            xy[i, 1] = coordinates[city_tour[i]-1, 1]
        else:
            xy[i, 0] = coordinates[city_tour[0]-1, 0]
            xy[i, 1] = coordinates[city_tour[0]-1, 1]
    plt.plot(xy[:,0], xy[:,1], marker = 's', alpha = 1, markersize = 7, color = 'black')
    plt.plot(xy[0,0], xy[0,1], marker = 's', alpha = 1, markersize = 7, color = 'red')
    plt.plot(xy[1,0], xy[1,1], marker = 's', alpha = 1, markersize = 7, color = 'orange')
    for i, txt in enumerate(city_tour):
        plt.annotate(txt, (xy[i, 0], xy[i, 1]), fontsize=14, color='red')
    return

##############################################################################################################
# Function: Capacity Check
def Capacity_Check(city_tour_C, Demand_vector):
    Demand_BR = Demand_vector[0, city_tour_C]
    Pay_load_vector = np.cumsum(Demand_BR)
    # print("payload vector: ", Pay_load_vector)
    if (np.sum(np.where(Pay_load_vector < 0)) > 0)  or (np.sum(np.where(Pay_load_vector > 2.0)) > 0):
        # print(Pay_load_vector)
        return False, None
    else:
        # print('True')
        # print(city_tour_C)
        # print("cap check payload vector: ", Pay_load_vector)
        return True, city_tour_C, Pay_load_vector
    
def power(payload, UAV_mass=10, speed=5):
    g = 10
    mot_eff, prop_eff = 0.9, 0.9
    const = speed*g/(mot_eff*prop_eff)
    return (payload+UAV_mass)*const
    
def energy_req(city_order, payload_vector, time_matrix, total_energy=50000):
    # converting into numpy array
    city_order=np.array(city_order)
    payload_vector=np.array(payload_vector)
    time_matrix=np.array(time_matrix)
    
    avl_energy = total_energy
    recharge_rate = total_energy/180              # take 180 sec to fully charge
    recharge_time = np.zeros((1,len(city_order)))
    _lambda,mean,var = 1000, 0, 1                   # considering atmospheric disturbances
    for i in range(len(city_order)-1):
        payload=payload_vector[i]
        time=time_matrix[city_order[i]][city_order[i+1]]
        req_energy = power(payload=payload)*time
        uncertainity = 0  #_lambda*random.gauss(mean, var)
        if avl_energy >= 1.15*req_energy:
            avl_energy -= (req_energy + uncertainity)
        else:
            recharge_time[0, i] = (total_energy-avl_energy)/recharge_rate
            avl_energy = total_energy - (req_energy+uncertainity)
    return recharge_time

def objective_function(Best_Route, T, recharge_time_vector, service_time_vector):
        total_travel_time = 0.0
        for i in range(len(Best_Route)-1):
            total_travel_time += T[Best_Route[i], Best_Route[i+1]]
        print("tt:", total_travel_time)
        service_time_vector = np.append(service_time_vector, 0.0)
        service_time_vector = service_time_vector[Best_Route]
        additional_time = np.sum(np.abs(recharge_time_vector - service_time_vector))
        print("add:", additional_time)
        return  total_travel_time + additional_time

# ######################## Part 1 - Usage ####################################

# # Load File - A Distance Matrix (17 cities,  optimal = 1922.33)
# X = pd.read_csv('Python-MH-Local Search-Variable Neighborhood Search-Dataset-01.txt', sep = '\t') 
# X = X.values

# # Start a Random Seed
# seed = seed_function(X)

# # Call the Function
# lsvns = variable_neighborhood_search(X, city_tour = seed, max_attempts = 75, neighbourhood_size = 7, iterations = 300)

# # Plot Solution. Red Point = Initial city; Orange Point = Second City # The generated coordinates (2D projection) are aproximated, depending on the data, the optimum tour may present crosses
# plot_tour_distance_matrix(X, lsvns)
# plt.savefig("best_route.png")
# output = open('output.txt', 'w')
# print(lsvns, file = output)
# output.close()

######################## Part 2 - Usage ####################################

# Load File - Level2_dataset (Auto generated)
df = pd.read_csv("training_dataset.txt", sep=",")
Y = df[['X coord','Y coord']].to_numpy()
# print(Y.shape)

# Build the Distance Matrix and Time Matrix
XData = build_distance_matrix(Y)
TData = XData/5
Demand_vector = df['Demand(kg)'].to_numpy().reshape(1, 10)
service_time_vector = df['Service_Time(sec)'].to_numpy().reshape(1, 10)
Cust_ID = df[['Cust']].to_numpy()

# Start a Random Seed
# city_tour = [0, 1, 2, 3, 0]
city_tour = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

# Call the Function
# vns = local_search(XData, city_tour, Demand_vector, max_attempts = 5, neighbourhood_size = 3)
# print(vns)

lsvns = variable_neighborhood_search(XData, city_tour, Demand_vector, max_attempts = 10, neighbourhood_size = 5, iterations = 350)
print("lsvns:", lsvns)
# payload_vector = Capacity_Check(lsvns, Demand_vector)[2]
# print("payload vector:", payload_vector)
# recharge_time_vector = energy_req(lsvns, payload_vector, TData, total_energy=50000)
# print("reharge time vector:", recharge_time_vector)
# Time = objective_function(lsvns, TData, recharge_time_vector, service_time_vector)
# print("time: ", Time)

# Plot Solution. Red Point = Initial city; Orange Point = Second City
plt.figure("2")
x_vector = Y[:, 0]
y_vector = Y[:, 1]
plt.scatter(x_vector, y_vector)
for i in range(len(lsvns)-1):
    j = lsvns[i]
    j_ = lsvns[i+1]
    x_list = [x_vector[j], x_vector[j_]]
    y_list = [y_vector[j], y_vector[j_]]
    # print(x_list, y_list)
    plt.plot(x_list, y_list)

plt.show()
# plt.savefig("best_route_New_5_100.png")
# print(lsvns)
# output = open('output_5_100.txt', 'w')
# output.close()