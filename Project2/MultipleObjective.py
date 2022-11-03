from turtle import distance
import numpy as np
import pandas as pd
import random
import statistics as stats
import matplotlib.pyplot as plt

import multiprocessing
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import base, creator, tools


#Read data from file
df_distCentral = pd.read_csv('CustDist_WHCentral.csv',decimal='.', sep=',' )
df_ordFile = pd.read_csv('CustOrd.csv',decimal='.', sep=',' )

distances = df_distCentral 
orders = df_ordFile


#Problem considerations
N_customers = 50       #individual size, (10/30/50)
NGEN = 100                #number of offsprings generation
CXPB = 0.5                #crossover probability
MUTPB = 0.2               #mutation probability
N_pop = 100               #number of individuals

#NGEN * N_pop = 10 000


#Truck only transports a maximum of 1000 products at a time
def maxOrder(individual):
    NOrders = 0
    cust = N_customers
    i = 0
    
    while i < cust:
        NOrders += orders.iloc[individual[i],1]
        if NOrders > 1000:
            individual.insert(i,0)
            cust += 1
            NOrders = 0
        i += 1
 
    return individual

#Computes cost for new individuals
def MultiCost(individual):
    #Cost to travel between those cities
    N_orders = 0 #Number of current orders
    N_warehouse_visits = 0 #always starts at the warehouse
    distance_cost = [] #Distance between cities
    cargo_totals = [] #Total cargo between visits to the warehouse
    cargo_cost = [] #Cargo in truck while visiting which city
    cust=N_customers

    for i in range(len(individual)):
        
        if i == 0: #If it's the first city, it is always distance from warehouse
            distance_cost.append(df_distCentral.iloc[0,individual[i]+1]) #because of how columns are organized
        else: #else it's the distance between the city we are visiting and last city visited
            distance_cost.append(df_distCentral.iloc[individual[i],individual[i]]) #not individual -1 because of how columns are organized

        if individual[i] == 0: #if the truck is in the warehouse
            
            cargo_totals.append(N_orders) #Total cargo in the truck for the cities tour is saved
            N_orders = 0
            cust = cust + 1
            
        else:
            N_orders += orders.iloc[individual[i],1] #Ammount of cargo for each city in a tour
        
    #print("individuals:", individual, "\ncost:", distance_cost)
    if N_orders != 0: #In case the last city visted isnt the warehouse, save the total cargo for last tour
        cargo_totals.append(N_orders)

    # Now for the cargo cost
    for i in range(len(individual)):
        if individual[i] == 0: 
            N_warehouse_visits +=1 #if in the warehouse, change to a different tour total in list in cargo_totals
            cargo_cost.append(0) #Truck empty when heading to warehouse
            flag = 1 #Flag to know when the warehouse is visited
        else:
            if i == 0 or flag == 1: #if coming from the warehouse
                total_to_append = cargo_totals[N_warehouse_visits] #truck is fully loaded (total)
                flag = 0 
                
            else: #if coming from another city
                total_to_append = total_to_append - orders.iloc[individual[i-1],1] #add the remaining on the truck

            cargo_cost.append(total_to_append)

    Final_Cost = []
    for i1, i2 in zip(cargo_cost, distance_cost):
        Final_Cost.append(i1*i2)     
   
    return Final_Cost

#Creates an individual randomly
def createIndividual():
        
    customers = [i for i in range(1, N_customers+1)]
    random.shuffle(customers)

    customers = maxOrder(customers)
    cost = MultiCost(customers)

    container = [list(a) for a in zip(customers, cost)]

    return container

#-----------------------------------------------Setup the Genetic Algorithm------------------------------------------
#Creating an appropriate type for this minimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

#Initialization 
toolbox = base.Toolbox()

#Permutation representation of individual
#individuals need to be setup to not repeat or skip a city. each city must be represented- toolbox.indices
toolbox.register("individual", tools.initIterate, creator.Individual, createIndividual)

#bag population setup, list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#Operators 
#Evaluation functions
def evalDist(individual):
    start = individual[0][0] #First city to visit    
    summation = distances.iloc[0, start+1] #sums distance from warehouse to 1st city 
    
    for i in range(1, len(individual)): #for all cities to visit
        end = individual[i][0] #destination city
        summation += distances.iloc[start,end+1] #distance from current city to next city 
        start = end

    summation += distances.iloc[start,1]
    
    return summation

def evalCost(individual):
 #Cost to travel between those cities
    N_orders = 0 #Number of current orders
    N_warehouse_visits = 0 #always starts at the warehouse
    distance_cost = [] #Distance between cities
    cargo_totals = [] #Total cargo between visits to the warehouse
    cargo_cost = [] #Cargo in truck while visiting which city
    cust=N_customers

    for i in range(len(individual)):
        
        if i == 0: #If it's the first city, it is always distance from warehouse
            distance_cost.append(df_distCentral.iloc[0,individual[i][0]+1]) #because of how columns are organized
        else: #else it's the distance between the city we are visiting and last city visited
            distance_cost.append(df_distCentral.iloc[individual[i][0],individual[i][0]]) #not individual -1 because of how columns are organized

        if individual[i][0] == 0: #if the truck is in the warehouse
            
            cargo_totals.append(N_orders) #Total cargo in the truck for the cities tour is saved
            N_orders = 0
            cust = cust + 1
            
        else:
            N_orders += orders.iloc[individual[i][0],1] #Ammount of cargo for each city in a tour
        
    if N_orders != 0: #In case the last city visted isnt the warehouse, save the total cargo for last tour
        cargo_totals.append(N_orders)

    # Now for the cargo cost
    for i in range(len(individual)):
        if individual[i][0] == 0: 
            N_warehouse_visits +=1 #if in the warehouse, change to a different tour total in list in cargo_totals
            cargo_cost.append(0) #Truck empty when heading to warehouse
            flag = 1 #Flag to know when the warehouse is visited
        else:
            if i == 0 or flag == 1: #if coming from the warehouse
                total_to_append = cargo_totals[N_warehouse_visits] #truck is fully loaded (total)
                flag = 0 
                
            else: #if coming from another city
                total_to_append = total_to_append - orders.iloc[individual[i-1][0],1] #add the remaining on the truck

            cargo_cost.append(total_to_append)

    Final_Cost = []
    for i1, i2 in zip(cargo_cost, distance_cost):
        Final_Cost.append(i1*i2)     

    summation = 0
    for i in range(1, len(individual)):
        summation = summation + Final_Cost[i]
    return summation

#computes the fitness list of an individual
def evaluation(individual):
    result = []
    result.append(evalDist(individual))
    result.append(evalCost(individual))
    return result

def customMutation(individual): #verified

    ind = [a for a,b in individual]
    ind_ = [b for a,b in individual]

    for idx in range(N_customers):
        
        if ind[idx] == 0: #if warehouse in position indx

            ind.pop(idx) #remove
            ind_.pop(idx)  
        
        ind[idx] = ind[idx]-1 #go back one because of index for tools
   
    new_ind = tools.mutShuffleIndexes(ind, 0.05)[0]

    for idx in range(N_customers):
        new_ind[idx] = new_ind[idx]+1
    
    individual_new = [list(a) for a in zip(new_ind, ind_)]

    return individual_new

def customCX(individual1, individual2): #verified
    
    ind1 = [a for a,b in individual1]
    ind2 = [a for a,b in individual2]
    ind1_ = [b for a,b in individual1]
    ind2_ = [b for a,b in individual2]
    
    for idx in range(N_customers):
        if ind1[idx] == 0:
            ind1.pop(idx)
            ind1_.pop(idx)

        if ind2[idx] == 0:
            ind2.pop(idx)
            ind2_.pop(idx)
                
    for idx in range(N_customers):
        ind1[idx] = ind1[idx]-1
        ind2[idx] = ind2[idx]-1

    new_ind1, new_ind2 = tools.cxOrdered(ind1,ind2)
    
    for idx in range(N_customers):
        new_ind1[idx] = new_ind1[idx]+1
        new_ind2[idx] = new_ind2[idx]+1   

    individual1_new = [list(a) for a in zip(new_ind1, ind1_)]
    individual2_new = [list(a) for a in zip(new_ind2, ind2_)]

    return individual1_new, individual2_new

def afterSearch(individual1, individual2):
    
    NOrders1 = 0
    NOrders2 = 0
    cust = N_customers
    idx = 0

    ind1 = [a for a,b in individual1]
    ind1_ = [b for a,b in individual1]
    
    if individual2 != None:
        ind2 = [a for a,b in individual2]
        ind2_ = [b for a,b in individual2]

    #print("\nindividual:", individual1)

    while idx < cust:

        NOrders1 += orders.iloc[ind1[idx],1]
        
        if NOrders1 > 1000:

            ind1.insert(idx,0)
            ind1_.insert(idx,0)
            cust += 1
            NOrders1 = 0

        if individual2 != None:

            NOrders2 += orders.iloc[ind2[idx],1]
            if NOrders2 > 1000:
                ind2.insert(idx,0)
                ind2_.insert(idx,0)
                NOrders2 = 0
            cust = max(len(ind1), len(ind2))
        idx += 1
    #print("\nindividual: after", individual1)
    if individual2 != None:
        #print("\nind:", len(ind1))
        #print("\nind:", ind1)
        #print("\nind:", ind1_)
        #print("\nindividual:", len(individual1))
        #print("\nind:", individual1)

   
        ind1_new = [list(a) for a in zip(ind1, ind1_)]
        ind2_new = [list(a) for a in zip(ind2, ind2_)]
        #print("\n NEW:", ind1_new)
        return ind1_new, ind2_new
    else:
        ind1_new = [list(a) for a in zip(ind1, ind1_)]
        return ind1_new

#computes the reference point as if the path of salesman was always the worst possible
def refPoint(pop):
    y1 = distances.iloc[:N_customers,:N_customers].to_numpy().max() * N_customers
    #x1 = costCar.iloc[:N_customers,:N_customers].to_numpy().max() * N_customers
    max_value = max(sublist[1] for sublist in pop)    
    x = max_value[1] * N_customers
    y = y1

    x = float(x)
    y = float(y)
    print('XY = ' + str(x) + ' ' + str(y))
    return (x,y) 



toolbox.register("evaluate", evaluation)
toolbox.register("mate", customCX)
toolbox.register("mutate", customMutation)
toolbox.register("select", tools.selNSGA2)

def main():
    
    #For results:
    hypervols = []
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max"

    pareto = tools.ParetoFront()

    random.seed(random.randint(1, 10000))

    #Create initial population, n individuals
    pop = toolbox.population(n=N_pop)
    
    ref = refPoint(pop)

    #------------------Start evolution-----------------------
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    #so far so good

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    #print(logbook.stream)

    # Begin the evolution
    
    # Through generations of offsprings 
    for g in range(1, NGEN):
        # Select the next generation individuals
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CXPB:
                #print("\nchild1:",child1)
                #print("\nchild2:",child1)
                toolbox.mate(child1, child2)
                #print("\nchild1: cx",child1)
                #print("\nchild2: cx",child1)
                
                del child1.fitness.values
                del child2.fitness.values

                child1, child2 = afterSearch(child1, child2)
                #print("\nchild1 after:",child1)
                #print("\nchild2 after:",child1)
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
                mutant = afterSearch(mutant, None )
        
            
        # Since the content of some of our offspring changed during the last step,
        #  we now need to re-evaluate their fitnesses.
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        pop = toolbox.select(pop + offspring, N_pop)
        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        
        pareto.update(pop)
        hypervols.append(hypervolume(pareto, ref))
            
        
            
    # best fitness value of each generation
    return pop, pareto, hypervols

    #------------------------------ End of (successful) evolution -----------------------------------
    
   

if __name__ == "__main__":
    #   Multiprocessing pool
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    pop, optimal_front, hypervols = main()
        
    for ind in optimal_front:
        print()
        print(ind)
        print('Dist = ' + str(evalDist(ind)))
        print('Cost = ' + str(evalCost(ind)))
        print()

    #%% Plot the Pareto Front
    x, y = zip(*[ind.fitness.values for ind in optimal_front])

    fig = plt.figure()
    plt.scatter(x, y, c='r', marker='x')
    plt.xlabel('Cost')
    plt.grid(True)
    plt.ylabel('Distance')
    plt.show()

    #%% Plot the hypervolume
    fig2 = plt.figure()
    x = [i for i in range(1, NGEN)]
    y = hypervols
    plt.scatter(x, y, c='r')
    plt.ylabel('Hypervolume')
    plt.grid(True)
    plt.xlabel('Nr gens')
    plt.show()

    #%% Print the coordinates of the pareto front's points
    for ind in optimal_front:
        print()
        print('Dist = ' + str(evalDist(ind)))
        print('Cost = ' + str(evalCost(ind)))
        print()
    #%%