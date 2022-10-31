from turtle import distance
import numpy as np
import pandas as pd
import random
import statistics as stats
import matplotlib.pyplot as plt

from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import base, creator, tools


#Read data from file
df_distCentral = pd.read_csv('CustDist_WHCentral.csv',decimal='.', sep=',' )
df_ordFile = pd.read_csv('CustOrd.csv',decimal='.', sep=',' )


#Problem considerations
N_customers = 20         #individual size, (10/30/50)
Nruns = 30                #max evaluations=10000
NGEN = 100                #number of offsprings generation
CXPB = 0.5                #crossover probability
MUTPB = 0.2               #mutation probability
N_pop = 100               #number of individuals


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
      
    #Cost to travel between those cities
    N_orders = 0 #Number of current orders
    N_warehouse_visits = 0 #always starts at the warehouse
    distance_cost = [] #Distance between cities
    cargo_totals = [] #Total cargo between visits to the warehouse
    cargo_cost = [] #Cargo in truck while visiting which city

    for i in range(cust):
        #Distances between cities
        
        if i == 0: #If it's the first city, it is always distance from warehouse
            distance_cost.append(df_distCentral.iloc[0,individual[i]+1]) #because of how columns are organized
        else: #else it's the distance between the city we are visiting and last city visited
            distance_cost.append(df_distCentral.iloc[individual[i],individual[i]]) #not individual -1 because of how columns are organized
        
        if individual[i] == 0: #if the truck is in the warehouse
            cargo_totals.append(N_orders) #Total cargo in the truck for the cities tour is saved
            N_orders = 0
        else:
            N_orders += orders.iloc[individual[i],1] #Ammount of cargo for each city in a tour
    #print("individuals:", individual, "\ncost:", distance_cost)
    if N_orders != 0: #In case the last city visted isnt the warehouse, save the total cargo for last tour
        cargo_totals.append(N_orders)

    # Now for the cargo cost
    for i in range(cust):
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
    
    
    """print("\nIndividuals:", individual)
    print("\nDistance_Cost:", distance_cost)
    print("\nCargo_Cost", cargo_cost)
    print("\nTOTAL:", Final_Cost)"""

    container = [list(a) for a in zip(individual, Final_Cost)]
    #return individual
    return container

#Creates an individual randomly
def createIndividual():
        
    customers = [i for i in range(1, N_customers+1)]
    random.shuffle(customers)


    
    return maxOrder(customers)


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
    summation = 0
    for i in range(1, len(individual)):
        summation = summation + individual[i][1]
    return summation

#computes the fitness list of an individual
def evaluation(individual):
    result = []
    result.append(evalCost(individual))
    result.append(evalDist(individual))
    return result

def preSearch(ind1, ind2):

    for idx in range(N_customers):
        if ind1[idx] == 0:
            ind1.pop(idx)
        ind1[idx] = ind1[idx]-1
        if ind2 != None:
            if ind2[idx] == 0:
                ind2.pop(idx)
            ind2[idx] = ind2[idx]-1
    
    if ind2 != None:
        return ind1, ind2
    else:
        return ind1

def afterSearch(ind1, ind2):
    NOrders1 = 0
    NOrders2 = 0
    cust = N_customers
    idx = 0

    while idx < cust:

        ind1[idx] = ind1[idx]+1
        NOrders1 += orders.iloc[ind1[idx],1]
        if NOrders1 > 1000:
            ind1.insert(idx,0)
            ind1[idx+1] = ind1[idx+1]-1
            cust += 1
            NOrders1 = 0

        if ind2 != None:
            ind2[idx] = ind2[idx]+1
            NOrders2 += orders.iloc[ind2[idx],1]
            if NOrders2 > 1000:
                ind2.insert(idx,0)
                ind2[idx+1] = ind2[idx+1]-1
                NOrders2 = 0
            cust = max(len(ind1), len(ind2))
        idx += 1
    
    if ind2 != None:
        return ind1, ind2
    else:
        return ind1

#custom crossover that crosses the first elements among each other 
#and the second elements among each other
def customCX(individual1, individual2):
    ind1 = [a for a,b in individual1]
    ind2 = [a for a,b in individual2]
    ind1_ = [b for a,b in individual1]
    ind2_ = [b for a,b in individual2]

    new_ind1, new_ind2 = tools.cxOrdered(ind1,ind2)
    new_ind1_, new_ind2_ = tools.cxTwoPoint(ind1_,ind2_)

    for i in range(len(individual1)):
        individual1[i][0] = new_ind1[i]
        individual2[i][0] = new_ind2[i]
        individual1[i][1] = new_ind1_[i]
        individual2[i][1] = new_ind2_[i]

    return individual1, individual2
    
toolbox.register("evaluate", evaluation)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

def main():
    
    global distances
    global orders 
    global coord
    warehouse = ['Central Location']
    orders_read = ['File Orders']
    
    #Info from the datasets
    distances = df_distCentral
    orders = df_ordFile

    all_results = np.zeros([Nruns,1])
    best_fitness = 1000000     #fitness value of the best run
    gen_cost = []              #fitness values across generations for best run
    best_path = []

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean, axis=0)
    # stats.register("std", numpy.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "min", "max"


    for i in range(Nruns):
        random.seed(random.randint(1, 10000))

        #Create initial population, n individuals
        pop = toolbox.population(n=N_pop)

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
        print(logbook.stream)
    
        # Begin the evolution
        gen_cost=np.zeros([NGEN,1])
        
        # Through generations of offsprings 
        for g in range(1, NGEN):
            # Select the next generation individuals
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
            
                if random.random() < CXPB:
                    child1,child2 = preSearch(child1,child2)
                    toolbox.mate(child1, child2)
                    
                    del child1.fitness.values
                    del child2.fitness.values

                    child1, child2 = afterSearch(child1, child2)
                    
            for mutant in offspring:
                if random.random() < MUTPB:
                    mutant = preSearch(mutant, None )
                    
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
            # The population is entirely replaced by the offspring - selected parents 
            pop[:] = offspring
                
            elite = tools.selBest(pop, 1)[0]
                
            # best fitness value of each generation
            gen_cost[g]=elite.fitness.values[0]

        #------------------------------ End of (successful) evolution -----------------------------------
        
        elite = tools.selBest(pop, 1)[0]
            
        #Saving the best solution
        if best_fitness > elite.fitness.values[0]:
            best_gen_cost = gen_cost
            best_fitness = elite.fitness.values[0]
            best_fitness_i=i
            best_path = list(elite)
    
        all_results[i]=elite.fitness.values[0]
    
    print("=============================================")
    print(warehouse + orders_read )
    print("Mean: ",np.mean(all_results))
    print("STD: ",np.std(all_results))
    print("Best Fitness: ", best_fitness, "run: ", best_fitness_i )
    print("Best path: ", best_path)

    # Cnvergence curve
    x=np.arange(NGEN)
    plt.plot(x,best_gen_cost, label=warehouse + orders_read)
           
    plt.title("%d Customers"% N_customers)
    plt.xlabel("Generations")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()


if __name__ == "__main__": 
        main()
