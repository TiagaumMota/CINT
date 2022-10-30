import numpy as np
import pandas as pd
import random
import statistics as stats
import matplotlib.pyplot as plt

from deap import base, creator, tools
from pandas.core import indexing

#Read data from file
df_distCentral = pd.read_csv('CustDist_WHCentral.csv',decimal='.', sep=',' )
df_distCorner = pd.read_csv('CustDist_WHCorner.csv',decimal='.', sep=',' )

df_ordFile = pd.read_csv('CustOrd.csv',decimal='.', sep=',' )
df_ord50=df_ordFile.copy()
df_ord50.iloc[1:,1] = 50

df_xyCentral = pd.read_csv('CustXY_WHCentral.csv',decimal='.', sep=',' )
df_xyCorner = pd.read_csv('CustXY_WHCorner.csv',decimal='.', sep=',' )

#Problem considerations
N_customers = 10          #individual size, (10/30/50)
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
    return individual

#Creates an individual randomly
def createIndividual():
        
    customers = [i for i in range(1, N_customers+1)]
    random.shuffle(customers)
    
    return maxOrder(customers)


#-----------------------------------------------Setup the Genetic Algorithm------------------------------------------
#Creating an appropriate type for this minimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

#Initialization 
toolbox = base.Toolbox()

#Permutation representation of individual
#individuals need to be setup to not repeat or skip a city. each city must be represented- toolbox.indices
toolbox.register("individual", tools.initIterate, creator.Individual, createIndividual)

#bag population setup, list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


#Operators 
#Evaluation function
def evaluate(individual):
    start = individual[0]
    summation = distances.iloc[0, start+1]
    
    for i in range(1, len(individual)):
        end = individual[i]
        summation += distances.iloc[start,end+1]
        start = end

    summation += distances.iloc[start,1]
    return summation,

toolbox.register("evaluate", evaluate)

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
    

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=10)

def heuristic():
    split=50
    customers=coord.iloc[1:N_customers+1]

    column_headers = customers.columns.values.tolist()
   
    #Left
    cust_left=customers[customers[column_headers[1]]<split]
    cust_left=cust_left.sort_values(by=column_headers[2])

    #Right
    cust_right=customers[customers[column_headers[1]]>split]
    cust_right=cust_right.sort_values(by=column_headers[2], ascending=False)

    cust_merge=cust_left.append(cust_right)
    route=cust_merge.index.to_list()

    return maxOrder(route)

def main():
    global distances
    global orders 
    global coord
    warehouse = ['Central Location - ', 'Corner Location - ']
    orders_read = ['50 Orders','File Orders']
    
    for position in range(2):
        if position == 0:
            distances = df_distCentral
            coord=df_xyCentral
        else:
            distances = df_distCorner
            coord=df_xyCorner

        for case in range(2):
            if case == 0:
                orders = df_ord50
            else:
                orders = df_ordFile
            
            all_results = np.zeros([Nruns,1])
            best_fitness = 1000000     #fitness value of the best run
            gen_cost = []              #fitness values across generations for best run
            best_path = [] 

            #Candidate solution generated by heuristic
            candidate=heuristic()
           

            for i in range(Nruns):
                random.seed(random.randint(1, 10000))

                #Create initial population, n individuals
                pop = toolbox.population(n=N_pop)

                #Candidate solution generated by heuristic
                pop.append(creator.Individual(candidate))
            
                #------------------Start evolution-----------------------

                # Evaluate the entire population
                fitnesses = list(map(toolbox.evaluate, pop))
                for ind, fit in zip(pop, fitnesses):
                    ind.fitness.values = fit
                
                # Begin the evolution
                gen_cost=np.zeros([NGEN,1])
                
                # Through generations of offsprings 
                for g in range(NGEN):
                    # Select the next generation individuals
                    offspring = toolbox.select(pop, len(pop))
                    
                    # Clone the selected individuals
                    offspring = list(map(toolbox.clone, offspring))

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
            print(warehouse[position] + orders_read[case] )
            print("Mean: ",np.mean(all_results))
            print("STD: ",np.std(all_results))
            print("Best Fitness: ", best_fitness, "run: ", best_fitness_i )
            print("Best path: ", best_path)

            # Cnvergence curve
            x=np.arange(NGEN)
            plt.plot(x,best_gen_cost, label=warehouse[position] + orders_read[case])
           
    plt.title("%d Customers"% N_customers)
    plt.xlabel("Generations")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()

if __name__ == "__main__": 
        main()



