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

df_ord = pd.read_csv('CustOrd.csv',decimal='.', sep=',' )

df_xyCentral = pd.read_csv('CustXY_WHCentral.csv',decimal='.', sep=',' )
df_xyCorner = pd.read_csv('CustXY_WHCorner.csv',decimal='.', sep=',' )

#Problem considerations
N_customers = 10           #individual size, (10/30/50)
distances = df_distCentral #(df_distCentral/df_distCorner)
orders = df_ord            #/orders.iloc[1:,1]=50
EVAL_NUM = 30              #max evaluations=10000
NGEN = 40                  #number of offsprings generation
CXPB = 0.5                 #crossover probability
MUTPB = 0.2                #mutation probability
N_pop = 100              #number of individuals

all_results = np.zeros([EVAL_NUM,1])
best_fit_i = -1            #index of the best run
best_fitness = 1000000     #fitness value of the best run
gen_cost = []              #fitness values across generations for best run
best_path = [] 


#Truck only transports a maximum of 1000 products at a time
def maxOrder(individual):
    NOrders=0
    for i in range(N_customers):
        NOrders += orders.iloc[individual[i],1]
        if NOrders > 1000:
            individual.insert(i,0)
            individual.pop()
            NOrders=0

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
toolbox.register("indices", random.sample, range(N_customers), N_customers)
toolbox.register("individual", tools.initIterate, creator.Individual, createIndividual)

#bag population setup, list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


#Operators 
#Evaluation function
def evaluate(individual):
    start = individual[0]
    summation = distances.iloc[0,start]
    
    for i in range(1, len(individual)):
        end = individual[i]
        summation += distances.iloc[start,end]
        start = end

    summation += distances.iloc[start,0]
    return summation,


toolbox.register("evaluate", evaluate)

def cxOrdered(ind1, ind2):
    """Executes an ordered crossover (OX) on the input
    individuals. The two individuals are modified in place. This crossover
    expects :term:`sequence` individuals of indices, the result for any other
    type of individuals is unpredictable.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    Moreover, this crossover generates holes in the input
    individuals. A hole is created when an attribute of an individual is
    between the two crossover points of the other individual. Then it rotates
    the element so that all holes are between the crossover points and fills
    them with the removed elements in order. For more details see
    [Goldberg1989]_.
    This function uses the :func:`~random.sample` function from the python base
    :mod:`random` module.
    .. [Goldberg1989] Goldberg. Genetic algorithms in search,
       optimization and machine learning. Addison Wesley, 1989
    """
    size = min(len(ind1), len(ind2))
    
    a, b = random.sample(range(size), 2)
   
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
   
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]-1] = False
            holes2[ind1[i]-1] = False

  
    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1, ind2
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]-1]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]-1]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2

toolbox.register("mate", cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=10)





for i in range(EVAL_NUM):
    def main():
        random.seed(random.randint(1, 1000))

        #Create initial population, n individuals
        pop= toolbox.population(n=N_pop)

        print("Start evolution")

        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(pop))

        # Extracting all the fitnesses of the population
        fits = [ind.fitness.values[0] for ind in pop]

        # Begin the evolution
        gen_cost=np.zeros([NGEN,1])
        
        # Through generations of offsprings 
        for g in range(NGEN):
            print("-- Generation %i --" % g)
            # Select the next generation individuals
            offspring = toolbox.select(pop, len(pop))

            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))
        
        
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Since the content of some of our offspring changed during the last step, we now need to re-evaluate their fitnesses.
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            print("  Evaluated %i individuals" % len(invalid_ind))
            # The population is entirely replaced by the offspring - selected parents 
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
                
            
            mean = stats.mean(fits)
            std = stats.stdev(fits)
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            elite = tools.selBest(pop, 1)[0]
            print("Best individual is %s, %s" % (elite, elite.fitness.values))
                
            # best fitness value of each generation
            gen_cost[g-1]=elite.fitness.values[0]

        print("-- End of (successful) evolution --")
        
        elite = tools.selBest(pop, 1)[0]
            
        print("Best individual is %s, %s" % (elite, elite.fitness.values))
            
            
        global best_fitness
        global best_fitness_i
        global best_gen_cost
        global best_path
            
         #if this run produces the best solution, than save its values
        if best_fitness > elite.fitness.values[0]:
            best_gen_cost = gen_cost
            best_fitness = elite.fitness.values[0]
            best_fitness_i=i
            best_path = list(elite)
    
        all_results[i]=elite.fitness.values[0]

    

    if __name__ == "__main__": 
        main()


print("=============================================")
print("Mean: ",np.mean(all_results))
print("STD: ",np.std(all_results))
print("Best Fitness: ", best_fitness, "run: ", best_fitness_i )
print("Best path: ", best_path)

# =plot convergence curve======================================================
x=np.arange(NGEN)
plt.plot(x,best_gen_cost)
plt.title("run:%d "% best_fitness_i)
plt.xlabel("Generations")
plt.ylabel("Distance")
plt.show()
