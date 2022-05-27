import numpy as np

def init_pop(pop_size):
    return np.random.randint(8,size=(pop_size,8))
    

#initial_population = init_pop(4)
#print("initial population is = ")
#print(initial_population)

def fitness_func(population):
    fitness_vals=[]
    for x in population:
        penalty= 0
        for i in range(8):
            r=x[i]
            for j in range(8):
                if i==j :
                    continue
                d=abs(i-j)
                if x[j] in [r,r-d,r+d]:
                    penalty +=1
        fitness_vals.append(penalty)
        
    return -1 * np.array(fitness_vals)

#fitness_values=fitness_func(initial_population)

#print("the fitness values is = ")
#print(fitness_values)


def selection_func(population, fitness_vals):
    probs = fitness_vals.copy()
    probs += abs(probs.min()) + 1
    probs = probs/probs.sum()
    n = len(population)
    indices = np.arange(n)
    selected_indices = np.random.choice(indices,size=n,p=probs)
    selected_population = population[selected_indices]
    return selected_population
    
#selected_population= selection_func(initial_population,fitness_values)
#print("the new selected population = ")
#print(selected_population)

def crossover_func(parent1,parent2,PC):
    # return value btween 0 and 1
    r= np.random.random() 
    if r < PC:
        m = np.random.randint(1,8)
        child1 = np.concatenate([parent1[:m], parent2[m:]])
        child2 = np.concatenate([parent2[:m], parent1[m:]])
    else:
        child1=parent1.copy()
        child2=parent2.copy()
    return child1,child2

#p1=selected_population[0]
#p2=selected_population[1]
#ch1,ch2 = crossover_func(p1,p2,PC=0.70)
#print(p1,' >>> ',ch1)
#print(p2,' >>> ',ch2)


def mutation_func(individual,pm):
    r= np.random.random()
    if r < pm:
        m=np.random.randint(8)
        individual[m] = np.random.randint(8)
    return individual

def crossover_mutation_func(selected_pop,PC,pm):
    n = len(selected_pop)
    new_pop = np.empty((n,8),dtype=int)
    for i in range(0,n,2):
        parent1 = selected_pop[i]
        parent2 = selected_pop[i+1]
        child1 , child2 = crossover_func(parent1 ,parent2 ,PC)
        new_pop[i] = child1
        new_pop[i+1] = child2
    for i in range(n):
        mutation_func(new_pop[i],pm)
    return new_pop

def eight_queens(pop_size, max_generations,PC , pm):
    population = init_pop(pop_size)
    best_fitness_overall= None
    for i in range(max_generations):
        fitness_vals = fitness_func(population)
        best_i = fitness_vals.argmax()
        best_fitness = fitness_vals[best_i]
        if best_fitness_overall is None or best_fitness > best_fitness_overall:
            best_fitness_overall = best_fitness
            best_solution = population[best_i]
        print(f'i_gen = {i:06} -f={-best_fitness_overall:03}')
        if best_fitness == 0 :
            print('\nOptimal solution is found')
            break
        selected_pop = selection_func(population,fitness_vals)
        population = crossover_mutation_func(selected_pop , PC , pm)
    print()
    print(best_solution)
            
eight_queens (pop_size=1000,max_generations=1000,PC=0.7,pm=0.3)


