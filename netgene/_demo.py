from netgene.core import Individual
from netgene.operators.crossover import OnePointCrossover
from netgene.ga import Population, GeneticConfiguration, GeneticAlgorithm, GenerationResult
from netgene.operators.mutator import BitFlipMutator
from netgene.chromosome import *

mutator = BitFlipMutator()
crossover = OnePointCrossover()

ga = GeneticConfiguration(mutation_rate = 0.5,
                          mutator_operator=mutator,
                          crossover_operator=crossover,
                          elitism_size=1,
                          max_generation=100,
                          target_fitness= 5.0).get_algorithm()

#ga1 = GeneticConfiguration().parent_selector(None).get_algorithm()

population = Population()

population_size = 3
chromosome_size = 5

for _ in range(population_size):
    ind = Individual(BitChromosome(chromosome_size))
    population.add_individual(ind)

# for individual in population:
#     print(individual)

def fitness_function(individual):
    chromosome = individual.chromosome
    fitness = 0
    for gene in chromosome:
        if gene.allele:
            fitness += 1
    individual.fitness = fitness
    #print("individual fitness: ", individual.fitness)


def generation_tracker(ga: GeneticAlgorithm, result: GenerationResult):
    print("Step: ", result.generation_number)
    print("best fitness: ", result.best_fitness)
    print("best individual: ", result.best_individual)
    print("evaluation execution: ", result.evaluation_duration)
    print("----------------------------------")

def custom_condition(population):
    for individual in population:
        if individual.fitness == 4:
            return True
        else:
            return False


#ga.set_custom_stop_condition(custom_condition)
ga.set_generation_tracker(generation_tracker)

ga.evolve(population, fitness_function)

fitness = ga.population.get_best_individual().fitness

print("======================================")
print("step: ", ga.population.generation)
print("fitness value: ", fitness)
print("individual: ", ga.population.get_best_individual())

