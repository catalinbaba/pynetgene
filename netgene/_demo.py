from netgene.core import Individual
from netgene.operators.crossover import OnePointCrossover
from netgene.ga import Population, GeneticConfiguration, GeneticAlgorithm, GenerationResult
from netgene.operators.mutator import BitFlipMutator
from netgene.chromosome import *
from netgene.operators.selection import *
import math, time

#mutator = BitFlipMutator()
crossover = OnePointCrossover()
selector = TournamentSelector(3)

ga = GeneticConfiguration(crossover_operator=crossover,
                          parent_selector=selector,
                          elitism_size=1,
                          max_generation=5000,
                          n_threads=24
                          ).get_algorithm()

#ga1 = GeneticConfiguration().parent_selector(None).get_algorithm()

population = Population()

population_size = 100
chromosome_size = 7

for _ in range(population_size):
    ind = Individual(FloatChromosome(chromosome_size))
    population.add_individual(ind)

# for individual in population:
#     print(individual)

def fitness_function(individual):
    chromosome = individual.chromosome
    result = 3.4*chromosome.get_gene(0).allele -7.5*chromosome.get_gene(1).allele + 21*chromosome.get_gene(2).allele + \
             1.2*chromosome.get_gene(3).allele -11.3*chromosome.get_gene(4).allele + 2.2*chromosome.get_gene(5).allele - \
             4.7*chromosome.get_gene(6).allele
    if(result == 0):
        fitness_score = float('inf')
    else:
        fitness_score = 1 / math.pow(result, 2)
    individual.fitness = fitness_score
    individual.custom_data = result


def generation_tracker(ga: GeneticAlgorithm, result: GenerationResult):
    print("Step: ", result.generation_number)
    print("best fitness: ", result.best_fitness)
    print("function output: ", result.best_individual.custom_data)
    print("best individual: ", result.best_individual)
    print("evaluation execution: ", result.evaluation_duration)
    print("----------------------------------")

def custom_condition(population):
    custom_data = population.get_best_individual().custom_data
    return -0.01 < custom_data < 0.01


#ga.set_custom_stop_condition(custom_condition)
#ga.set_generation_tracker(generation_tracker)
start_time = time.time()
ga.evolve(population, fitness_function)
end_time = time.time()


fitness = ga.population.get_best_individual().fitness

print("======================================")
print("step: ", ga.population.generation)
print("fitness value: ", fitness)
print("function output: ", ga.population.get_best_individual().custom_data)
print("individual: ", ga.population.get_best_individual())

duration = (end_time - start_time) * 1000  # in milliseconds

print("Execution Time:", duration, "milliseconds")

