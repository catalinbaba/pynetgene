from pynetgene.chromosome import PermutationChromosome, FloatChromosome
from pynetgene.core import Population, Individual
from pynetgene.ga import GeneticConfiguration, GeneticAlgorithm, GenerationResult
from pynetgene.operators.crossover import Order1Crossover, OnePointCrossover
from pynetgene.operators.mutator import InversionMutator
from pynetgene.operators.selection import TournamentSelector




population = Population()

populationSize = 100
chromosomeSize = 7

for i in range(populationSize):
    ch = FloatChromosome(chromosomeSize)
    individual = Individual(ch)
    population.add_individual(individual)


crossover = OnePointCrossover()
selector = TournamentSelector(3)

ga = GeneticConfiguration(elitism_size=1,
                          max_generation=1000,
                          crossover_operator=crossover,
                          parent_selector=selector,
                          n_threads= 6,
                          ).get_algorithm()

def fitness_function(individual):
    chromosome = individual.chromosome
    x1 = chromosome.get_gene(0).allele
    x2 = chromosome.get_gene(1).allele
    x3 = chromosome.get_gene(2).allele
    x4 = chromosome.get_gene(3).allele
    x5 = chromosome.get_gene(4).allele
    x6 = chromosome.get_gene(5).allele
    x7 = chromosome.get_gene(6).allele
    result = 3.4 * x1 - 7.5 * x2 + 21 * x3 + 1.2 * x4 - 11.3 * x5 + 2.2 * x6 - 4.7 * x7
    fitness_score = 0
    if result == 21:
        fitness_score = float("intf")
    else:
        fitness_score = 1 / (result -21 ) ** 2
    individual.fitness = fitness_score
    individual.custom_data = result

def tracker(g: GeneticAlgorithm, r: GenerationResult):
    chromosome = r.best_individual.chromosome
    x1 = chromosome.get_gene(0).allele
    x2 = chromosome.get_gene(1).allele
    x3 = chromosome.get_gene(2).allele
    x4 = chromosome.get_gene(3).allele
    x5 = chromosome.get_gene(4).allele
    x6 = chromosome.get_gene(5).allele
    x7 = chromosome.get_gene(6).allele
    result = r.best_individual.custom_data
    print("x1: ", x1)
    print("x2: ", x2)
    print("x3: ", x3)
    print("x4: ", x4)
    print("x5: ", x5)
    print("x6: ", x6)
    print("x7: ", x7)
    print("Step: ", r.generation_number)
    print("Best fitness: ", r.best_individual)
    print("function output: ", result)
    print("best individual: ", r.best_individual)
    print("evaluation duration: ", r.evaluation_duration)
    print("evolution duration: ", r.evolution_duration)
    print("-------------------------------")

def custom_stop(p):
    best_individual = p.get_best_individual()
    result = best_individual.custom_data
    if 20.999 < result < 21.001:
        return True
    else:
        return False

ga.set_custom_stop_condition(custom_stop)
#ga.set_generation_tracker(tracker)
ga.evolve(population, fitness_function)

custom_data = ga.population.get_best_individual().custom_data
print("---------------FINAL RESULT---------------")
print("Step: ", ga.population.generation)
print("Best fitness: ", ga.population.get_best_individual().fitness)
print("Function output: ", custom_data)



