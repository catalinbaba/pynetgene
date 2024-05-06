from pynetgene.chromosome import BitChromosome
from pynetgene.core import Population, Individual
from pynetgene.ga import GeneticConfiguration, GenerationResult, GeneticAlgorithm
from pynetgene.operators.mutator import BitFlipMutator
from pynetgene.tests.test_genetic_algorighm import lesson1_fitness

import pynetgene.ga


def lesson1_fitness(individual):
    fitness = 0
    chromosome = individual.chromosome
    for i in range(len(chromosome)):
        gene = chromosome.get_gene(i)
        if gene == True:
            fitness += 1
    individual.fitness = fitness

def tracker(g: GeneticAlgorithm, r: GenerationResult):
    print("Step: ", r.generation_number)
    print("Best fitness: ", r.best_fitness)
    print("Evaluation duration: ", r.evaluation_duration)
    print("Evolution duration: ", r.evolution_duration)
    print("---------------------------------------------")

mutator = BitFlipMutator()

ga = GeneticConfiguration(mutator_operator=mutator,
                          elitism_size=1,
                          max_generation=100,
                          target_fitness=5.0,
                          ).get_algorithm()
population = Population()
populationSize = 10
chromosomeSize =5

for i in range(populationSize):
    bitChromosome = BitChromosome(chromosomeSize)
    individual = Individual(bitChromosome)
    population.add_individual(individual)

ga.set_generation_tracker(tracker)
ga.evolve(population, lesson1_fitness)

individual = ga.population.get_best_individual()
fitness_score = individual.fitness

print("Individual: ", individual)
print("fitness score: ", fitness_score)
