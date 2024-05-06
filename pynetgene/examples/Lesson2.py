from pynetgene.chromosome import PermutationChromosome
from pynetgene.core import Population, Individual
from pynetgene.ga import GeneticConfiguration, GenerationResult, GeneticAlgorithm
from pynetgene.operators.crossover import Order1Crossover
from pynetgene.operators.mutator import InversionMutator
from pynetgene.operators.selection import TournamentSelector

distances = [
    [0, 3, 7, 1, 3, 5],
    [3, 0, 8, 5, 1, 2],
    [7, 8, 0, 4, 3, 8],
    [1, 5, 4, 0, 6, 7],
    [3, 1, 3, 6, 0, 1],
    [5, 2, 8, 7, 1, 0]
]

population = Population()

populationSize = 50
chromosomeSize = 5

for i in range(populationSize):
    ch = PermutationChromosome(chromosomeSize, 1)
    individual = Individual(ch)
    population.add_individual(individual)

mutator = InversionMutator()
crossover = Order1Crossover()
selector = TournamentSelector(3)

ga = GeneticConfiguration(elitism_size=1,
                          max_generation=100,
                          mutator_operator=mutator,
                          crossover_operator=crossover,
                          parent_selector=selector).get_algorithm()

def fitness(individual):
    chromosome = individual.chromosome
    totalDistance = 0
    dStart = distances[0][chromosome.get_gene(0)]
    for i in range(len(chromosome) -1):
        totalDistance = totalDistance + distances[chromosome.get_gene(i)][chromosome.get_gene(i+1)]
    dEnd = distances[chromosome.get_gene(len(chromosome)-1)][0]
    totalDistance = dStart + dEnd + totalDistance
    fitness_score = 1/totalDistance * 1000
    individual.fitness = fitness_score


def fitness_pythonic(individual):
    chromosome = individual.chromosome
    alleles = chromosome.genes # Extract alleles once using list comprehension

    # Calculate path distance using a generator inside the sum function
    total_distance = sum(distances[alleles[i]][alleles[i + 1]] for i in range(len(alleles) - 1))
    total_distance += distances[0][alleles[0]] + distances[alleles[-1]][0]  # Add start and end distances

    # Calculate fitness score
    fitness_score = 1/total_distance * 1000
    individual.fitness = fitness_score


def calculate_distance(chromosome):
    totalDistance = 0
    dStart = distances[0][chromosome.get_gene(0)]
    for i in range(len(chromosome) - 1):
        totalDistance = totalDistance + distances[chromosome.get_gene(i)][chromosome.get_gene(i + 1)]
    dEnd = distances[chromosome.get_gene(len(chromosome) - 1)][0]
    totalDistance = dStart + dEnd + totalDistance
    return totalDistance

def tracker(g: GeneticAlgorithm, r: GenerationResult):
    print("Step: ", r.generation_number)
    print("Best fitness: ", r.best_fitness)
    print("Evaluation duration: ", r.evaluation_duration)
    print("Evolution duration: ", r.evolution_duration)
    print("---------------------------------------------")

def stop(p: Population):
    bestIndividual = p.get_best_individual()
    chromosome = bestIndividual.chromosome
    distance = calculate_distance(chromosome)
    if distance < 19:
        return True
    else:
        return False

ga.set_generation_tracker(tracker)
ga.evolve(population, fitness)

print("==============================================")
print("Fitness value: ", ga.population.get_best_individual().fitness)
print("Generation: ", ga.population.generation)
bestChromosome = ga.population.get_best_individual().chromosome
best_route = "0->"
for gene in bestChromosome:
    best_route += str(gene) + "->"
best_route += "0"
print("Best route:", best_route)
print("Total distance:", calculate_distance(bestChromosome))

