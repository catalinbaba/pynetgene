import pytest

from pynetgene.chromosome import BitChromosome, PermutationChromosome, IntegerChromosome, FloatChromosome
from pynetgene.core import Population, Individual
from pynetgene.operators.crossover import OnePointCrossover, Order1Crossover, TwoPointCrossover
from pynetgene.ga import GeneticAlgorithm, GenerationResult, GeneticConfiguration
from pynetgene.operators.selection import RouletteSelector, TournamentSelector, RankSelector, CompetitionSelector
from pynetgene.operators.mutator import GaussianMutator, BitFlipMutator, InversionMutator, IntegerMutator
import time


@pytest.fixture
def setup_genetic_algorithm():
    selector = RouletteSelector()
    crossover = OnePointCrossover()
    mutator = GaussianMutator()
    ga = GeneticAlgorithm(
        parent_selector=selector,
        crossover_operator=crossover,
        mutator_operator=mutator,
        crossover_rate=0.7,
        mutation_rate=0.01,
        elitism=True,
        elitism_size=1,
        max_generation=100,
        target_fitness=100.0,
        skip_crossover=False,
        skip_mutation=False,
        n_threads=4,
        clock=time.time,
        printer=None
    )
    return ga


def test_initialization(setup_genetic_algorithm):
    ga = setup_genetic_algorithm
    assert ga._mutation_rate == 0.01
    assert ga._crossover_rate == 0.7
    assert ga._elitism is True
    assert ga._max_generation == 100



def test_population_size_with_elitism(setup_genetic_algorithm):
    ga = setup_genetic_algorithm
    ga._max_generation = 3 # set maximum generations to 1 for this test
    ga.elitism_size = 2

    individuals = [Individual() for _ in range(10)]
    for individual in individuals:
        individual.fitness = 1  # Set fitness after creation
    population = Population(individuals)
    ga.population = population

    # Define a fitness function that just returns the current fitness
    fitness_function = lambda ind: ind.fitness

    ga.evolve(population, fitness_function)

    # Assert that only one generation was evolved
    assert ga.population.generation == 3, f"Expected generation 1, got {ga.population.generation}"
    assert len(ga.population) == 10, f"Expected fixed size 10, got {len(ga.population)}"

def test_population_size_without_elitism(setup_genetic_algorithm):
    ga = setup_genetic_algorithm
    ga._max_generation = 3 # set maximum generations to 1 for this test
    ga.elitism = False

    individuals = [Individual() for _ in range(10)]
    for individual in individuals:
        individual.fitness = 1  # Set fitness after creation
    population = Population(individuals)
    ga.population = population

    # Define a fitness function that just returns the current fitness
    fitness_function = lambda ind: ind.fitness

    ga.evolve(population, fitness_function)

    # Assert that only one generation was evolved
    assert ga.elitism == False, f"Expected elitism to be False, got {ga.elitism}"
    assert ga.population.generation == 3, f"Expected generation 1, got {ga.population.generation}"
    assert len(ga.population) == 10, f"Expected fixed size 10, got {len(ga.population)}"

def increment_fitness(ind):
    ind.fitness += 1

def test_stop_condition_target_fitness(setup_genetic_algorithm):
    ga = setup_genetic_algorithm

    individuals = [Individual() for _ in range(10)]
    for individual in individuals:
        individual.fitness = 97  # Set fitness after creation
    population = Population(individuals)
    ga.population = population

    ga.evolve(population, increment_fitness)

    # Assert that only one generation was evolved
    assert ga.population.generation == 3, f"Expected generation 1, got {ga.population.generation}"

def fitness_integer(individual):
    chromosome = individual.chromosome
    fitness_score = 0
    for i in range(len(chromosome)):
        if chromosome[i].allele == 1:
            fitness_score += 1
    individual.fitness = fitness_score

def test_ga_integer_funct1():
    mutator = IntegerMutator(1,10)

    ga = GeneticConfiguration(mutator_operator=mutator,
                              elitism_size=1,
                              max_generation=100,
                              target_fitness=3.0,
                              ).get_algorithm()

    population = Population()
    populationSize = 10
    chromosomeSize = 3

    for i in range(populationSize):
        chromosome = IntegerChromosome(chromosomeSize)
        individual = Individual(chromosome)
        population.add_individual(individual)

    ga.evolve(population, fitness_integer)

    bestFitness = ga.population.get_best_individual().fitness
    assert bestFitness == 3, f"Expected evolution and fitness score to be 3, got {ga.population.generation}"

def test_ga_integer_funct2():
    mutator = IntegerMutator(1,10)
    crossover = TwoPointCrossover()
    ga = GeneticConfiguration(mutator_operator=mutator,
                              crossover_operator=crossover,
                              elitism_size=1,
                              max_generation=100,
                              target_fitness=3.0,
                              ).get_algorithm()

    population = Population()
    populationSize = 20
    chromosomeSize = 3

    for i in range(populationSize):
        chromosome = IntegerChromosome(chromosomeSize)
        individual = Individual(chromosome)
        population.add_individual(individual)

    ga.evolve(population, fitness_integer)
    bestFitness = ga.population.get_best_individual().fitness
    assert bestFitness == 3, f"Expected evolution and fitness score to be 3, got {ga.population.generation}"

def test_ga_integer_funct3():
    mutator = IntegerMutator(1,10)
    selector = RankSelector()
    ga = GeneticConfiguration(mutator_operator=mutator,
                              parent_selector=selector,
                              elitism_size=1,
                              max_generation=100,
                              target_fitness=3.0,
                              ).get_algorithm()

    population = Population()
    populationSize = 20
    chromosomeSize = 3

    for i in range(populationSize):
        chromosome = IntegerChromosome(chromosomeSize)
        individual = Individual(chromosome)
        population.add_individual(individual)

    ga.evolve(population, fitness_integer)
    #print("generation: ", ga.population.generation)
    bestFitness = ga.population.get_best_individual().fitness
    assert bestFitness == 3, f"Expected evolution and fitness score to be 3, got {ga.population.generation}"

def test_ga_integer_funct4():
    mutator = IntegerMutator(1,10)
    selector = TournamentSelector(2)
    ga = GeneticConfiguration(mutator_operator=mutator,
                              parent_selector=selector,
                              elitism_size=1,
                              max_generation=100,
                              target_fitness=3.0,
                              ).get_algorithm()

    population = Population()
    populationSize = 30
    chromosomeSize = 3

    for i in range(populationSize):
        chromosome = IntegerChromosome(chromosomeSize)
        individual = Individual(chromosome)
        population.add_individual(individual)

    ga.evolve(population, fitness_integer)
    #print("generation: ", ga.population.generation)
    bestFitness = ga.population.get_best_individual().fitness
    assert bestFitness == 3, f"Expected evolution and fitness score to be 3, got {ga.population.generation}"

def test_ga_integer_funct5():
    mutator = IntegerMutator(1,10)
    selector = CompetitionSelector()
    ga = GeneticConfiguration(mutator_operator=mutator,
                              parent_selector=selector,
                              elitism_size=1,
                              max_generation=100,
                              target_fitness=3.0,
                              ).get_algorithm()

    population = Population()
    populationSize = 30
    chromosomeSize = 3

    for i in range(populationSize):
        chromosome = IntegerChromosome(chromosomeSize)
        individual = Individual(chromosome)
        population.add_individual(individual)

    ga.evolve(population, fitness_integer)
    #print("generation: ", ga.population.generation)
    bestFitness = ga.population.get_best_individual().fitness
    assert bestFitness == 3, f"Expected evolution and fitness score to be 3, got {ga.population.generation}"

################################Lesson 1###############################################
def lesson1_fitness(individual):
    fitness = 0
    chromosome = individual.chromosome
    for i in range(len(chromosome)):
        gene = chromosome.get_gene(i)
        if gene.allele == True:
            fitness += 1
    individual.fitness = fitness

def test_lesson1():
    mutator = BitFlipMutator()

    ga = GeneticConfiguration(mutator_operator=mutator,
                              elitism_size=1,
                              max_generation=100,
                              target_fitness=5.0,
                              ).get_algorithm()
    population = Population()
    populationSize = 10
    chromosomeSize  = 5

    for i in range(populationSize):
        bitChromosome = BitChromosome(chromosomeSize)
        individual = Individual(bitChromosome)
        population.add_individual(individual)

    ga.evolve(population, lesson1_fitness)
    #print("\n----------")
    #print("Generation number: ", ga.population.generation)
    #print("\n----------")
    individual = ga.population.get_best_individual()
    fitness_score = individual.fitness

    assert fitness_score == 5, f"Expected evolution and fitness score to be 5, got {ga.population.generation}"
    assert ga.population.generation <= 20, f"Expected generation expected to be lower than 20, got {ga.population.generation}"

#################################Lesson 2###########################

distances = [
    [0, 3, 7, 1, 3, 5],
    [3, 0, 8, 5, 1, 2],
    [7, 8, 0, 4, 3, 8],
    [1, 5, 4, 0, 6, 7],
    [3, 1, 3, 6, 0, 1],
    [5, 2, 8, 7, 1, 0]
]

def calculate_distance(chromosome):
    totalDistance = 0
    dStart = distances[0][chromosome.get_gene(0).allele]
    for i in range(len(chromosome) - 1):
        totalDistance = totalDistance + distances[chromosome.get_gene(i).allele][chromosome.get_gene(i + 1).allele]
    dEnd = distances[chromosome.get_gene(len(chromosome) - 1).allele][0]
    totalDistance = dStart + dEnd + totalDistance
    return totalDistance

def fitness(individual):
    chromosome = individual.chromosome
    totalDistance = 0
    dStart = distances[0][chromosome.get_gene(0).allele]
    for i in range(len(chromosome) -1):
        totalDistance = totalDistance + distances[chromosome.get_gene(i).allele][chromosome.get_gene(i+1).allele]
    dEnd = distances[chromosome.get_gene(len(chromosome)-1).allele][0]
    totalDistance = dStart + dEnd + totalDistance
    fitness_score = 1/totalDistance * 1000
    individual.fitness = fitness_score

def test_lesson2():
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

    ga.evolve(population, fitness)

    bestChromosome = ga.population.get_best_individual().chromosome
    totalDistance = calculate_distance(bestChromosome)
    #print("Total distance:", calculate_distance(bestChromosome))
    assert totalDistance >=14 and totalDistance <=19, f"Expected total distance to be 5, got {totalDistance}"

####################################Lesson 4################################################

def fitness_function_lesson4(individual):
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

def custom_stop(p):
    best_individual = p.get_best_individual()
    result = best_individual.custom_data
    if 20.999 < result < 21.001:
        return True
    else:
        return False


def test_lesson4():
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
                              n_threads=6,
                              ).get_algorithm()

    ga.set_custom_stop_condition(custom_stop)
    start_time = time.time()  # Start timing
    ga.evolve(population, fitness_function_lesson4)
    end_time = time.time()  # End timing

    duration = end_time - start_time
    print(f"The evolve method took {duration:.2f} seconds to complete.")

    result = ga.population.get_best_individual().custom_data
    generation = ga.population.generation

    assert 20.999 < result < 21.001, f"Expected function output to be in [20.999-21.001] inteval, but got {result}"
    assert generation < 200, f"Expected the ovolution to stop in less than 200 generations, but got {generation}"
    assert duration < 5, "The evolve method should complete within 1 seconds."
