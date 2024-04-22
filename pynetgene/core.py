from typing import List
from functools import total_ordering
import random


@total_ordering
class Individual:
    def __init__(self, chromosome=None):
        """
        Create a new instance of Individual.

        :param chromosome: Individual's chromosome.
        """
        self._chromosome = chromosome
        self.custom_data = None
        self._fitness_score = 0.0

    @property
    def chromosome(self):
        """
        Get or set the chromosome of the individual.

        :return: The chromosome of the individual.

        """
        return self._chromosome

    @chromosome.setter
    def chromosome(self, chromosome):
        self._chromosome = chromosome

    @property
    def fitness(self):
        return self._fitness_score

    @fitness.setter
    def fitness(self, fitness):
        self._fitness_score = fitness

    def __lt__(self, other):
        """
        Compare two individuals by fitness value.

        :param other: Individual to compare with.

        :return: A negative integer, zero, or a positive integer as this individual's fitness is less than, equal to,
        or greater than the specified individual.
        """
        return self._fitness_score < other.fitness

    def __eq__(self, other):
        """
        Compare two individuals for equality based on fitness value.

        :param other: Individual to compare with.

        :return: True if both individuals have the same fitness value, False otherwise.
        """
        return self._fitness_score == other.fitness

    def __str__(self):
        """
        String representation of the individual.

        :return: String representation of the individual.
        """
        return f"Fitness: {self._fitness_score}\n{str(self.chromosome)}"

    def copy(self):
        """
        Create a deep copy of the individual.

        :return: A new Individual object with a deep copy of the chromosome.
        """
        return Individual(chromosome=self.chromosome.copy())


@total_ordering
class Population:
    def __init__(self, population: list = None):
        """
        Initialize a Population object.

        Args:
            population (list, optional): Initial population as a list of individuals. Defaults to an empty list if None.

        Raises:
            TypeError: If the population is not a list.

        """
        if population is None:
            self._population = []
        elif not isinstance(population, list):
            raise TypeError("Expected 'population' to be a list.")
        else:
            self._population = population
        self._generation = 0

    @property
    def generation(self):
        """
        Get the current generation of the population.

        Returns:
            int: The current generation.

        """
        return self._generation

    @generation.setter
    def generation(self, generation):
        """
        Set the current generation of the population.

        Args:
            generation (int): The generation to set.

        """
        self._generation = generation

    @property
    def fitness(self):
        """
        Calculate the total fitness of the population.

        Returns:
            float: The sum of fitness scores for all individuals in the population.

        """
        return sum(individual.fitness for individual in self._population)

    def add_individual(self, individual):
        """
        Add an individual to the population.

        Args:
            individual: The individual to add.

        """
        self._population.append(individual)

    def remove_individual(self, individual):
        """
        Remove an individual from the population.

        Args:
            individual: The individual to remove.

        """
        self._population.remove(individual)

    def remove_individual_at(self, index):
        """
        Remove an individual from the population at a specific index.

        Args:
            index (int): The index of the individual to remove.

        """
        del self._population[index]

    def clear(self):
        """
        Remove all individuals from the population, effectively clearing it.

        """
        self._population.clear()

    def sort(self):
        """
        Sort the population in descending order based on fitness scores.

        """
        self._population.sort(reverse=True)

    def get_random_individual(self):
        """
        Get a random individual from the population.

        Returns:
            individual: A randomly selected individual.

        """
        return self._population[random.randint(0, len(self._population) - 1)]

    def get_individual(self, index):
        """
        Get an individual from the population at the specified index.

        Returns:
            individual: Selected individual.

        """
        return self._population[index]

    def get_best_individual(self):
        """
        Get the best (highest fitness) individual from the population.

        Returns:
            individual: The best individual.

        """
        return max(self._population)

    def clone(self):
        """
        Create a copy of the population.

        Returns:
            Population: A new Population object with a copy of the population.

        """
        return Population(population=self._population.copy())

    def __str__(self):
        """
        Get a string representation of the population.

        Returns:
            str: A string representing the population's size and individual details.

        """
        return f"Population size: {len(self)}\n" + "\n".join(
            f"Individual : {i}\n{str(individual)}" for i, individual in enumerate(self._population)
        )

    def __eq__(self, other):
        """
        Compare two populations for equality.

        Args:
            other (Population): Another Population object to compare with.

        Returns:
            bool: True if populations are equal, False otherwise.

        """
        if isinstance(other, Population):
            return self._population == other._population
        return False

    def __lt__(self, other):
        """
        Compare two populations based on their populations' lexicographical order.

        Args:
            other (Population): Another Population object to compare with.

        Returns:
            bool: True if the current population is less than the other population, False otherwise.

        """
        if isinstance(other, Population):
            return self._population < other._population
        return NotImplemented

    def __len__(self):
        """
        Get the number of individuals in the population.

        Returns:
            int: The number of individuals in the population.

        """
        return len(self._population)

    def __getitem__(self, index):
        """
        Get an individual from the population using indexing.

        Args:
            index (int): The index of the individual to retrieve.

        Returns:
            individual: The individual at the specified index.

        """
        return self._population[index]

    def __setitem__(self, index, individual):
        """
        Set an individual in the population using indexing.

        Args:
            index (int): The index of the individual to set.
            individual: The individual to set at the specified index.

        """
        self._population[index] = individual


class Offspring:
    def __init__(self):
        self._offspring: List[Individual] = []

    def add_offspring(self, individual: Individual):
        """
        Add a new offspring.

        :param individual: Individual to add.
        """
        self._offspring.append(individual)

    def get_offspring(self, index: int) -> Individual:
        """
        Get a new offspring at the specified index.

        :param index: Index of the offspring.
        :return: Offspring at the specified index.
        """
        return self._offspring[index]

    def get_size(self) -> int:
        """
        Get the total number of offspring.

        :return: Number of offspring.
        """
        return len(self._offspring)

    def get_individuals(self) -> List[Individual]:
        """
        Get a list of individuals.

        :return: List of individuals.
        """
        return self._offspring


    def __getitem__(self, index: int) -> Individual:
        """
        Allows for bracket notation access (e.g., offspring[2]).

        :param index: Index of the desired offspring.
        :return: Offspring at the specified index.
        """
        return self._offspring[index]

    def __setitem__(self, index: int, individual: Individual):
        """
        Allows for bracket notation setting (e.g., offspring[2] = new_individual).

        :param index: Index where the individual should be set.
        :param individual: New individual to place at the specified index.
        """
        self._offspring[index] = individual


class Parents:
    """
    Parents class holds two individuals.
    """

    def __init__(self, first_individual: Individual, second_individual: Individual):
        """
        Create a new instance of Parents.

        :param first_individual: first individual (parent)
        :param second_individual: second individual (parent)
        """
        self._first_parent = first_individual
        self._second_parent = second_individual

    @property
    def first_parent(self):
        """
        Get the first individual (parent).

        :return: first parent
        """
        return self._first_parent

    @property
    def second_parent(self):
        """
        Get the second individual (parent).

        :return: second parent
        """
        return self._second_parent

    def get_parents(self):
        """
        Get both parents as a list.

        :return: List with both parents
        """
        return [self._first_parent, self._second_parent]



#
# from pynetgene.chromosome import IntegerChromosome
#
# ind1 = Individual(IntegerChromosome(10))
# ind2 = Individual(IntegerChromosome(10))
# ind3 = Individual(IntegerChromosome(10))
# ind4 = Individual(IntegerChromosome(10))
# ind5 = Individual(IntegerChromosome(10))
#
# ind1.fitness = 1
# ind2.fitness = 2
# ind3.fitness = 3
# ind4.fitness = 4
# ind5.fitness = 5
#
#
# population = Population()
#
# population.add_individual(ind1)
# population.add_individual(ind2)
# population.add_individual(ind3)
# population.add_individual(ind4)
# population.add_individual(ind5)
#
#
#
# #print("best individual: ", population.get_best_individual())
# #print("random individual: ", population.get_random_individual())
#
# population.sort()

# population[0] = Individual(IntegerChromosome(10))

# print(population[0])

# population.clear()

# print("fitness: ", population.fitness)

# print(population)

# parents = Parents(ind1, ind2)
#
# prts = parents.get_parents()
#
#
# print("first parent: " , prts[0])
#
# print("second parent: " , prts[1])

