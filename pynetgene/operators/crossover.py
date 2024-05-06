from abc import ABC, abstractmethod
from pynetgene.core import Individual
from pynetgene.chromosome import Chromosome
from pynetgene.chromosome import PermutationChromosome
from pynetgene.exception import CrossoverException
import random
import numpy as np


class CrossoverOperator(ABC):
    def __init__(self, single_offspring=False):
        if not isinstance(single_offspring, bool):
            raise CrossoverException("Single Offspring variable needs to be a boolean value")
        self.single_offspring = single_offspring

    @abstractmethod
    def recombine(self, x: Individual, y: Individual) -> []:
        pass

    def set_single_offspring(self, single_offspring: bool):
        self.single_offspring = single_offspring

    def has_single_offspring(self) -> bool:
        return self.single_offspring

    @staticmethod
    def swap(x: Chromosome, y: Chromosome, index: int):
        temp = x.get_gene(index)
        x.set_gene(index, y.get_gene(index))
        y.set_gene(index, temp)


class OnePointCrossover(CrossoverOperator):
    def __init__(self, single_offspring=False):
        super().__init__(single_offspring)

    def recombine(self, x: Individual, y: Individual) -> []:
        first_offspring = x.chromosome.copy()
        second_offspring = y.chromosome.copy()

        if isinstance(first_offspring, PermutationChromosome) or isinstance(second_offspring, PermutationChromosome):
            raise CrossoverException(
                "Cannot use One Point Crossover for Permutation Chromosome. Only Permutation Crossover Operators are allowed")
        if first_offspring.length() != second_offspring.length():
            raise CrossoverException("Cannot recombine chromosomes with different lengths")

        crossover_point = random.randint(0, first_offspring.length() - 1)

        # Optimized swapping using tuple unpacking
        first_offspring.genes[crossover_point:], second_offspring.genes[crossover_point:] = \
            second_offspring.genes[crossover_point:], first_offspring.genes[crossover_point:]

        offspring = [Individual(first_offspring)]
        if not self.single_offspring:
            offspring.append(Individual(second_offspring))

        return offspring


class FixedPointCrossover(CrossoverOperator):
    def __init__(self, fixed_point, single_offspring=False):
        if fixed_point < 0:
            raise CrossoverException("Fixed point must take a positive value")
        super().__init__(single_offspring)
        self._fixed_point = fixed_point

    @property
    def fixed_point(self):
        return self._fixed_point

    def recombine(self, x: Individual, y: Individual) -> []:

        """
        Combine two individuals using the Fixed Point Crossover method.

        Args:
            x (Individual): The first individual to combine.
            y (Individual): The second individual to combine.

        Returns:
            Offspring: An Offspring object containing the result of the crossover.

        Raises:
            CrossoverException: If the crossover operation cannot be performed due to constraints.

        """

        first_offspring = x.chromosome.copy()
        second_offspring = y.chromosome.copy()

        if isinstance(first_offspring, PermutationChromosome) or isinstance(second_offspring, PermutationChromosome):
            raise CrossoverException(
                "Cannot use Fixed Point Crossover for Permutation Chromosome. Only Permutation Crossover Operators "
                "are allowed")
        if first_offspring.length() != second_offspring.length():
            raise CrossoverException("Cannot recombine chromosomes with different lengths")
        if self._fixed_point > first_offspring.length():
            raise CrossoverException("Fixed crossover point is greater than the chromosome length")
        if self._fixed_point < 0:
            raise CrossoverException("Fixed crossover point is less than 0")

        for i in range(self._fixed_point, first_offspring.length()):
            self.swap(first_offspring, second_offspring, i)

        offspring = [Individual(first_offspring)]
        if not self.single_offspring:
            offspring.append(Individual(second_offspring))

        return offspring


class HalfPointCrossover(CrossoverOperator):
    """
    In this crossover, a fixed crossover is selected at the middle of the chromosome,
    and the tails of its two parents are swapped to get new offspring.
    """

    def __init__(self, single_offspring: bool = False):
        """
        Create a new HalfPointCrossover instance with an option to generate a single offspring.

        Args:
            single_offspring (bool, optional): Set to True if only one offspring should be generated. Defaults to False.
        """
        super().__init__(single_offspring)

    def recombine(self, x: Individual, y: Individual) -> []:
        """
        Combine two individuals using the Half Point Crossover method.

        Args:
            x (Individual[C]): The first individual to combine.
            y (Individual[C]): The second individual to combine.

        Returns:
            Offspring[C]: An Offspring object containing the result of the crossover.

        Raises:
            CrossoverException: If the crossover operation cannot be performed due to constraints.
        """
        # Create copies of the parent chromosomes
        first_offspring = x.chromosome.copy()
        second_offspring = y.chromosome.copy()

        # Check if permutation chromosomes are used
        if isinstance(first_offspring, PermutationChromosome) or isinstance(second_offspring, PermutationChromosome):
            raise CrossoverException(
                "Cannot use Half Point Crossover for Permutation Chromosomes. Only Permutation Crossover Operators "
                "are allowed")

        # Check if the chromosomes have different lengths
        if first_offspring.length() != second_offspring.length():
            raise CrossoverException("Cannot recombine chromosomes with different lengths")


        # Determine the crossover point at the middle of the chromosome
        crossover_point = first_offspring.length() // 2

        # Swap the tails of the two parents
        for i in range(crossover_point, first_offspring.length()):
            self.swap(first_offspring, second_offspring, i)

        offspring = [Individual(first_offspring)]
        if not self.single_offspring:
            offspring.append(Individual(second_offspring))

        return offspring


class Order1Crossover(CrossoverOperator):

    def __init__(self, single_offspring: bool = False):
        super().__init__(single_offspring)

    def recombine(self, x: Individual, y: Individual) -> []:
        if not isinstance(x.chromosome, PermutationChromosome) or not isinstance(y.chromosome, PermutationChromosome):
            raise CrossoverException("Order 1 Crossover can be used only for Permutation Chromosome")

        if x.chromosome.length() != y.chromosome.length():
            raise CrossoverException("Cannot recombine chromosomes with different lengths")

        offspring = []
        offspring.append(Individual(self._get_offspring(x, y)))
        if not self.single_offspring:
            offspring.append(Individual(self._get_offspring(y, x)))
        return offspring

    def _get_offspring(self, x: Individual, y: Individual) -> PermutationChromosome:

        first_parent = x.chromosome
        second_parent = y.chromosome

        first_crossover_point = random.randint(0,
                                               len(x.chromosome) - 2)  # avoid to set the first crossover point the last index
        second_crossover_point = random.randint(first_crossover_point + 1, len(x.chromosome) - 1)

        offspring = PermutationChromosome()

        for i in range(first_crossover_point, second_crossover_point):
            offspring.add_gene(first_parent.get_gene(i))

        #start from the right
        if second_crossover_point != first_parent.length():
            for i in range(second_crossover_point, second_parent.length()):
                if second_parent.get_gene(i) not in offspring:
                    offspring.add_gene(second_parent.get_gene(i))

        j = 0

        for i in range(second_crossover_point):
            if second_parent.get_gene(i) not in offspring:
                if offspring.length() < second_parent.length() - first_crossover_point:
                    offspring.add_gene(second_parent.get_gene(i))
                else:
                    offspring.insert_gene(j, second_parent.get_gene(i))
                    j += 1

        return offspring

class UniformCrossover(CrossoverOperator):

    def __init__(self, probability=0.5, single_offspring: bool = False):
        if not 0 <= probability <= 1:
            raise CrossoverException("Probability must take a value between 0.0 and 1.0")
        super().__init__(single_offspring)
        self._probability = probability

    def recombine(self, x: Individual, y: Individual) -> []:
        first_offspring = x.chromosome.copy()
        second_offspring = y.chromosome.copy()
        if isinstance(first_offspring, PermutationChromosome) or isinstance(second_offspring, PermutationChromosome):
            raise CrossoverException(
                "Permutation Chromosomes are not allowed")
        if first_offspring.length() != second_offspring.length():
            raise CrossoverException("Cannot recombine chromosomes with different lengths.")

        for i in range(first_offspring.length()):
            if random.random() < self.probability:
                self.swap(first_offspring, second_offspring, i)

        offspring = [Individual(first_offspring)]
        if not self.single_offspring:
            offspring.append(Individual(second_offspring))

        return offspring

    def swap(self, chromosome1: Chromosome, chromosome2: Chromosome, index: int):
        temp = chromosome1.get_gene(index)
        chromosome1.set_gene(index, chromosome2.get_gene(index))
        chromosome2.set_gene(index, temp)

    @property
    def probability(self):
        return self._probability

class TwoPointCrossover(CrossoverOperator):
    def __init__(self, single_offspring=False):
        super().__init__(single_offspring)

    def recombine(self, x: Individual, y: Individual) -> []:
        first_offspring = x.chromosome.copy()
        second_offspring = y.chromosome.copy()

        if isinstance(first_offspring, PermutationChromosome) or isinstance(second_offspring, PermutationChromosome):
            raise CrossoverException("Cannot use Two Point Crossover for Permutation Chromosome. Only Permutation Crossover Operators are allowed")
        if first_offspring.length() != second_offspring.length():
            raise CrossoverException("Cannot recombine chromosomes with different lengths")
        if first_offspring.length() < 3:
            raise CrossoverException("Multi point crossover cannot work if chromosome length is lower than 3")

        # Generate two crossover points
        first_crossover_point = random.randint(0, first_offspring.length() - 2)
        second_crossover_point = random.randint(first_crossover_point + 1, first_offspring.length() - 1)

        for i in range(first_crossover_point, second_crossover_point):
            self.swap(first_offspring, second_offspring, i)

        offspring = [Individual(first_offspring)]
        if not self.single_offspring:
            offspring.append(Individual(second_offspring))

        return offspring

    def swap(self, chromosome1: Chromosome, chromosome2: Chromosome, index: int):
        temp = chromosome1.get_gene(index)
        chromosome1.set_gene(index, chromosome2.get_gene(index))
        chromosome2.set_gene(index, temp)

#
# from pynetgene.chromosome import IntegerChromosome
# from pynetgene.gene import IntegerGene
#
# ch1 = PermutationChromosome()
#
# ch1.add_gene(IntegerGene(1))
# ch1.add_gene(IntegerGene(2))
# ch1.add_gene(IntegerGene(3))
# ch1.add_gene(IntegerGene(4))
# ch1.add_gene(IntegerGene(5))
# ch1.add_gene(IntegerGene(6))
#
# ch2 = PermutationChromosome()
#
# ch2.add_gene(IntegerGene(6))
# ch2.add_gene(IntegerGene(5))
# ch2.add_gene(IntegerGene(4))
# ch2.add_gene(IntegerGene(3))
# ch2.add_gene(IntegerGene(2))
# ch2.add_gene(IntegerGene(1))
#
# ind1 = Individual(ch1)
# ind2 = Individual(ch2)
#
# opc = Order1Crossover()
#
# for i in range(100):
#     offspring = opc.recombine(ind1, ind2)
#
# indvs = offspring.get_individuals()
#
# print("first indv: ", indvs[0])
# print("second indv: ", indvs[1])
#
# list = [1,2,3,4,5,6,7,8,9]
#
# list.insert(1, 10)
#
# print("list: ", list)
#
