from pynetgene.exception import MutatorException
from pynetgene.chromosome import *
from random import shuffle

from pynetgene.ga import Individual


class MutatorOperator(ABC):

    def __init__(self):
        self._mutation_rate = 0.05

    @property
    def mutation_rate(self):
        return self._mutation_rate

    @mutation_rate.setter
    def mutation_rate(self, mutation_rate):
        self._mutation_rate = mutation_rate

    @abstractmethod
    def mutate(self, individual: Individual):
        pass

class GaussianMutator(MutatorOperator):

    def __init__(self, sigma=0.5):
        super().__init__()
        self._sigma = sigma

    @property
    def standard_deviation(self):
        return self._sigma

    @standard_deviation.setter
    def standard_deviation(self, sigma):
        self._sigma = sigma

    def mutate(self, individual: Individual):
        if not isinstance(individual.chromosome, FloatChromosome):
            raise MutatorException("Gaussian Mutator is applicable only for float chromosomes")

        chromosome = individual.chromosome

        for i in range(chromosome.length()):
            rand = random.random()
            if rand < self._mutation_rate:
                delta = random.gauss(mu=0, sigma=self._sigma)
                value = delta + chromosome.get_gene(i)
                chromosome.set_gene(i, value)

class BitFlipMutator(MutatorOperator):

    def mutate(self, individual: Individual):
        if not isinstance(individual.chromosome, BitChromosome):
            raise MutatorException("Bit Flip Mutator is applicable only for BitChromosome")

        chromosome = individual.chromosome
        for i in range(chromosome.length()):
            rand = random.random()
            if rand < self.mutation_rate:
                current_gene = chromosome.get_gene(i)  # Assuming this returns a boolean
                chromosome.set_gene(i, not current_gene)

class IntegerMutator(MutatorOperator):

    def __init__(self, min_range=0, max_range=9):
        super().__init__()
        self._min_range = min_range
        self._max_range = max_range

    @property
    def min_range(self):
        return self._min_range

    @property
    def max_range(self):
        return self._max_range

    def mutate(self, individual: Individual):
        if not isinstance(individual.chromosome, IntegerChromosome):
            raise MutatorException("Integer Mutator is applicable only for int chromosomes")

        chromosome = individual.chromosome
        for i in range(chromosome.length()):
            rand = random.random()
            if rand < self._mutation_rate:
                value = random.randint(self._min_range, self._max_range)
                chromosome.set_gene(i, value)

class InversionMutator(MutatorOperator):

    def mutate(self, individual: Individual):
        first_index = random.randint(0, len(individual.chromosome) - 2)  # avoid to set the first crossover point the last index
        second_index = random.randint(first_index + 1, len(individual.chromosome) - 1)
        rand = random.random()
        if rand < self._mutation_rate:
            genes = []
            for i in range(second_index, first_index - 1, -1):
                genes.append(individual.chromosome.get_gene(i))
            for i in range(first_index, second_index + 1):
                individual.chromosome.set_gene(i, genes[i - first_index])

class SwapMutator(MutatorOperator):

    def mutate(self, individual: Individual):
        first_index = random.randint(0, len(individual.chromosome) - 2)  # avoid to set the first crossover point the last index
        second_index = random.randint(first_index + 1, len(individual.chromosome) - 1)
        rand = random.random()
        if rand < self._mutation_rate:
            tmp = individual.chromosome.get_gene(first_index)
            individual.chromosome.set_gene(first_index, individual.chromosome.get_gene(second_index))
            individual.chromosome.set_gene(second_index, tmp)

class ScrambleMutator(MutatorOperator):

    def mutate(self, individual: Individual):
        first_index = random.randint(0,
                                     len(individual.chromosome) - 2)  # avoid to set the first crossover point the last index
        second_index = random.randint(first_index + 1, len(individual.chromosome) - 1)
        rand = random.random()
        if rand < self._mutation_rate:
            shuffled = list(range(first_index, second_index+1))
            shuffle(shuffled)

            genes = []
            for i in shuffled:
                genes.append(individual.chromosome.get_gene(i))

            for i in range(first_index, second_index + 1):
                individual.chromosome.set_gene(i, genes[i - first_index])

class RandomMutator(GaussianMutator):

    def mutate(self, individual: Individual):
        chromosome = individual.chromosome

        if not isinstance(chromosome, NumericChromosome):
            raise MutatorException("Random Mutator is applicable only for numeric chromosomes")

        if isinstance(chromosome, FloatChromosome):
            self._mutate_float_chromosome(chromosome)
        elif isinstance(chromosome, IntegerChromosome):
            self._mutate_integer_chromosome(chromosome)

    def _mutate_integer_chromosome(self, chromosome):
        for i in range(chromosome.length()):
            rand = random.random()
            if rand < self.mutation_rate:
                delta = random.gauss(mu=0, sigma=self._sigma) * 10
                chromosome.set_gene(i, delta)

    def _mutate_float_chromosome(self, chromosome):
        for i in range(chromosome.length()):
            rand = random.random()
            if rand < self.mutation_rate:
                delta = random.gauss(mu=0, sigma=self._sigma)
                chromosome.set_gene(i, delta)

#
# from pynetgene.ga import Individual
#
# ch1 = FloatChromosome()
#
# ch1.add_gene(FloatGene(1))
# ch1.add_gene(FloatGene(2))
# ch1.add_gene(FloatGene(3))
# ch1.add_gene(FloatGene(4))
# ch1.add_gene(FloatGene(5))
# ch1.add_gene(FloatGene(6))
#
# indv = Individual(ch1)
#
# #mut = ScrambleMutator()
# mut = GaussianMutator()
# mut.mutation_rate = 1.0
#
# mut.mutate(indv)
#
# #print("individual: ", indv)
