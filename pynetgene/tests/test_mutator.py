import pytest
from pynetgene.ga import Individual
from pynetgene.exception import MutatorException
from pynetgene.chromosome import FloatGene, FloatChromosome, BitChromosome, IntegerChromosome, BitGene, IntegerGene
from pynetgene.operators.mutator import GaussianMutator, BitFlipMutator, IntegerMutator, InversionMutator, SwapMutator, ScrambleMutator, RandomMutator


@pytest.fixture
def float_individual():
    chromosome = FloatChromosome()
    for i in range(10):
        chromosome.add_gene(FloatGene(i))
    return Individual(chromosome)

@pytest.fixture
def bit_individual():
    chromosome = BitChromosome()
    for i in range(10):
        chromosome.add_gene(BitGene(bool(i % 2)))
    return Individual(chromosome)

@pytest.fixture
def integer_individual():
    chromosome = IntegerChromosome()
    for i in range(10):
        chromosome.add_gene(IntegerGene(i))
    return Individual(chromosome)

@pytest.fixture
def integer_individual2():
    chromosome = IntegerChromosome()
    for i in range(100):
        chromosome.add_gene(IntegerGene(i))
    return Individual(chromosome)

def test_gaussian_mutator_mutation_rate():
    mutator = GaussianMutator()
    assert mutator.mutation_rate == 0.05

def test_gaussian_mutator_set_mutation_rate():
    mutator = GaussianMutator()
    mutator.mutation_rate = 0.1
    assert mutator.mutation_rate == 0.1

def test_gaussian_mutator_exception(bit_individual):
    mutator = GaussianMutator()
    with pytest.raises(MutatorException):
        mutator.mutate(bit_individual)

def test_gaussian_mutator_fnct1(float_individual):
    mutator = GaussianMutator()
    mutator.mutation_rate = 1.0  # Set to 100% for testing purposes
    original_genes = [gene.allele for gene in float_individual.chromosome]
    mutator.mutate(float_individual)
    mutated_genes = [gene.allele for gene in float_individual.chromosome]
    assert any(og != mg for og, mg in zip(original_genes, mutated_genes)), "Mutation did not occur"

def test_bit_flip_mutator_fnct1(bit_individual):
    mutator = BitFlipMutator()
    mutator.mutation_rate = 1.0
    original_genes = [gene.allele for gene in bit_individual.chromosome]
    mutator.mutate(bit_individual)
    mutated_genes = [gene.allele for gene in bit_individual.chromosome]
    assert all(og != mg for og, mg in zip(original_genes, mutated_genes)), "Mutation did not occur as expected"

def test_integer_mutator_fnct1(integer_individual):
    mutator = IntegerMutator(10,20)
    mutator.mutation_rate = 1.0  # Set to 100% for testing purposes
    original_genes = [gene.allele for gene in integer_individual.chromosome]
    mutator.mutate(integer_individual)
    mutated_genes = [gene.allele for gene in integer_individual.chromosome]
    mutation_occurred = True
    # Loop through each pair of original and mutated genes
    for og, mg in zip(original_genes, mutated_genes):
        if og == mg:
            mutation_occurred = False
            break  # Stop checking once a mutation is found
    assert mutation_occurred, "Mutation did not occur"

def test_scramble_mutator_fnct1(integer_individual2):
    mutator = ScrambleMutator()
    mutator.mutation_rate = 1.0
    original_genes = [gene.allele for gene in integer_individual2.chromosome]
    mutator.mutate(integer_individual2)
    mutated_genes = [gene.allele for gene in integer_individual2.chromosome]
    isMutated = False
    for i in range(len(original_genes)):
        if original_genes[i] != mutated_genes[i]:
            isMutated = True
            break
    assert isMutated, "Mutation did not occur as expected"

def test_inversion_mutator_fnct1(integer_individual):
    mutator = InversionMutator()
    mutator.mutation_rate = 1.0
    original_genes = [gene.allele for gene in integer_individual.chromosome]
    mutator.mutate(integer_individual)
    mutated_genes = [gene.allele for gene in integer_individual.chromosome]
    isMutated = False
    for i in range(len(original_genes)):
        if original_genes[i] != mutated_genes[i]:
            isMutated = True
            break
    assert isMutated, "Mutation did not occur as expected"

def test_swap_mutator_fnct1(integer_individual):
    mutator = SwapMutator()
    mutator.mutation_rate = 1.0
    original_genes = [gene.allele for gene in integer_individual.chromosome]
    mutator.mutate(integer_individual)
    mutated_genes = [gene.allele for gene in integer_individual.chromosome]
    isMutated = False
    for i in range(len(original_genes)):
        if original_genes[i] != mutated_genes[i]:
            isMutated = True
            break
    assert isMutated, "Mutation did not occur as expected"

def test_random_mutator_fnct1(float_individual):
    mutator = RandomMutator()
    mutator.mutation_rate = 1.0  # Set to 100% for testing purposes
    original_genes = [gene.allele for gene in float_individual.chromosome]
    mutator.mutate(float_individual)
    mutated_genes = [gene.allele for gene in float_individual.chromosome]
    assert any(og != mg for og, mg in zip(original_genes, mutated_genes)), "Mutation did not occur"

# This line allows the tests to be run via the command line
if __name__ == "__main__":
    pytest.main()
