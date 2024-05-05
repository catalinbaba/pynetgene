import pytest
from pynetgene.core import Individual
from pynetgene.chromosome import Chromosome, PermutationChromosome
from pynetgene.exception import CrossoverException
from pynetgene.chromosome import FloatGene, FloatChromosome, BitChromosome, IntegerChromosome, BitGene, IntegerGene
from pynetgene.operators.crossover import *

# Test CrossoverOperator Initialization
def test_crossover_operator_initialization():
    with pytest.raises(CrossoverException):
        OnePointCrossover(single_offspring="not_bool")
    co = OnePointCrossover(single_offspring=True)
    assert co.single_offspring == True

# Test OnePointCrossover behavior
def test_one_point_crossover_funct1():
    chromosome_x = IntegerChromosome(50)
    chromosome_y = IntegerChromosome(50)
    ind_x = Individual(chromosome_x)
    ind_y = Individual(chromosome_y)
    op_crossover = OnePointCrossover()
    offspring = op_crossover.recombine(ind_x, ind_y)
    assert len(offspring) == 2
    ch1 = offspring[0].chromosome
    print("len: ", len(ch1))
    genes_ch1 = offspring[0].chromosome.to_list()
    genes_ch2 = offspring[1].chromosome.to_list()
    set_x = set(chromosome_x.to_list())
    set_y = set(chromosome_y.to_list())
    set_offspring1 = set(genes_ch1)
    set_offspring2 = set(genes_ch2)

    # Check that all genes from each parent are in the combined set of both offspring
    assert set_x <= (set_offspring1 | set_offspring2), "All genes from chromosome_x should be in the offspring"
    assert set_y <= (set_offspring1 | set_offspring2), "All genes from chromosome_y should be in the offspring"

    # Additionally check intersection to ensure crossover mixing
    assert set_x & set_offspring1, "Offspring 1 should have genes from chromosome_x"
    assert set_x & set_offspring2, "Offspring 2 should have genes from chromosome_x"
    assert set_y & set_offspring1, "Offspring 1 should have genes from chromosome_y"
    assert set_y & set_offspring2, "Offspring 2 should have genes from chromosome_y"

def test_one_point_crossover_funct2():
    chromosome_x = FloatChromosome(50)
    chromosome_y = FloatChromosome(50)
    ind_x = Individual(chromosome_x)
    ind_y = Individual(chromosome_y)
    op_crossover = OnePointCrossover()
    offspring = op_crossover.recombine(ind_x, ind_y)
    assert len(offspring) == 2
    genes_ch1 = offspring[0].chromosome.to_list()
    genes_ch2 = offspring[1].chromosome.to_list()
    set_x = set(chromosome_x.to_list())
    set_y = set(chromosome_y.to_list())
    set_offspring1 = set(genes_ch1)
    set_offspring2 = set(genes_ch2)

    # Check that all genes from each parent are in the combined set of both offspring
    assert set_x <= (set_offspring1 | set_offspring2), "All genes from chromosome_x should be in the offspring"
    assert set_y <= (set_offspring1 | set_offspring2), "All genes from chromosome_y should be in the offspring"

    # Additionally check intersection to ensure crossover mixing
    assert set_x & set_offspring1, "Offspring 1 should have genes from chromosome_x"
    assert set_x & set_offspring2, "Offspring 2 should have genes from chromosome_x"
    assert set_y & set_offspring1, "Offspring 1 should have genes from chromosome_y"
    assert set_y & set_offspring2, "Offspring 2 should have genes from chromosome_y"


def test_one_point_crossover_funct3():
    chromosome_x = BitChromosome(50)
    chromosome_y = BitChromosome(50)
    ind_x = Individual(chromosome_x)
    ind_y = Individual(chromosome_y)
    op_crossover = OnePointCrossover()
    offspring = op_crossover.recombine(ind_x, ind_y)
    assert len(offspring) == 2
    genes_ch1 = offspring[0].chromosome.to_list()
    genes_ch2 = offspring[1].chromosome.to_list()
    set_x = set(chromosome_x.to_list())
    set_y = set(chromosome_y.to_list())
    set_offspring1 = set(genes_ch1)
    set_offspring2 = set(genes_ch2)

    # Check that all genes from each parent are in the combined set of both offspring
    assert set_x <= (set_offspring1 | set_offspring2), "All genes from chromosome_x should be in the offspring"
    assert set_y <= (set_offspring1 | set_offspring2), "All genes from chromosome_y should be in the offspring"

    # Additionally check intersection to ensure crossover mixing
    assert set_x & set_offspring1, "Offspring 1 should have genes from chromosome_x"
    assert set_x & set_offspring2, "Offspring 2 should have genes from chromosome_x"
    assert set_y & set_offspring1, "Offspring 1 should have genes from chromosome_y"
    assert set_y & set_offspring2, "Offspring 2 should have genes from chromosome_y"

def test_fixed_point_crossover_funct1():
    chromosome_x = IntegerChromosome(50)
    chromosome_y = IntegerChromosome(50)
    ind_x = Individual(chromosome_x)
    ind_y = Individual(chromosome_y)
    op_crossover = FixedPointCrossover(10)
    offspring = op_crossover.recombine(ind_x, ind_y)
    assert len(offspring) == 2
    genes_ch1 = offspring[0].chromosome.to_list()
    genes_ch2 = offspring[1].chromosome.to_list()
    set_x = set(chromosome_x.to_list())
    set_y = set(chromosome_y.to_list())
    set_offspring1 = set(genes_ch1)
    set_offspring2 = set(genes_ch2)

    # Check that all genes from each parent are in the combined set of both offspring
    assert set_x <= (set_offspring1 | set_offspring2), "All genes from chromosome_x should be in the offspring"
    assert set_y <= (set_offspring1 | set_offspring2), "All genes from chromosome_y should be in the offspring"

    # Additionally check intersection to ensure crossover mixing
    assert set_x & set_offspring1, "Offspring 1 should have genes from chromosome_x"
    assert set_x & set_offspring2, "Offspring 2 should have genes from chromosome_x"
    assert set_y & set_offspring1, "Offspring 1 should have genes from chromosome_y"
    assert set_y & set_offspring2, "Offspring 2 should have genes from chromosome_y"

def test_fixed_point_crossover_funct2():
    chromosome_x = IntegerChromosome(10, 0, 10)
    chromosome_y = IntegerChromosome(10, 0, 10)
    ind_x = Individual(chromosome_x)
    ind_y = Individual(chromosome_y)
    crossover_point = 3
    op_crossover = FixedPointCrossover(crossover_point)
    offspring = op_crossover.recombine(ind_x, ind_y)
    assert len(offspring) == 2
    genes_ch1 = offspring[0].chromosome.to_list()
    genes_ch2 = offspring[1].chromosome.to_list()
    assert genes_ch1[:crossover_point] == chromosome_x.to_list()[:crossover_point]
    assert genes_ch1[crossover_point:] == chromosome_y.to_list()[crossover_point:]
    assert genes_ch2[:crossover_point] == chromosome_y.to_list()[:crossover_point]
    assert genes_ch2[crossover_point:] == chromosome_x.to_list()[crossover_point:]

def test_order1_crossover_funct1():
    chromosome_x = PermutationChromosome(10, 0)
    chromosome_y = PermutationChromosome(10, 0)
    ind_x = Individual(chromosome_x)
    ind_y = Individual(chromosome_y)
    op_crossover = Order1Crossover()
    offspring = op_crossover.recombine(ind_x, ind_y)
    genes_ch1 = offspring[0].chromosome.to_list()
    genes_ch2 = offspring[1].chromosome.to_list()

    assert sorted(chromosome_x.to_list()) == sorted(genes_ch1), "Offspring genes must be a permutation of the parent genes"
    assert sorted(chromosome_y.to_list()) == sorted(genes_ch2), "Offspring genes must be a permutation of the parent genes"

def test_uniform_crossover_funct1():
    chromosome_x = IntegerChromosome()
    chromosome_y = IntegerChromosome()
    for i in range(1, 6):
        chromosome_x.add_gene(IntegerGene(i))
    for i in range(5, 0, -1):
        chromosome_y.add_gene(IntegerGene(i))
    ind_x = Individual(chromosome_x)
    ind_y = Individual(chromosome_y)
    op_crossover = UniformCrossover(probability=1.0)
    offspring = op_crossover.recombine(ind_x, ind_y)

    # Check if all genes were swapped
    assert offspring[0].chromosome.to_list() == [5, 4, 3, 2, 1], "All genes should be swapped"

    # Test with 0% swapping probability
    uc = UniformCrossover(probability=0.0)
    offspring = uc.recombine(ind_x, ind_y)

    # Check if no genes were swapped
    assert offspring[0].chromosome.to_list() == [1, 2, 3, 4, 5], "No genes should be swapped"

def test_uniform_crossover_probability():
    with pytest.raises(CrossoverException):
        UniformCrossover(probability=-0.1)
    with pytest.raises(CrossoverException):
        UniformCrossover(probability=1.1)

def test_uniform_crossover_exception():
    chromosome_x = PermutationChromosome()
    chromosome_y = PermutationChromosome()
    individual_x = Individual(chromosome_x)
    individual_y = Individual(chromosome_y)
    uc = UniformCrossover()

    # Expect an exception for using permutation chromosomes
    with pytest.raises(CrossoverException):
        uc.recombine(individual_x, individual_y)

def test_two_point_crossover_funct1():
    chromosome_x = IntegerChromosome()
    chromosome_y = IntegerChromosome()
    for i in range(1, 6):
        chromosome_x.add_gene(IntegerGene(i))
    for i in range(5, 0, -1):
        chromosome_y.add_gene(IntegerGene(i))
    ind_x = Individual(chromosome_x)
    ind_y = Individual(chromosome_y)
    op_crossover = TwoPointCrossover()
    offspring = op_crossover.recombine(ind_x, ind_y)
    assert offspring[0].chromosome.to_list() != [1, 2, 3, 4, 5], "Genes should be swapped"
    assert offspring[0].chromosome.to_list() != [5, 4, 3, 2, 1], "Genes should be swapped"

def test_two_point_crossover_exception():
    # Create permutation chromosomes which should not be allowed
    chromosome_x = PermutationChromosome()
    chromosome_y = PermutationChromosome()
    ind_x = Individual(chromosome_x)
    ind_y = Individual(chromosome_y)
    crossover = TwoPointCrossover()

    with pytest.raises(CrossoverException):
        crossover.recombine(ind_x, ind_y)


# This line allows the tests to be run via the command line
if __name__ == "__main__":
    pytest.main()

