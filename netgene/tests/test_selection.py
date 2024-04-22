import pytest

from netgene.core import Population
from netgene.exception import SelectionException
from netgene.ga import Individual
from netgene.operators.selection import *

# Helper function to create a population with specified fitness values
def create_population_with_fitness(fitness_values):
    population = Population()
    for fitness in fitness_values:
        ind = Individual()
        ind.fitness = fitness
        population.add_individual(ind)
    return population

# Test the basic functionality with a suitable population and tournament size
def test_basic_tournament_selection():
    population = create_population_with_fitness([1, 2, 3, 4, 5])
    selector = TournamentSelector(tournament_size=2)
    ind1, ind2 = selector.select_parents(population)
    assert ind1 != ind2

# Test behavior when the tournament size exceeds the population size
def test_tournament_size_exceeds_population():
    population = create_population_with_fitness([1, 2, 3])
    selector = TournamentSelector(tournament_size=4)
    with pytest.raises(SelectionException):
        selector.select(population)

# Test behavior with a single individual in the population
def test_single_individual_population():
    population = create_population_with_fitness([10])
    selector = TournamentSelector(tournament_size=1)
    with pytest.raises(SelectionException):
        selector.select_parents(population)

# Test the incest prevention feature
def test_incest_prevention():
    population = create_population_with_fitness([1, 2, 3, 4, 5])
    selector = TournamentSelector(tournament_size=2)
    selector.incest_prevention = True
    ind1, ind2 = selector.select_parents(population)
    assert ind1 != ind2

# Test the randomness of the tournament selection
def test_randomness_in_selection():
    population = create_population_with_fitness([1, 2, 3, 4, 5])
    selector = TournamentSelector(tournament_size=2)
    outcomes = set()
    for _ in range(10):
        ind1, ind2 = selector.select_parents(population)
        outcomes.add((ind1.fitness, ind2.fitness))
    assert len(outcomes) > 1