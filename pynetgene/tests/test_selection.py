import pytest

from pynetgene.core import Population
from pynetgene.exception import SelectionException
from pynetgene.ga import Individual
from pynetgene.operators.selection import *

# Helper function to create a population with specified fitness values
def create_population_with_fitness(fitness_values):
    population = Population()
    for fitness in fitness_values:
        ind = Individual()
        ind.fitness = fitness
        population.add_individual(ind)
    return population


#####################Tournament Selector############################
# Test the basic functionality with a suitable population and tournament size
def test_tournament_selector_basic_tournament_selection():
    population = create_population_with_fitness([1, 2, 3, 4, 5])
    selector = TournamentSelector(tournament_size=2)
    ind1, ind2 = selector.select_parents(population)
    assert ind1 != ind2

# Test behavior when the tournament size exceeds the population size
def test_tournament_selector_size_exceeds_population():
    population = create_population_with_fitness([1, 2, 3])
    selector = TournamentSelector(tournament_size=4)
    with pytest.raises(SelectionException):
        selector.select(population)

# Test behavior with a single individual in the population
def test_tournament_selector_single_individual_population():
    population = create_population_with_fitness([10])
    selector = TournamentSelector(tournament_size=1)
    with pytest.raises(SelectionException):
        selector.select_parents(population)

# Test the incest prevention feature
def test_tournament_selector_incest_prevention():
    population = create_population_with_fitness([1, 2, 3, 4, 5])
    selector = TournamentSelector(tournament_size=2)
    selector.incest_prevention = True
    ind1, ind2 = selector.select_parents(population)
    assert ind1 != ind2

# Test the randomness of the tournament selection
def test_tournamet_selector_randomness_in_selection():
    population = create_population_with_fitness([1, 2, 3, 4, 5])
    selector = TournamentSelector(tournament_size=2)
    outcomes = set()
    for _ in range(10):
        ind1, ind2 = selector.select_parents(population)
        outcomes.add((ind1.fitness, ind2.fitness))
    assert len(outcomes) > 1


######################RANK_Selector#####################

def test_rank_selection_biased_towards_higher_ranks():
    population = create_population_with_fitness([1, 2, 3, 4, 5])
    selector = RankSelector()
    outcomes = {}
    for _ in range(1000):
        selected = selector.select(population)
        outcomes[selected.fitness] = outcomes.get(selected.fitness, 0) + 1

    # Expect more selections of higher fitness individuals
    assert outcomes[5] > outcomes[1], "Higher ranked individuals should be selected more frequently"

# Test behavior with a single individual in the population
def test_rank_selection_single_individual_population():
    population = create_population_with_fitness([10])
    selector = RankSelector()
    selected = selector.select(population)
    assert selected.fitness == 10, "The only individual should be selected"

# Test the randomness of the rank-based selection
def test_rank_selection_randomness_in_rank_selection():
    population = create_population_with_fitness([1, 2, 3, 4, 5])
    selector = RankSelector()
    outcomes = {}
    for _ in range(1000):
        selected = selector.select(population)
        outcomes[selected.fitness] = outcomes.get(selected.fitness, 0) + 1

    # Checking for at least some distribution across ranks, not just the highest
    assert len(outcomes) > 1, "Selection should not always pick the same rank"

# Test the incest prevention feature
def test_rank_selection_incest_prevention():
    population = create_population_with_fitness([1, 2, 3, 4, 5])
    selector = RankSelector()
    selector.incest_prevention = True
    ind1, ind2 = selector.select_parents(population)
    assert ind1 != ind2, "Incest prevention should avoid selecting the same individual"


#####################Roulette Selection#########################

# Test the basic functionality where selection should be biased towards higher fitness individuals
def test_roulette_selection_biased_towards_higher_fitness():
    population = create_population_with_fitness([1, 2, 3, 4, 10])
    selector = RouletteSelector()
    outcomes = {}
    for _ in range(1000):
        selected = selector.select(population)
        outcomes[selected.fitness] = outcomes.get(selected.fitness, 0) + 1

    # Expect more selections of higher fitness individuals
    assert outcomes[10] > outcomes[1], "Higher fitness individuals should be selected more frequently"

# Test behavior with a single individual in the population
def test_roulette_selection_single_individual_population():
    population = create_population_with_fitness([10])
    selector = RouletteSelector()
    selected = selector.select(population)
    assert selected.fitness == 10, "The only individual should be selected"

# Test randomness of the roulette selection
def test_roulette_selection_randomness_in_roulette_selection():
    population = create_population_with_fitness([5, 5, 5, 5, 5])
    selector = RouletteSelector()
    outcomes = {}
    for _ in range(1000):
        selected = selector.select(population)
        outcomes[selected.fitness] = outcomes.get(selected.fitness, 0) + 1

    # Checking for uniform distribution across individuals with the same fitness
    assert len(set(outcomes.values())) == 1, "Selection should be evenly distributed among individuals of the same fitness"

# Test the incest prevention feature
def test_roulette_selection_incest_prevention():
    population = create_population_with_fitness([1, 2, 3, 4, 5])
    selector = RouletteSelector()
    selector.incest_prevention = True
    ind1, ind2 = selector.select_parents(population)
    assert ind1 != ind2, "Incest prevention should avoid selecting the same individual"

#################################Competition Selector######################

# Test the basic functionality of selecting the higher fitness individual
def test_competition_selector_basic():
    population = create_population_with_fitness([1, 10])
    selector = CompetitionSelector()
    selected = selector.select(population)
    assert selected.fitness == 10, "Selector should return the individual with higher fitness"

# Test behavior with a minimum valid population
def test_competition_selector_min_population():
    population = create_population_with_fitness([5, 15])
    selector = CompetitionSelector()
    selected = selector.select(population)
    assert selected.fitness == 15, "Selector should return the individual with higher fitness from a two-person population"

# Test randomness and fairness of selection
def test_competition_selector_fairness():
    population = create_population_with_fitness([1, 2, 3, 4, 5])
    selector = CompetitionSelector()
    results = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    num_trials = 1000
    for _ in range(num_trials):
        selected = selector.select(population)
        results[selected.fitness] += 1

    # Check if higher fitness individuals are indeed more likely to win
    assert all(results[i] < results[i+1] for i in range(1, 5)), "Higher fitness individuals should win more frequently"

# Test the incest prevention mechanism
def test_competition_selector_incest_prevention():
    population = create_population_with_fitness([1, 2, 3, 4, 5])
    selector = CompetitionSelector()
    selector.incest_prevention = True
    ind1, ind2 = selector.select_parents(population)
    assert ind1 != ind2, "Incest prevention should ensure two different individuals are selected"