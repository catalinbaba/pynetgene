import pytest

from pynetgene.exception import GaException
from pynetgene.ga import GeneticAlgorithm, GeneticConfiguration
from pynetgene.operators.crossover import OnePointCrossover, TwoPointCrossover
from pynetgene.operators.mutator import IntegerMutator, GaussianMutator, InversionMutator
from pynetgene.operators.selection import RouletteSelector, ParentSelector, RankSelector


def setup_genetic_algorithm():
    ga = GeneticConfiguration(elitism_size=1,
                              max_generation=100,
                              target_fitness=3.0,
                              ).get_algorithm()
    return ga

def test_mutation_rate_valid():
    ga = setup_genetic_algorithm()
    ga.mutation_rate = 0.5  # This should work, as it's a valid mutation rate
    assert ga.mutation_rate == 0.5

def test_mutation_rate_too_high():
    ga = setup_genetic_algorithm()
    with pytest.raises(GaException) as e:
        ga.mutation_rate = 1.1  # This should raise an error because it's greater than 1.0
    assert "Mutation rate must be between 0.0 and 1.0" in str(e.value)

def test_mutation_rate_negative():
    ga = setup_genetic_algorithm()
    with pytest.raises(GaException) as e:
        ga.mutation_rate = -0.1  # This should raise an error because it's less than 0.0
    assert "Mutation rate must be between 0.0 and 1.0" in str(e.value)

def test_elitism_valid():
    ga = setup_genetic_algorithm()
    try:
        ga.elitism = True
        ga.elitism = False
    except Exception as e:
        pytest.fail(f"Setting elitism should not raise any exception, but raised {e}")

def test_elitism_invalid_type():
    ga = setup_genetic_algorithm()
    with pytest.raises(GaException) as exc_info:
        ga.elitism = "not a boolean"  # This should raise an error because it's not a boolean
    assert "Elitism must be either True or False" in str(exc_info.value)

def test_elitism_size_valid():
    ga = setup_genetic_algorithm()
    try:
        ga.elitism_size = 0  # Boundary condition at zero
        ga.elitism_size = 10  # A positive value
    except Exception as e:
        pytest.fail(f"Setting elitism_size should not raise any exception, but raised {e}")

def test_elitism_size_negative():
    ga = setup_genetic_algorithm()
    with pytest.raises(GaException) as exc_info:
        ga.elitism_size = -1  # This should raise an error because it's negative
    assert "Elitism size cannot be negative" in str(exc_info.value)

def test_mutator_operator_valid():
    ga = setup_genetic_algorithm()
    mutator = IntegerMutator()
    try:
        ga.mutator_operator = mutator
        assert ga.mutator_operator == mutator, "Mutator operator should have been set to the provided instance"
    except Exception as e:
        pytest.fail(f"No exception should be raised for valid inputs: {e}")

def test_mutator_operator_invalid_type():
    ga = setup_genetic_algorithm()
    with pytest.raises(GaException) as exc_info:
        ga.mutator_operator = "not a mutator operator"  # Not an instance of MutatorOperator
    assert "Mutator operator must be an instance of MutatorOperator" in str(exc_info.value)

def test_crossover_operator_valid():
    ga = setup_genetic_algorithm()
    crossover = OnePointCrossover()
    try:
        ga.crossover_operator = crossover
        assert ga.crossover_operator == crossover, "Crossover operator should have been set to the provided instance"
    except Exception as e:
        pytest.fail(f"No exception should be raised for valid inputs: {e}")

def test_crossover_operator_invalid_type():
    ga = setup_genetic_algorithm()
    with pytest.raises(GaException) as exc_info:
        ga.crossover_operator = "not a crossover operator"  # Not an instance of CrossoverOperator
    assert "Crossover operator must be an instance of CrossoverOperator" in str(exc_info.value)

def tracker(g, r):
    pass
def test_set_generation_tracker_valid():
    ga = setup_genetic_algorithm()
    ga.set_generation_tracker(tracker)
    assert ga._generation_tracker is tracker, "Generation tracker should be set correctly"


def test_set_generation_tracker_none():
    ga = setup_genetic_algorithm()
    with pytest.raises(ValueError) as exc_info:
        ga.set_generation_tracker(None)
    assert "Object cannot be None" in str(exc_info.value), "Should raise ValueError when tracker is None"

def stop_condition(p):
    pass
def test_set_custom_stop_condition_valid():
    ga = setup_genetic_algorithm()
    ga.set_custom_stop_condition(stop_condition)
    assert stop_condition in ga._stop_conditions, "Custom stop condition should be added correctly"

def test_set_custom_stop_condition_none():
    ga = setup_genetic_algorithm()
    with pytest.raises(ValueError) as exc_info:
        ga.set_custom_stop_condition(None)
    assert "Object cannot be None" in str(exc_info.value), "Should raise ValueError when stop condition is None"

############################Genetic Configuration######################################

def test_default_parent_selector():
    config = GeneticConfiguration()
    assert isinstance(config._parent_selector, RouletteSelector), "Default parent selector should be a RouletteSelector"

def test_valid_custom_parent_selector():
    custom_selector = RankSelector()
    config = GeneticConfiguration(parent_selector=custom_selector)
    assert config._parent_selector is custom_selector, "Custom parent selector should be set correctly"

def test_invalid_parent_selector_type():
    config = GeneticConfiguration()
    with pytest.raises(GaException) as exc_info:
        # Assume String is not a valid ParentSelector
        config = GeneticConfiguration(parent_selector="invalid_selector")
    assert "Parent Selector operator must be an instance of ParentSelector" in str(exc_info.value)

def test_default_crossover_operator():
    config = GeneticConfiguration()
    assert isinstance(config._crossover_operator, OnePointCrossover), "Default crossover operator should be a OnePointCrossover"

def test_valid_custom_crossover_operator():
    custom_operator = TwoPointCrossover()
    config = GeneticConfiguration(crossover_operator=custom_operator)
    assert config._crossover_operator is custom_operator, "Custom crossover operator should be set correctly"

def test_invalid_crossover_operator_type():
    with pytest.raises(GaException) as exc_info:
        # Assume an integer is not a valid CrossoverOperator
        config = GeneticConfiguration(crossover_operator=123)
    assert "Crossover operator must be an instance of CrossoverOperator" in str(exc_info.value), "Should raise GaException for non-CrossoverOperator types"

def test_default_mutator_operator():
    config = GeneticConfiguration()
    assert isinstance(config._mutator_operator, GaussianMutator), "Default mutator operator should be a GaussianMutator"

def test_valid_custom_mutator_operator():
    custom_operator = InversionMutator()
    config = GeneticConfiguration(mutator_operator=custom_operator)
    assert config._mutator_operator is custom_operator, "Custom mutator operator should be set correctly"

def test_invalid_mutator_operator_type():
    with pytest.raises(GaException) as exc_info:
        # Assume a string is not a valid MutatorOperator
        config = GeneticConfiguration(mutator_operator="invalid_operator")
    assert "Mutator operator must be an instance of MutatorOperator" in str(exc_info.value), "Should raise GaException for non-MutatorOperator types"


