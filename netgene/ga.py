import concurrent.futures
import threading
import time

from netgene.core import *
from netgene.operators.crossover import OnePointCrossover
from netgene.exception import *
from netgene.operators.mutator import *
from netgene.operators.selection import *
from netgene.utils import ConsolePrinter, TaskExecutor
import logging
from concurrent.futures import ThreadPoolExecutor


class GeneticAlgorithm:
    def __init__(self, parent_selector, crossover_operator,
                 mutator_operator, crossover_rate, mutation_rate,
                 elitism, elitism_size, max_generation,
                 target_fitness, skip_crossover,
                 skip_mutation, n_threads,
                 clock, printer):
        self._parent_selector = parent_selector
        self._crossover_operator = crossover_operator
        self._mutator_operator = mutator_operator
        self._mutation_rate = mutation_rate
        self._mutator_operator.mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._elitism = elitism
        self._elitism_size = elitism_size
        self._max_generation = max_generation
        self._n_threads = n_threads
        self._target_fitness = verify_is_not_null(target_fitness)
        self._skip_crossover = skip_crossover
        self._skip_mutation = skip_mutation
        self._clock = verify_is_not_null(clock)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_threads)
        self._printer = printer
        self.lock = threading.Lock()

        self._stop_conditions = []
        self._fitness_function = None
        self._generation_tracker = None
        self._population = None

        max_generation_stop = lambda population: population.generation == self._max_generation
        target_fitness_stop = lambda population: self._target_fitness <= population.get_best_individual().fitness
        self._stop_conditions.append(max_generation_stop)
        self._stop_conditions.append(target_fitness_stop)

    @property
    def mutation_rate(self):
        return self._mutation_rate
    @mutation_rate.setter
    def mutation_rate(self, new_mutation_rate):
        if 0.0 <= new_mutation_rate <= 1.0:
            self._mutation_rate = new_mutation_rate
        else:
            raise ValueError("Mutation rate must be between 0.0 and 1.0")

    @property
    def crossover_rate(self):
        return self._crossover_rate

    @crossover_rate.setter
    def crossover_rate(self, new_crossover_rate):
        if 0.0 <= new_crossover_rate <= 1.0:
            self._mutation_rate = new_crossover_rate
        else:
            raise ValueError("Crossover rate must be between 0.0 and 1.0")

    @property
    def elitism(self):
        return self._elitism

    @elitism.setter
    def elitism(self, new_elitism):
        # Check if elitism is True or False
        if isinstance(new_elitism, bool):
            self._elitism = new_elitism
        else:
            raise ValueError("Elitism must be either True or False")

    @property
    def elitism_size(self):
        return self._elitism_size

    @elitism_size.setter
    def elitism_size(self, new_elitism_size):
        # Check if elitism is True or False
        if new_elitism_size < 0:
            raise GaException("Elitism size cannot be negative")
        self._elitism_size = new_elitism_size

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, new_population):
        self._population = new_population

    @property
    def mutator_operator(self):
        return self._mutator_operator

    @mutator_operator.setter
    def mutator_operator(self, new_mutator_operator):
        if new_mutator_operator is not None:
            self._mutator_operator = new_mutator_operator
        else:
            raise GaException("Mutator Operator cannot be None")

    @property
    def crossover_operator(self):
        return self._crossover_operator

    @crossover_operator.setter
    def crossover_operator(self, new_crossover_operator):
        if new_crossover_operator is not None:
            self._crossover_operator = new_crossover_operator
        else:
            raise GaException("Crossover Operator cannot be None")

    def evolve(self, population, fitness_function):
        self._population = population
        self._fitness_function = fitness_function

        while not self._has_reached_stop_condition():  # as long as the stop condition is not reached
            result = self._evolve_generation()                  # evolve
            if self._generation_tracker is not None:
                self._generation_tracker(self, result)
        self._executor.shutdown()                      #shut down the executor - free up the system resources used by executor


    def _evolve_generation(self):
        generation_number = self._population.generation     #generation number will be incremented after population evolves

        limit = len(self._population) - self._elitism_size if self._elitism else len(self._population)  #set limit -> take in consideration the elitism_size

        # Elitism: directly copy the best individuals to the new population
        if self._elitism:
            # Get the elite individuals and create a new Population object
            elite_individuals = self._population[:self._elitism_size]
            new_population = Population(elite_individuals)
        else:
            # Create an empty Population object
            new_population = Population([])

        task_runner = TaskExecutor()

        evolution_duration = task_runner.run_task(self._evolution_task, self._executor, limit, new_population)

        evaluation_duration = task_runner.run_task(self._calculate_population_fitness, self._executor)

        self._population.sort()

        self._population.generation = generation_number + 1

        print("!!!!!!!!!!population size: ", len(self._population))

        return GenerationResult(evolution_duration, evaluation_duration, self._population.get_best_individual(), self._population.generation)

    def _evolution_task(self, limit, new_population):
        individual_stream = []

        if not self._skip_crossover:
            threads = []
            # Generate individuals with crossover using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(self._n_threads) as executor:
                for _ in range(limit):
                    if len(individual_stream) >= limit:
                        break  # Stop if we have reached the limit

                    # Submit a crossover operation to the thread pool
                    future = executor.submit(self.crossover_thread, individual_stream, limit)
                    threads.append(future)

                # Wait for all futures to complete
                concurrent.futures.wait(threads)
        else:
            # If crossover is skipped, take individuals directly from the current population (up to the limit)
            individual_stream = list(self._population)[:limit]

        if not self._skip_mutation:
            # Apply mutation to each individual in the stream
            with concurrent.futures.ThreadPoolExecutor(self._n_threads) as executor:
                for individual in individual_stream:
                    executor.submit(self._mutate, individual)

        # Add individuals from the stream to the new population
        for individual in individual_stream:
            new_population.add_individual(individual)

        # Update the current population with the new population
        self._population = new_population

    def crossover_thread(self, individual_stream, limit):
        couple = self._parents_supplier()
        if couple:
            offspring = self._crossover(couple)

            # Add offspring to the stream, but respect the limit
            with self.lock:
                for child in offspring.get_individuals():
                    if len(individual_stream) < limit:
                        individual_stream.append(child)

    def _parents_supplier(self):
        try:
             # Attempt to select parents using the parent selector
            parents = self._parent_selector.select_parents(self._population)
            # Return the selected parents
            return parents
        except Exception as e:
            # If an exception (SelectionException) occurs during parent selection, log the error
            logging.exception("Error in parent selection", e)
            # Return None to indicate that the parent selection failed
            return None

    def _crossover(self, couple):
        # Unpack the parents from the couple
        first_parent, second_parent = couple

        # Generate a random value
        random_value = random.random()

        # Create an empty Offspring object
        offspring = Offspring()

        if self._crossover_rate > random_value:
            try:
                # Perform crossover using the crossover operator
                offspring = self._crossover_operator.recombine(first_parent, second_parent)
            except CrossoverException as e:
                # Handle exceptions that might occur during crossover
                logging.exception("Exception occurred crossover", e)
        else:
            # If crossover is skipped, create a child as a copy of each parent
            first_offspring_chromosome = first_parent.chromosome.copy()
            offspring.add_offspring(Individual(first_offspring_chromosome))

            if not self._crossover_operator.has_single_offspring():
                second_offspring_chromosome = second_parent.chromosome.copy()
                offspring.add_offspring(Individual(second_offspring_chromosome))

        return offspring

    def _mutate(self, individual):
        try:
            # Applying mutation to the individual using the mutator operator
            self._mutator_operator.mutate(individual)
        except MutatorException as e:
            # Handling exceptions that might occur during mutation
            logging.exception("Exception occurred during mutation", e)

    def _calculate_population_fitness(self):
        # for individual in self._population:
        #     self._fitness_function(individual)
        # Using ThreadPoolExecutor to parallelize fitness calculation
        with ThreadPoolExecutor() as executor:
            executor.map(self._fitness_function, self._population)

    def _has_reached_stop_condition(self):
        return any(stop_condition(self._population) for stop_condition in self._stop_conditions)

    def set_generation_tracker(self, tracker):
        self._generation_tracker = tracker

    def set_custom_stop_condition(self, custom_stop_condition):
        self._stop_conditions.append(custom_stop_condition)



class GeneticConfiguration:

    def __init__(self, parent_selector=None, crossover_operator=None,
                 mutator_operator=None, crossover_rate=0.8, mutation_rate=0.05,
                 elitism=True, elitism_size=1, max_generation=float('inf'),
                 target_fitness=float('inf'), skip_crossover=False,
                 skip_mutation=False, n_threads=threading.active_count(),
                 clock=time.time, printer=None):
        self._parent_selector = parent_selector if parent_selector is not None else RouletteSelector()
        self._crossover_operator = crossover_operator if crossover_operator is not None else OnePointCrossover()
        self._mutator_operator = mutator_operator if mutator_operator is not None else GaussianMutator()
        if mutation_rate <= 0:
            raise GaException("Mutation rate value cannot be negative")
        if mutation_rate > 1.0:
            raise GaException("Mutation rate value cannot be higher than 1.0")
        self._mutation_rate = mutation_rate
        self._mutator_operator.mutation_rate = mutation_rate
        if crossover_rate <= 0:
            raise GaException("Crossover rate value cannot be negative")
        if crossover_rate > 1.0:
            raise GaException("Crossover rate value cannot be higher than 1.0")
        self._crossover_rate = crossover_rate
        if not isinstance(elitism, bool):
            raise GaException("Elitism must be a boolean value")
        self._elitism = elitism
        if elitism_size < 0:
            raise GaException("Elitism size cannot be negative")
        self._elitism_size = elitism_size
        if max_generation <= 0:
            raise GaException("Maximum generation cannot be lower than 1")
        self._max_generation = max_generation
        if n_threads < 1:
            raise GaException("Thread Pool Size cannot be lower than 1")
        self._n_threads = n_threads
        self._target_fitness = verify_is_not_null(target_fitness)
        if not isinstance(skip_crossover, bool):
            raise GaException("skip_crossover must be a boolean value")
        self._skip_crossover = skip_crossover
        if not isinstance(skip_mutation, bool):
            raise GaException("skip_mutation must be a boolean value")
        self._skip_mutation = skip_mutation
        self._clock = verify_is_not_null(clock)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_threads)
        self._printer = printer if printer is not None else ConsolePrinter()

    @property
    def parent_selector(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @parent_selector.setter
    def parent_selector(self, parent_selector):
        self._parent_selector = parent_selector if parent_selector is not None else RouletteSelector()

    @property
    def crossover_operator(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @crossover_operator.setter
    def crossover_operator(self, crossover_operator):
        self._crossover_operator = crossover_operator if crossover_operator is not None else OnePointCrossover()

    @property
    def mutator_operator(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @mutator_operator.setter
    def mutator_operator(self, mutator_operator):
        self._mutator_operator = mutator_operator if mutator_operator is not None else GaussianMutator()

    @property
    def crossover_rate(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @crossover_rate.setter
    def crossover_rate(self, crossover_rate):
        if crossover_rate <= 0:
            raise GaException("Crossover rate value cannot be negative")
        if crossover_rate > 1.0:
            raise GaException("Crossover rate value cannot be higher than 1.0")
        self._crossover_rate = crossover_rate

    @property
    def mutation_rate(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @mutation_rate.setter
    def mutation_rate(self, mutation_rate):
        if mutation_rate <= 0:
            raise GaException("Mutation rate value cannot be negative")
        if mutation_rate > 1.0:
            raise GaException("Mutation rate value cannot be higher than 1.0")
        self._mutation_rate = mutation_rate
        self._mutator_operator.mutation_rate = mutation_rate

    @property
    def elitism(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @elitism.setter
    def elitism(self, elitism):
        if not isinstance(elitism, bool):
            raise GaException("Elitism must be a boolean value")
        self._elitism = elitism

    @property
    def elitism_size(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @elitism_size.setter
    def elitism_size(self, elitism_size):
        if elitism_size < 0:
            raise GaException("Elitism size cannot be negative")
        self._elitism_size = elitism_size

    @property
    def max_generation(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @max_generation.setter
    def max_generation(self, max_generation):
        if max_generation <= 0:
            raise GaException("Maximum generation cannot be lower than 1")
        self._max_generation = max_generation

    @property
    def target_fitness(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @target_fitness.setter
    def target_fitness(self, target_fitness):
        self._target_fitness = verify_is_not_null(target_fitness)

    @property
    def skip_crossover(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @skip_crossover.setter
    def skip_crossover(self, skip_crossover):
        if not isinstance(skip_crossover, bool):
            raise GaException("skip_crossover must be a boolean value")
        self._skip_crossover = skip_crossover

    @property
    def skip_mutation(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @skip_mutation.setter
    def skip_mutation(self, skip_mutation):
        if not isinstance(skip_mutation, bool):
            raise GaException("skip_mutation must be a boolean value")
        self._skip_mutation = skip_mutation

    @property
    def n_threads(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @n_threads.setter
    def n_threads(self, n_threads):
        if n_threads < 1:
            raise GaException("Thread Pool Size cannot be lower than 1")
        self._n_threads = n_threads

    @property
    def clock(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @clock.setter
    def clock(self, clock):
        self._clock = verify_is_not_null(clock)

    @property
    def printer(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @printer.setter
    def printer(self, printer):
        self._printer = printer

    @property
    def executor(self):
        # This is a dummy getter to satisfy the @property decorator
        pass

    @executor.setter
    def executor(self, executor):
        self._executor = executor

    def get_algorithm(self):
        ga = GeneticAlgorithm(self._parent_selector, self._crossover_operator, self._mutator_operator,
                              self._crossover_rate, self._mutation_rate, self._elitism, self._elitism_size,
                              self._max_generation, self._target_fitness, self._skip_crossover, self._skip_mutation,
                              self._n_threads, self._clock, self._printer)
        return ga

class GenerationResult:
    def __init__(self, evolution_duration, evaluation_duration, individual, generation_number):
        self._evolution_duration = evolution_duration
        self._evaluation_duration = evaluation_duration
        self._individual = individual
        self._best_fitness = individual.fitness
        self._generation_number = generation_number
        self._generation_duration = evolution_duration + evaluation_duration

    @property
    def evolution_duration(self):
        return self._evolution_duration

    @property
    def evaluation_duration(self):
        return self._evaluation_duration

    @property
    def best_individual(self):
        return self._individual

    @property
    def best_fitness(self):
        return self._best_fitness

    @property
    def generation_number(self):
        return self._generation_number

    @property
    def generation_duration(self):
        return self._generation_duration


class GenerationTracker(ABC):
    """
    A tracker for generations in a genetic algorithm.

    This class should be subclassed by concrete implementations
    that track the progress of genetic algorithms.
    """

    @abstractmethod
    def track(self, genetic_algorithm, generation_result):
        """
        Track the given genetic algorithm and its generation result.

        Parameters:
        genetic_algorithm: An instance of a GeneticAlgorithm or subclass thereof.
        generation_result: The result of the current generation that needs to be tracked.
        """
        pass

def verify_is_not_null(obj):
    if obj is None:
        raise ValueError("Object cannot be None")
    return obj
