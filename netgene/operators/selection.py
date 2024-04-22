from abc import ABC, abstractmethod
from netgene.exception import SelectionException
import random

from netgene.ga import Individual


class ParentSelector(ABC):

    def __init__(self):
        self._incest_prevention = True

    @property
    def incest_prevention(self) -> bool:
        return self._incest_prevention

    @incest_prevention.setter
    def incest_prevention(self, value: bool):
        self._incest_prevention = value

    def select_parents(self, population) -> tuple:
        first_individual = self.select(population)
        if self._incest_prevention:
            select_from = population.clone()
            select_from.remove_individual(first_individual)
            second_individual = self.select(select_from)
        else:
            second_individual = self.select(population)

        return first_individual, second_individual

    @abstractmethod
    def select(self, population) -> 'Individual':
        pass

class RandomSelector(ParentSelector):

    def select(self, population) -> 'Individual':
        if len(population) == 0:
            raise SelectionException("Population size is 0! Cannot select parents.")
        return population.get_random_individual()

class TournamentSelector(ParentSelector):

    def __init__(self, tournament_size=1):
        super().__init__()
        if tournament_size < 1:
            raise SelectionException("Tournament size cannot be lower than 1.")
        self._tournament_size = tournament_size

    @property
    def tournament_size(self):
        return self._tournament_size

    @tournament_size.setter
    def tournament_size(self, tournament_size):
        self._tournament_size = tournament_size

    def select(self, population) -> 'Individual':
        if len(population) == 0:
            raise SelectionException("Population size is 0! Cannot select parents.")
        if len(population) <= self.tournament_size:
            raise SelectionException("Tournament size cannot be equal or higher than the population size!")

        select_from = population.clone()
        random.shuffle(select_from)
        selected = max(select_from[:self.tournament_size], key=lambda x: x.fitness)
        return selected


class RankSelector(ParentSelector):

    def select(self, population) -> 'Individual':
        spin_wheel = 0
        population_rank = len(population) * (len(population) + 1) // 2
        roulette_wheel_position = random.randint(0, population_rank - 1)

        for i in range(len(population), 0, -1):
            spin_wheel += i
            if spin_wheel > roulette_wheel_position:
                return population[len(population) - i]

        #return None  # Unreachable code to make the compiler happy

class RouletteSelector(ParentSelector):

    def select(self, population) -> 'Individual':
        spin_wheel = 0.0
        roulette_wheel_position = random.random() * population.fitness
        for i in range(len(population)):
            spin_wheel += population[i].fitness
            if spin_wheel >= roulette_wheel_position:
                return population[i]

class CompetitionSelector(ParentSelector):

    def select(self, population) -> 'Individual':
        select_from = population.clone()
        ind1 = select_from.get_random_individual()
        select_from.remove_individual(ind1)
        ind2 = select_from.get_random_individual()

        if ind1.fitness > ind2.fitness:
            return ind1
        else:
            return ind2

# from netgene.chromosome import IntegerChromosome
# from netgene.ga import Individual, Population
#
#
# ind1 = Individual(IntegerChromosome(3))
# ind2 = Individual(IntegerChromosome(3))
# ind3 = Individual(IntegerChromosome(3))
# ind4 = Individual(IntegerChromosome(3))
# ind5 = Individual(IntegerChromosome(3))
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
# print(population)
#
# selector = CompetitionSelector()
#
#
# indv1, indv2 = selector.select_parents(population)
#
# print("##################################################")
# print(indv1)
# print(indv2)
