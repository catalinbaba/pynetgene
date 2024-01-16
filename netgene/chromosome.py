import random
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic

from netgene.exception import GaException
from netgene.gene import BitGene
from netgene.gene import IntegerGene
from netgene.gene import FloatGene
import copy

# Defining a Gene type variable
G = TypeVar("G", bound="Gene")


class Chromosome(Generic[G], ABC):

    def __init__(self):
        self._chromosome: List[G] = []

    def get_gene(self, index: int) -> G:
        return self._chromosome[index]

    def length(self) -> int:
        return len(self._chromosome)

    def contains(self, gene: G) -> bool:
        return gene in self._chromosome

    @abstractmethod
    def add_gene(self, gene: G):
        pass

    @abstractmethod
    def set_gene(self, index: int, gene: G):
        pass

    @abstractmethod
    def insert_gene(self, index: int, gene: G):
        pass

    @abstractmethod
    def copy(self) -> 'Chromosome':
        pass

    @abstractmethod
    def to_list(self) -> List[bool]:
        pass

    def __iter__(self):
        return iter(self._chromosome)

    def __getitem__(self, index):
        return self._chromosome[index]

    def __setitem__(self, index, value):
        self._chromosome[index] = value

    def __len__(self):
        return len(self._chromosome)

    def __str__(self):  # Centralized the __str__ method
        return f"Chromosome:\n" + "\n".join([f"Gene {i} = {gene}" for i, gene in enumerate(self._chromosome)])


class BitChromosome(Chromosome[BitGene]):
    def __init__(self, size=None):
        super().__init__()

        if size is not None:
            self._chromosome = [BitGene() for _ in range(size)]
        else:
            self._chromosome = []

    @property
    def chromosome(self):
        return self._chromosome

    @chromosome.setter
    def chromosome(self, chromosome: List[BitGene]):
        self._chromosome = chromosome

    def add_gene(self, gene: BitGene):
        if not isinstance(gene, BitGene):
            raise ValueError("Only BitGene can be added to a BitChromosome")
        self._chromosome.append(gene)

    def set_gene(self, index: int, gene: BitGene):
        if not isinstance(gene, BitGene):
            raise ValueError("Only BitGene can be added to a BitChromosome")
        self._chromosome[index] = gene

    def insert_gene(self, index: int, gene: BitGene):
        if not isinstance(gene, BitGene):
            raise ValueError("Only BitGene can be added to a BitChromosome")
        self._chromosome.insert(index, gene)

    def to_list(self) -> List[bool]:
        return [gene.allele for gene in self._chromosome]

    def copy(self) -> 'BitChromosome':
        new_chromosome = BitChromosome()
        new_chromosome._chromosome = copy.deepcopy(self._chromosome)
        return new_chromosome


class NumericChromosome(Chromosome, Generic[G], ABC):
    def average(self, gene: "NumericChromosome[G]") -> "NumericChromosome[G]":
        raise NotImplementedError


class IntegerChromosome(NumericChromosome[IntegerGene]):
    def __init__(self, size=None, min_range=None, max_range=None):
        super().__init__()

        if size is not None and min_range is not None and max_range is not None:
            self._chromosome = [IntegerGene(min_range=min_range, max_range=max_range) for _ in range(size)]
        elif size is not None:
            self._chromosome = [IntegerGene() for _ in range(size)]
        else:
            self._chromosome = []

    @property
    def chromosome(self):
        return self._chromosome

    @chromosome.setter
    def chromosome(self, chromosome: List[IntegerGene]):
        self._chromosome = chromosome

    def add_gene(self, gene: IntegerGene):
        if not isinstance(gene, IntegerGene):
            raise ValueError("Only IntegerGene can be added to a IntegerChromosome")
        self._chromosome.append(gene)

    def set_gene(self, index: int, gene: IntegerGene):
        if not isinstance(gene, IntegerGene):
            raise ValueError("Only IntegerGene can be added to a IntegerChromosome")
        self._chromosome[index] = gene

    def insert_gene(self, index: int, gene: IntegerGene):
        if not isinstance(gene, IntegerGene):
            raise ValueError("Only IntegerGene can be added to a IntegerChromosome")
        self._chromosome.insert(index, gene)

    def to_list(self) -> List[int]:
        return [gene.get_allele() for gene in self._chromosome]

    def copy(self) -> 'IntegerChromosome':
        new_chromosome = IntegerChromosome()
        new_chromosome._chromosome = copy.deepcopy(self._chromosome)
        return new_chromosome

    def average(self, that_chromosome: "IntegerChromosome") -> "IntegerChromosome":
        average_chromosome = IntegerChromosome()
        for i in range(len(self._chromosome)):
            average_chromosome.add_gene(self._chromosome[i].average(that_chromosome.get_gene(i)))
        return average_chromosome


class FloatChromosome(NumericChromosome[FloatGene]):
    def __init__(self, size=None, min_range=None, max_range=None):
        super().__init__()

        if size is not None and min_range is not None and max_range is not None:
            self._chromosome = [FloatGene(min_range=min_range, max_range=max_range) for _ in range(size)]
        elif size is not None:
            self._chromosome = [FloatGene() for _ in range(size)]
        else:
            self._chromosome = []

    @property
    def chromosome(self):
        return self._chromosome

    @chromosome.setter
    def chromosome(self, chromosome: List[FloatGene]):
        self._chromosome = chromosome

    def add_gene(self, gene: FloatGene):
        if not isinstance(gene, FloatGene):
            raise ValueError("Only FloatGene can be added to a FloatChromosome")
        self._chromosome.append(gene)

    def set_gene(self, index: int, gene: FloatGene):
        if not isinstance(gene, FloatGene):
            raise ValueError("Only FloatGene can be added to a FloatChromosome")
        self._chromosome[index] = gene

    def insert_gene(self, index: int, gene: FloatGene):
        if not isinstance(gene, FloatGene):
            raise ValueError("Only FloatGene can be added to a FloatChromosome")
        self._chromosome.insert(index, gene)

    def to_list(self) -> List[float]:
        return [gene.get_allele() for gene in self._chromosome]

    def copy(self) -> 'FloatChromosome':
        new_chromosome = FloatChromosome()
        new_chromosome._chromosome = copy.deepcopy(self._chromosome)
        return new_chromosome

    def average(self, that_chromosome: "FloatChromosome") -> "FloatChromosome":
        average_chromosome = FloatChromosome()
        for i in range(len(self._chromosome)):
            average_chromosome.add_gene(self._chromosome[i].average(that_chromosome.get_gene(i)))
        return average_chromosome


class PermutationChromosome(IntegerChromosome):
    def __init__(self, size=None, start=0):
        super().__init__()

        if size is not None and start != 0:
            self._chromosome = [IntegerGene(i + start) for i in range(size)]
            random.shuffle(self._chromosome)
        elif size is not None:
            self._chromosome = [IntegerGene(i) for i in range(size)]
            random.shuffle(self._chromosome)
        else:
            self._chromosome = []

    def set_gene(self, index: int, gene: IntegerGene):
        if not isinstance(gene, IntegerGene):
            raise ValueError("Only IntegerGene can be added to a PermutationChromosome")
        if self.contains(gene):
            raise GaException(
                "Gene with the same allele value was already added to the chromosome. Values must not be repeated in "
                "a single chromosome.")
        self._chromosome[index] = gene

    def add_gene(self, gene: IntegerGene):
        if not isinstance(gene, IntegerGene):
            raise ValueError("Only IntegerGene can be added to a PermutationChromosome")
        if self.contains(gene):
            raise GaException(
                "Gene with the same allele value was already added to the chromosome. Values must not be repeated in "
                "a single chromosome.")
        self._chromosome.append(gene)

    def insert_gene(self, index: int, gene: IntegerGene):
        if not isinstance(gene, IntegerGene):
            raise ValueError("Only IntegerGene can be added to a PermutationChromosome")
        if self.contains(gene):
            raise GaException(
                "Gene with the same allele value was already added to the chromosome. Values must not be repeated in "
                "a single chromosome.")
        self._chromosome.insert(index, gene)

    def copy(self) -> 'PermutationChromosome':
        new_chromosome = PermutationChromosome()
        new_chromosome._chromosome = copy.deepcopy(self._chromosome)
        return new_chromosome
