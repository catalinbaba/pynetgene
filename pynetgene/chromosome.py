import random
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic

from pynetgene.exception import GaException
from pynetgene.gene import BitGene
from pynetgene.gene import IntegerGene
from pynetgene.gene import FloatGene
import copy
import numbers

# Defining a Gene type variable
#G = TypeVar("G", bound="Gene")
# Define a type variable that can be any subclass of `numbers.Number`
N = TypeVar('N', bound=numbers.Number)

class Chromosome(ABC):

    def __init__(self):
        self._genes = []

    def get_gene(self, index: int):
        return self._genes[index]

    def length(self) -> int:
        return len(self._genes)

    def contains(self, gene) -> bool:
        return gene in self._genes

    @abstractmethod
    def add_gene(self, gene):
        pass

    @abstractmethod
    def set_gene(self, index: int, gene):
        pass

    @abstractmethod
    def insert_gene(self, index: int, gene):
        pass

    @abstractmethod
    def copy(self) -> 'Chromosome':
        pass

    @abstractmethod
    def to_list(self) -> List[bool]:
        pass

    def __iter__(self):
        return iter(self._genes)

    def __getitem__(self, index):
        return self._genes[index]

    def __setitem__(self, index, value):
        self._genes[index] = value

    def __len__(self):
        return len(self._genes)

    def __str__(self):  # Centralized the __str__ method
        return f"Chromosome:\n" + "\n".join([f"Gene {i} = {gene}" for i, gene in enumerate(self._genes)])


class BitChromosome(Chromosome):
    def __init__(self, size=None):
        super().__init__()

        if size is not None:
            self._genes = [random.choice([True, False]) for _ in range(size)]
        else:
            self._genes = []

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, genes: []):
        self._genes = genes

    def add_gene(self, gene: bool):
        if gene is not None and not isinstance(gene, bool):
            raise ValueError("allele must be a boolean value (True or False)")
        self._genes.append(gene)

    def set_gene(self, index: int, gene: bool):
        if gene is not None and not isinstance(gene, bool):
            raise ValueError("allele must be a boolean value (True or False)")
        self._genes[index] = gene

    def insert_gene(self, index: int, gene: bool):
        if not isinstance(gene, BitGene):
            raise ValueError("Only BitGene can be added to a BitChromosome")
        self._genes.insert(index, gene)

    def to_list(self) -> List[bool]:
        return self._genes

    def copy(self) -> 'BitChromosome':
        new_chromosome = BitChromosome()
        # Since the list contains only boolean values, a shallow copy is sufficient
        new_chromosome._genes = self._genes.copy()
        return new_chromosome


class NumericChromosome(Chromosome, Generic[N], ABC):
    def average(self) -> "NumericChromosome[N]":
        raise NotImplementedError


class IntegerChromosome(NumericChromosome):
    def __init__(self, size=None, min_range=None, max_range=None):
        super().__init__()

        if size is not None and min_range is not None and max_range is not None:
            self._genes = [random.randint(min_range, max_range) for _ in range(size)]
        elif size is not None:
            self._genes = [random.randint(-2 ** 31, 2 ** 31 - 1) for _ in range(size)]
        else:
            self._genes = []

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, genes: List[N]):
        self._genes = genes

    def add_gene(self, gene: N):
        if gene is not None and not isinstance(gene, int):
            raise ValueError("allele must be of type int1")
        self._genes.append(gene)

    def set_gene(self, index: int, gene: N):
        if gene is not None and not isinstance(gene, int):
            raise ValueError("allele must be of type int2")
        self._genes[index] = gene

    def insert_gene(self, index: int, gene: N):
        if gene is not None and not isinstance(gene, int):
            raise ValueError("allele must be of type int3")
        self._genes.insert(index, gene)

    def to_list(self) -> List[int]:
        return self._genes

    def copy(self) -> 'IntegerChromosome':
        new_chromosome = IntegerChromosome()
        # Since the list contains only boolean values, a shallow copy is sufficient
        new_chromosome._genes = self._genes.copy()
        return new_chromosome

    def average(self) -> float:
        if not self._genes:
            raise ValueError("Chromosome is empty, cannot calculate average.")

        return sum(self._genes) / len(self._genes)


class FloatChromosome(NumericChromosome[N]):
    def __init__(self, size=None, min_range=None, max_range=None):
        super().__init__()

        if size is not None and min_range is not None and max_range is not None:
            self._genes = [random.uniform(min_range, max_range) for _ in range(size)]
        elif size is not None:
            self._genes = [random.gauss(0, 1) for _ in range(size)]
        else:
            self._genes = []

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, genes: List[N]):
        self._genes = genes

    def add_gene(self, gene: N):
        if not isinstance(gene, (float, int)):  # allow int because they can be implicitly converted to float
            raise ValueError("allele must be a float value")
        self._genes.append(gene)

    def set_gene(self, index: int, gene: N):
        if not isinstance(gene, (float, int)):  # allow int because they can be implicitly converted to float
            raise ValueError("allele must be a float value")
        self._genes[index] = gene

    def insert_gene(self, index: int, gene: FloatGene):
        if not isinstance(gene, (float, int)):  # allow int because they can be implicitly converted to float
            raise ValueError("allele must be a float value")
        self._genes.insert(index, gene)

    def to_list(self) -> List[float]:
        return self._genes

    def copy(self) -> 'FloatChromosome':
        new_chromosome = FloatChromosome()
        # Since the list contains only boolean values, a shallow copy is sufficient
        new_chromosome._genes = self._genes.copy()
        return new_chromosome

    def average(self) -> float:
        if not self._genes:
            raise ValueError("Chromosome is empty, cannot calculate average.")

        return sum(self._genes) / len(self._genes)


class PermutationChromosome(IntegerChromosome):
    def __init__(self, size=None, start=0):
        super().__init__()

        if size is not None and start != 0:
            self._genes = [i + start for i in range(size)]
            random.shuffle(self._genes)
        elif size is not None:
            self._genes = [i for i in range(size)]
            random.shuffle(self._genes)
        else:
            self._genes = []

    def set_gene(self, index: int, gene: N):
        if not isinstance(gene, int):
            raise ValueError("Only int allele values can be added to a PermutationChromosome")
        if self.contains(gene):
            raise GaException(
                "Gene with the same allele value was already added to the chromosome. Values must not be repeated in "
                "a single chromosome.")
        self._genes[index] = gene

    def add_gene(self, gene: N):
        if not isinstance(gene, int):
            raise ValueError("Allele must be an int value")
        if self.contains(gene):
            raise GaException(
                "Gene with the same allele value was already added to the chromosome. Values must not be repeated in "
                "a single chromosome.")
        self._genes.append(gene)

    def insert_gene(self, index: int, gene: N):
        if not isinstance(gene, int):
            raise ValueError("Allele must be an int value")
        if self.contains(gene):
            raise GaException(
                "Gene with the same allele value was already added to the chromosome. Values must not be repeated in "
                "a single chromosome.")
        self._genes.insert(index, gene)

    def copy(self) -> 'PermutationChromosome':
        new_chromosome = PermutationChromosome()
        # Since the list contains only boolean values, a shallow copy is sufficient
        new_chromosome._genes = self._genes.copy()
        return new_chromosome
