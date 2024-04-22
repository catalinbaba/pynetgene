from abc import ABC, abstractmethod
import random
from functools import total_ordering


class Gene(ABC):
    """
    Represents a general gene with a specific allele type.

    :author: Catalin Baba
    """

    @abstractmethod
    def copy(self):
        """
        Creates a copy of the gene.

        :return: copy of the gene
        """
        pass


class BitGene(Gene):
    """
    Implementation of a BitGene, which is a gene with a Boolean allele.

    :author: Catalin Baba
    """

    def __init__(self, allele=None):
        """
        Initializes a BitGene with the given allele or a random Boolean value if none is provided.

        :param allele: Boolean allele value. Default is random.
        """
        if allele is not None and not isinstance(allele, bool):
            raise ValueError("allele must be a boolean value (True or False)")

        if allele is not None:
            self._allele = allele
        else:
            self._allele = random.choice([True, False])
            # self.allele = allele if allele is not None else random.choice([True, False])

    @property
    def allele(self) -> bool:
        return self._allele

    def __str__(self):
        """
        Returns a string representation of the BitGene.

        :return: string representation of the allele
        """
        return str(self._allele)

    def __eq__(self, another_gene):
        """
        Checks equality between this BitGene and another based on their allele values.

        :param another_gene: another BitGene to compare with
        :return: True if the alleles are the same, False otherwise
        """
        return self._allele == another_gene.allele

    def copy(self):
        """
        Creates a copy of this BitGene.

        :return: copy of this BitGene
        """
        return BitGene(self._allele)


class NumericGene(Gene, ABC):
    """
    Represents a generic gene for numerical values.

    :author: Catalin Baba
    """

    @abstractmethod
    def average(self, gene):
        """
        Compute the average of this gene with another gene.

        :param gene: Another gene to compute the average with.
        :return: The average gene.
        """
        pass


@total_ordering
class IntegerGene(NumericGene):
    """
    Implementation of a IntegerGene, which is a gene with an Integer allele.

    :author: Catalin Baba
    """

    def __init__(self, allele=None, min_range=None, max_range=None):
        """
        Initializes an IntegerGene with the given allele or a random integer value if none is provided.
        If min_range and max_range are provided, the random value is drawn between these bounds.

        :param allele: Integer allele value. Default is random.
        :param min_range: Minimum value for random allele. Only used if allele is None.
        :param max_range: Maximum value for random allele. Only used if allele is None.
        """

        if allele is not None and not isinstance(allele, int):
            raise ValueError("allele must be of type int")
        if min_range is not None and not isinstance(min_range, int):
            raise ValueError("min_range must be of type int")
        if max_range is not None and not isinstance(max_range, int):
            raise ValueError("max_range must be of type int")

        if allele is not None:
            self._allele = allele
        elif min_range is not None and max_range is not None:
            self._allele = random.randint(min_range, max_range - 1)
        else:
            self._allele = random.randint(-2 ** 31, 2 ** 31 - 1)  # assuming 32-bit integers

    @property
    def allele(self) -> int:
        return self._allele

    def __str__(self):
        """
        Returns a string representation of the IntegerGene.

        :return: string representation of the allele
        """
        return str(self.allele)

    def __lt__(self, other):
        """
        Compares this IntegerGene with another based on their allele values.

        :param other: another IntegerGene to compare with
        :return: True if this IntegerGene's allele is less than the other, False otherwise
        """
        if isinstance(other, IntegerGene):
            return self._allele < other.allele
        return False

    def __eq__(self, other):
        """
        Checks equality between this IntegerGene and another based on their allele values.

        :param other: another IntegerGene to compare with
        :return: True if the alleles are the same, False otherwise
        """
        if isinstance(other, IntegerGene):
            return self._allele == other.allele
        return False

    def copy(self):
        """
        Creates a copy of this IntegerGene.

        :return: copy of this IntegerGene
        """
        return IntegerGene(self._allele)

    def average(self, gene):
        """
        Computes the average of this gene and another gene.

        :param gene: Another IntegerGene to compute the average with.
        :return: The average gene.
        """
        return IntegerGene((self._allele + gene.allele) // 2)


@total_ordering
class FloatGene(NumericGene):
    """
    Implementation of a FloatGene, which is a gene with a float allele.

    :author: Catalin Baba
    """

    def __init__(self, allele=None, min_range=None, max_range=None):
        """
        Initializes a FloatGene with the given allele or a random float value if none is provided.
        If min_range and max_range are provided, the random value is drawn between these bounds.

        :param allele: float allele value. Default is random.
        :param min_range: Minimum value for random allele. Only used if allele is None.
        :param max_range: Maximum value for random allele. Only used if allele is None.
        """
        if allele is not None:
            if not isinstance(allele, (float, int)):  # allow int because they can be implicitly converted to float
                raise ValueError("allele must be a float value")
            self._allele = float(allele)
        elif min_range is not None and max_range is not None:
            if not (isinstance(min_range, (float, int)) and isinstance(max_range, (float, int))):
                raise ValueError("min_range and max_range must be float values")
            self._allele = random.uniform(min_range, max_range)
        else:
            self._allele = random.gauss(0, 1)  # A Gaussian distribution around 0

    @property
    def allele(self):
        return self._allele

    def __str__(self):
        """
        Returns a string representation of the FloatGene.

        :return: string representation of the allele
        """
        return str(self._allele)

    def __lt__(self, another_gene):
        """
        Compares this FloatGene with another based on their allele values.

        :param another_gene: another FloatGene to compare with
        :return: True if this FloatGene's allele is less than the other, False otherwise
        """
        return self._allele < another_gene.allele

    def __eq__(self, another_gene):
        """
        Checks equality between this FloatGene and another based on their allele values.

        :param another_gene: another FloatGene to compare with
        :return: True if the alleles are the same, False otherwise
        """
        return self._allele == another_gene.allele

    def copy(self):
        """
        Creates a copy of this FloatGene.

        :return: copy of this FloatGene
        """
        return FloatGene(self.allele)

    def average(self, gene):
        """
        Computes the average of this gene and another gene.

        :param gene: Another FloatGene to compute the average with.
        :return: The average gene.
        """
        return FloatGene((self.allele + gene.allele) / 2)
