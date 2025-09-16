"""Base interface for M1 submodule implementations."""

from abc import ABC, abstractmethod


class M1SubModule(ABC):
    """Abstract base class for M1 submodule implementations.

    Defines the interface that all M1 implementations must follow.
    """

    @abstractmethod
    def connect(self, source_population):
        """Connect source population to this M1 submodule.

        Args:
            source_population: The input population to connect to this module

        Returns:
            Nothing
        """
        pass

    @abstractmethod
    def get_output_pops(self):
        """Get the output populations from this M1 submodule.

        Returns:
            tuple: Output populations (one pos, one neg)
        """
        pass
