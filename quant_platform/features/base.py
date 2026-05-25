from abc import ABC, abstractmethod
import pandas as pd

class BaseSignal(ABC):
    """
    Abstract base class for all feature extraction and alpha signal generators.
    Any new custom signal or feature should inherit from this class and
    implement the abstract methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns a unique name identifier for the signal generator.
        """
        pass

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        Returns the list of database columns required to compute this signal.
        Example: ['close'] or ['high', 'low', 'close', 'volume']
        """
        pass

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the signal/features for a single ticker's historical dataframe.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing historical prices for one symbol.
            Guaranteed to contain `required_columns` and sorted chronologically.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the key fields ['symbol', 'date']
            and the newly calculated signal columns.
            Should contain the exact same number of rows as the input data
            to ensure proper alignment.
        """
        pass
