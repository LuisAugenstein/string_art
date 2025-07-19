import pickle
from dataclasses import dataclass, field

@dataclass
class StringArtReconstruction:
    strings: list = field(default_factory=list)

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'StringArtReconstruction':
        """Loads a StringArtReconstruction instance from disk using pickle."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)