import pickle
from dataclasses import dataclass
from string_art.edges import EdgesAngleBased

@dataclass
class StringArtReconstruction:
    strings: EdgesAngleBased = None

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'StringArtReconstruction':
        """Loads a StringArtReconstruction instance from disk using pickle."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
        