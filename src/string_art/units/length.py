from dataclasses import dataclass

@dataclass(frozen=True)
class Length:
    _meter: float

    @property
    def cm(self) -> float:
        return self._meter / 100.
    
    @property
    def mm(self) -> float:
        return self._meter / 1000.
    
    @staticmethod
    def cm(value: float) -> 'Length':
        return Length(value / 100.0)
    
    @staticmethod
    def mm(value: float) -> 'Length':
        return Length(value / 1000.0)