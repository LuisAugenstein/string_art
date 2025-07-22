from string_art.core.string_art_config import StringArtConfig
from string_art.core.string_art_store import StringArtStore

class StringArtVisualizer:
    config: StringArtConfig
    store: StringArtStore
    _is_first_call: bool

    def __init__(self, config: StringArtConfig, store: StringArtStore):
        self.config = config
        self.store = store
        self._is_first_call = True

    def notify(self) -> None:
        if self._is_first_call:
            self._is_first_call = False
            self.create_figure()
        self.update()

    def create_figure(self) -> None:
        ...

    def update(self) -> None:
        ...