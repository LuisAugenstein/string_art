import os
import yaml
import torch
import hashlib
from dataclasses import asdict
from string_art.core import StringArtConfig
from string_art.core.string_art_listener import StringArtListener
from string_art.core.string_art_reconstruction import StringArtReconstruction

class StringArtStore:
    config: StringArtConfig
    listeners: list[StringArtListener] = []
    _IMAGE_FILE_NAME = 'image.pt'
    _RECONSTRUCTION_FILE_NAME = 'reconstruction.pkl'
    _CONFIG_FILE_NAME = 'config.yaml'

    def __init__(self, config: StringArtConfig):
        self.config = config
        
    def load(self, image: torch.Tensor) -> StringArtReconstruction | None:
        if not os.path.exists(self.config.store_path):
            os.mkdir(self.config.store_path)
        
        store_path = self.get_store_path(image)

        if os.path.exists(store_path):
            print(f"Load existing reconstruction from '{store_path}'\nconfig: {self.config}")
            return StringArtReconstruction.load(f'{store_path}/{self._RECONSTRUCTION_FILE_NAME}')

        print(f"Initialize new store directory in '{store_path}'\nconfig:{self.config}")
        os.makedirs(store_path)
        self._save_config(self.config, store_path)
        
    def update(self, image: torch.Tensor, reconstruction: StringArtReconstruction, save_to_disk=False) -> None:
        for listener in self.listeners:
            listener.notify(image, reconstruction)
        if save_to_disk:
            self.save(image, reconstruction)
    
    def save(self, image: torch.Tensor, reconstruction: StringArtReconstruction) -> None:
        store_path = self.get_store_path(image)
        torch.save(image, f'{store_path}/{self._IMAGE_FILE_NAME}')
        reconstruction.save(f'{store_path}/{self._RECONSTRUCTION_FILE_NAME}')

    def register(self, listener: StringArtListener) -> None:
        self.listeners.append(listener)

    def get_store_path(self, image: torch.Tensor) -> str:
        hash = self._generate_config_hash(self.config, image)
        return f'{self.config.store_path}/{hash}'

    def _generate_config_hash(self, config: StringArtConfig, image: torch.Tensor, hash_length=20) -> str:
        config_dict = asdict(config)
        config_dict['image'] = image
        config_str = ''.join(f'{key}:{value}' for key, value in sorted(config_dict.items()))
        hash_object = hashlib.sha256(config_str.encode())
        hash_hex = hash_object.hexdigest()
        return hash_hex[:hash_length]

    def _save_config(self, config: StringArtConfig, store_path: str):
        config_dict = asdict(config)
        with open(f'{store_path}/config.yaml', 'w') as file:
            yaml.dump(config_dict, file)

    