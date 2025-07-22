from functools import cached_property
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
    
    image: torch.Tensor
    reconstruction: StringArtReconstruction

    @cached_property
    def store_path(self) -> str:
        hash = self._generate_config_hash(self.config, self.image)
        return f'{self.config.store_path}/{hash}'

    def __init__(self, config: StringArtConfig):
        self.config = config
        
    def load(self) -> StringArtReconstruction | None:
        os.makedirs(self.store_path, exist_ok=True)

        if os.path.exists(f'{self.store_path}/{self._RECONSTRUCTION_FILE_NAME}'):
            print(f"Load existing reconstruction from '{self.store_path}'\nconfig: {self.config}")
            self.reconstruction = StringArtReconstruction.load(f'{self.store_path}/{self._RECONSTRUCTION_FILE_NAME}')
            return self.reconstruction

        print(f"Initialize new store directory in '{self.store_path}'\nconfig:{self.config}")
        self._save_config(self.config, self.store_path)
        
    def update(self, reconstruction: StringArtReconstruction, save_to_disk=False) -> None:
        self.reconstruction = reconstruction
        for listener in self.listeners:
            listener.notify()
        if save_to_disk:
            self.save()
    
    def save(self) -> None:
        torch.save(self.image, f'{self.store_path}/{self._IMAGE_FILE_NAME}')
        self.reconstruction.save(f'{self.store_path}/{self._RECONSTRUCTION_FILE_NAME}')

    def register(self, listener: StringArtListener) -> None:
        self.listeners.append(listener)

    @staticmethod
    def _generate_config_hash(config: StringArtConfig, image: torch.Tensor, hash_length=20) -> str:
        config_dict = asdict(config)
        config_dict['image'] = image
        config_str = ''.join(f'{key}:{value}' for key, value in sorted(config_dict.items()))
        hash_object = hashlib.sha256(config_str.encode())
        hash_hex = hash_object.hexdigest()
        return hash_hex[:hash_length]

    @staticmethod
    def _save_config(config: StringArtConfig, store_path: str):
        config_dict = asdict(config)
        with open(f'{store_path}/config.yaml', 'w') as file:
            yaml.dump(config_dict, file)

    