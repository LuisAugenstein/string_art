# String Art
This repository aims to reproduce the results of ["String Art: Towards Computational Fabrication of String Images"](https://www.geometrie.tuwien.ac.at/geom/ig/publications/stringart/stringart.pdf). The original [code](https://github.com/Exception1984/StringArt) was implemented in Matlab. This project implements the proposed algorithm in python and tries to make it more accessible. 

# Installation
The following setup assumes you are using miniconda on Ubuntu after you have cloned the repo
```
# current directory ~/string_art: 
conda create -n string_art
conda activate string_art
conda install pip
pip install -r requirements.txt
mkdir -p ~/miniconda3/envs/string_art/etc/conda/activate.d
echo 'export PYTHONPATH=/path/to/string_art' >> ~miniconda3/envs/string_art/etc/conda/activate.d/env_vars.sh
```