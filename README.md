# String Art
This repository aims to reconstruct images by spanning a string through pins that lie on the perimeter of a circle. Different algorithms are benchmarked against each other.

## Setup
```bash
# Install python
curl https://pyenv.run | bash
pyenv install --list # search for the most recent version, e.g., 3.12.5 at the time of writing
pyenv install 3.12.5
pyenv global 3.12.5

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

# Getting Started
The `examples/` directory contains executable scripts to get familiar with the usage of this library.

# References
["String Art: Towards Computational Fabrication of String Images"](https://www.geometrie.tuwien.ac.at/geom/ig/publications/stringart/stringart.pdf) introduces the general goal of creating stringart images in a nice way and proposes a greedy algorithm implemented in Matlab (see [code](https://github.com/Exception1984/StringArt)). The approach relies on huge string matrices which quickly requires too much memory when applying it to larger images with many pins. 

["How To Make a Computer Create Something Beautiful: String Art"](https://www.youtube.com/watch?v=dBlSmg5T13M&t=84s) introduces the radon transform for creating string art images and served as the main inspiration for creating this project. 

# TODOs
- better understand the analytical radon transform of a line
- create presentation 
- add some metrics to get more insights, e.g.,
    - how many strings are typically and maximally on one pin? Is the maximum of N_pins-1 connections reached by pins regularly in different images ? how does that change when N_pins increases ? This influences the actual required height of the pins. 
- add algorithm to create a consecutive string path that can be manufactured with one long string - The greedy paper should already have done something like this
- build an actual machine that creates these string images
- currently we use a large domain of radon parameters and then only keep the strings are closest to the strings between two pins. This is unnecessary overhead. Instead the radon parameter domain should already be smaller in the first place to only contain the parameters that exactly represent the strings between pins. 
- introduce a caching mechanism such that string reconstructions for the same image and the same parameters does not need to be recomputed if it already is stored on disk. 
    -> probably create a new directory for each run with the "{image_name}-{hash(config_object)}" to easily check if we already have a run that matches the given configuration.
- create a nice frontend to get a better experience than currently with matplotlib
    - add a slider for selecting how many strings should be shown
- extend the algorithm to work on different shapes than only the circle
- experiment with multicolor string images