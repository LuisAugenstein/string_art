# String Art
This repository aims to reconstruct images by spanning a string through pins that lie on the perimeter of a circle. 

# Getting Started
The `examples/` directory contains executable scripts to get familiar with the usage of this library.

# References
["String Art: Towards Computational Fabrication of String Images"](https://www.geometrie.tuwien.ac.at/geom/ig/publications/stringart/stringart.pdf) introduces the general goal of creating stringart images in a nice way and proposes a greedy algorithm implemented in Matlab (see [code](https://github.com/Exception1984/StringArt)). The approach relies on huge string matrices which quickly requires too much memory when applying it to larger images with many pins. 

["How To Make a Computer Create Something Beautiful: String Art"](https://www.youtube.com/watch?v=dBlSmg5T13M&t=84s) introduces the radon transform for creating string art images and served as the main inspiration for creating this project. 