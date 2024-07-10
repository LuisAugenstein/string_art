# Assumptions / Definitions

depending on the context we represent an edge either by its two pin_indices or its two pins itself.
edges_index_based: [N_strings, 2]    used anywhere else
edges_pin_based: [N_strings, 2, 2]   required for computing the string_matrix

string_matrix: [IMAGE_SIZE**2, N_strings]

# String Art
This repository aims to reproduce the results of ["String Art: Towards Computational Fabrication of String Images"](https://www.geometrie.tuwien.ac.at/geom/ig/publications/stringart/stringart.pdf). The original [code](https://github.com/Exception1984/StringArt) was implemented in Matlab. This project restructures the code and implements the proposed algorithm in python to make it more accessible. 

# Installation
The following setup assumes you are using miniconda on Ubuntu after you have cloned the repo
```
# current directory ~/string_art: 
conda create -n string_art
conda activate string_art
conda install pip
pip install -r requirements.txt
mkdir -p ~/miniconda3/envs/string_art/etc/conda/activate.d
echo 'export PYTHONPATH=/path/to/string_art' >> ~/miniconda3/envs/string_art/etc/conda/activate.d/env_vars.sh
```

# Get Started
The `examples/` directory contains executable scripts to get familiar with the usage of this library. Try the `examples/reproduce_matlab_results.py` to run a small 16 pin example. You can also run the original matlab [code](https://github.com/Exception1984/StringArt) with the instructions given [below](#exactly-reproduce-results-from-original-matlab-code) to compare the results.

# Entities
The String art images are created by pulling a string around pins to approximate a target image as close as possible. To better understand the concepts and the accompanying code, this section introduces three key terms: `Pin`, `Edge`, and `String`:

- `Pins` are numerated nodes that are positioned in a circle around the image. They are uniquely identified by their indices, forming an ordered list like `pins = [0, 1, 2, 3, ..., n_pins-1]`.

- `Edges` are abstract connections between two pins. An edge between a pin `i` and `j` is defined as the tuple `(i,j)`. Since every pin can be connected to every other pin we can numerate all pins and edges as follows:
    ```python
    pins = list(range(n_pins))
    edges=[(i, j) for i in pins for j in pins[i+1:]]
    ```
    In total there are `n_edges = n_choose_k(n_pins, 2)` edges. Based on the enumeration we can also refer to edges by their indices in the `edges` list, e.g., the edge at index $4$ is $e_4=(0,5)$.

<div align='center'>  
  <img src="docs/pin_and_edge_visualization.svg" width="350" height="350">  <img src="docs/connection_types.svg" width="350" height="350">
</div>

- `Strings` represent the pixels $(x,y)$ and grayscale values $v$ of actual straight lines that connect two pins. Since real pins have a width there are 4 different connection types to connect two pins, namely: `Straight In: 0, Straight Out: 1, Diagonal In: 2, Diagonal Out: 3`. Note, that the connection types $3$ and $4$ can always be associated to one specific line. To assign connection types $1$ and $2$ we need to know whether the edge is ingoing or outgoing. For `n_pins` pins there are `n_strings = 4*n_edges` possible strings that can be drawn. 


# Exactly Reproduce Results from Original Matlab Code
Running the `examples/example.py` file starts a small 16 pin run with the 'cat.png' image. To see the exact same results in the original matlab [code](https://github.com/Exception1984/StringArt), a few changes have to be made:

1. remove the randomness in the `pinPositions.m` file, i.e., replace
```matlab
function [pinPos, angles] = pinPositions(numPins)
    
    if nargin == 0
        numPins = 512;
    end
    
    maxAngle = 2 * pi /numPins;
    minAngle = 0.95 * 2 * pi /numPins;
    range = maxAngle - minAngle;
    
    rng(0);
    
    % Accumulative Method
    angles = range * rand(numPins + 1, 1) + minAngle;
    angles = cumsum(angles);
    firstHookAngle = 2 * pi + angles(1);
    angles = angles .* (firstHookAngle / angles(end));
    angles = angles(1 : end - 1);
    
    pinPos = [cos(angles) sin(angles)];
end

```
with
```matlab

function [pinPos, angles] = pinPositions(numPins)
    
    if nargin == 0
        numPins = 512;
    end
    
    pin_angles = linspace(0, 2*pi, numPins+1);
    angles = pin_angles(1:numPins);
    pinPos = [cos(angles); sin(angles)]';
end

```

2. In line 59 of `Hook.m` the convex hull computation is not consistent when the pins are positioned directly opposite to each other. Only in those cases replace the `convhull` call with a `boundary` call: 
```matlab
...
for i = 1 : 4
    bPoint = bPoints(:, i);
    K = convhull([aPoints(1, :) bPoint(1)], [aPoints(2, :) bPoint(2)]);
    % this is new
    if all(abs(aPoints(:) + bPoints(:)) < 1.0e-8)
        K = boundary([aPoints(1, :) bPoint(1)]', [aPoints(2, :) bPoint(2)]');
    end
    % new code end
...
```

3. Optional: Fix the `computeValidEdgesMask` function in the `GreedyMultiSamplingDataObjectL2.m` file 
from
```matlab
obj.validEdgesMask(edgeAngles <= minAngle) = false;
```
to
```matlab
obj.validEdgesMask(edgeAngles - minAngle <= 1e-8) = false;
```
to accurately exclude the edges between pins that are closer than minAngle.

4. Set the number of pins in `example_cat.m` to `numPins=16` and run it.