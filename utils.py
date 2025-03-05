import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Any, Union

# Supported extentions
EXTENTIONS = ['.png', '.jpg', '.jpeg']

@dataclass
class Parameters:
    input_path:     str     = 'inputs_bitmap/Dungeon.png'
    N:              int     = 3
    flip:           bool    = True
    rotate:         bool    = True
    save_patterns:  bool    = True
    print_rules:    bool    = False
    width:          int     = 50
    height:         int     = 50
    pixel_size:     int     = 10
    record:         bool    = False
    save_image:     bool    = True
    save_pathname:      str     = 'video\WFC_Dungeon'

@dataclass
class WFCProperties:
    coefficient_matrix:     np.ndarray                              = field(default_factory=lambda: np.zeros(shape=1, dtype=np.float64))
    directions:             List[Tuple[int, int]]                   = field(default_factory=list)
    frequencies:            List[int]                               = field(default_factory=list)
    patterns:               np.ndarray                              = field(default_factory=lambda: np.zeros(shape=1, dtype=np.float64))
    rules:                  List[Dict[Tuple[int, int], Set[int]]]   = field(default_factory=list)
    min_entropy_pos:        Tuple[int, int]                         = (-1, -1)
    current_img:            np.ndarray                              = field(default_factory=lambda: np.zeros(shape=1, dtype=np.float64))
    outvid:                 np.ndarray                              = field(default_factory=lambda: np.zeros(shape=1, dtype=np.float64))
    status:                 int                                     = 1
    iteration:              int                                     = 0

def to_tuple(array: np.ndarray) -> Union[Tuple, np.ndarray]:
    # Convert array to tuple.
    if isinstance(array, np.ndarray):
        return tuple(map(to_tuple, array))
    else:
        return array
    
def to_ndarray(tup: Tuple) -> Union[Tuple, np.ndarray]:
    # Convert tuple to NumPy ndarray.
    if isinstance(tup, tuple):
        return np.array(list(map(to_ndarray, tup)))
    else:
        return tup

def flip_dir(dir: Tuple[int, int]) -> Tuple[int, int]:
    # Flip the direction of the given direction
    return -1*dir[0], -1*dir[1]

def in_matrix(pos: Tuple[int, int], dims: Tuple[int, ...]) -> bool:
    return 0 <= pos[0] < dims[0] and 0 <= pos[1] < dims[1]

def is_cell_collapsed(coefficient_matrix: np.ndarray, cell_pos: Tuple[int, int]) -> bool:
    return np.sum(coefficient_matrix[cell_pos[0], cell_pos[1], :]) == 1
        
def image_from_coefficients(properties: WFCProperties) -> Tuple[int, np.ndarray]:
    # Generate an image of the state of the wave function
    # Each cell is the mean of all valid patterns for it
    
    height, width, nbpattern = properties.coefficient_matrix.shape

    # Create the resulting image of the wave function
    img = np.empty((height, width, 3), dtype='uint8')

    # Iterate over all cells of coefficient_matrix
    for h in range(height):
        for w in range(width):
            # For each cell, finc all the valid patterns
            valid_patterns = np.where(properties.coefficient_matrix[h, w])[0]

            # Assign the corresponding cell in the output to be the mean of all valid patterns in the cell
            img[h, w] = np.mean(properties.patterns[valid_patterns], axis=0)[0, 0][::-1]
    
    # Return the number of collapsed cells and the final image
    return np.count_nonzero(np.sum(properties.coefficient_matrix, axis=2) == 1), img