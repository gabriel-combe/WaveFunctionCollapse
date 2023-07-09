import numpy as np
from typing import List, Tuple, Dict, Set, Any, Union

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
        
def image_from_coefficients(coefficient_matrix: np.ndarray, patterns: np.ndarray) -> Tuple[int, np.ndarray]:
    # Generate an image of the state of the wave function
    # Each cell is the mean of all valid patterns for it
    
    height, width, nbpattern = coefficient_matrix.shape

    # Create the resulting image of the wave function
    img = np.empty((height, width, 3), dtype='uint8')

    # Iterate over all cells of coefficient_matrix
    for h in range(height):
        for w in range(width):
            # For each cell, finc all the valid patterns
            valid_patterns = np.where(coefficient_matrix[h, w])[0]

            # Assign the corresponding cell in the output to be the mean of all valid patterns in the cell
            img[h, w] = np.mean(patterns[valid_patterns], axis=0)[0, 0][::-1]
    
    # Return the number of collapsed cells and the final image
    return np.count_nonzero(np.sum(coefficient_matrix, axis=2) == 1), img