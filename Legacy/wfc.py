import cv2
import queue
import numpy as np
from typing import List, Tuple, Dict, Set, Any, Union
from utils import to_tuple, to_ndarray, flip_dir, in_matrix, is_cell_collapsed, image_from_coefficients
from gen_pattern import generate_patterns_and_frequencies, save_patterns
from gen_rules import get_rules, get_dirs, print_adjacency_rules
from args import get_opts

SAVE_VIDEO = True # Record a video frame by frame
SAVE_PATTERNS = True # Save the generated patterns
PRINT_RULES = True # Print all adjacency rules

# Status constants
CONTRADICTION = -2
WAVE_COLLAPSED = -1
RUNNING = 0


def initialize( input_path: str,
                N: int,
                width: int,
                height: int,
                flip: bool=True,
                rotate: bool=True,
                save_patterns: bool =False,
                print_rules: bool =False) -> Tuple[np.ndarray, List[Tuple[int, int]], List[int], np.ndarray, List[Dict[Tuple[int, int], Set[int]]]]:
    # Initialize the WFC algorithm

    # Get all possible offset directions for patterns
    directions = get_dirs(N)

    # Get a dictionnary that map patterns with teir corresponding frequencies
    pattern_to_freq = generate_patterns_and_frequencies(input_path, N, flip, rotate)

    # Split the patterns and the frequencies
    patterns, frequencies = np.array(np.array([to_ndarray(tup) for tup in pattern_to_freq.keys()])), list(pattern_to_freq.values())

    if save_patterns:
        save_patterns(patterns, frequencies)

    # Get all the rules of adjacency for all patterns
    rules = get_rules(patterns, directions)

    if print_rules:
        print_adjacency_rules(rules)
    
    # Create the coefficient_matrix representing the wave function
    coefficient_matrix = np.full((height, width, len(patterns)), True, dtype=bool)

    return coefficient_matrix, directions, frequencies, patterns, rules

def min_entropy_coordinates(coefficient_matrix: np.ndarray, frequencies: List[int]) -> Tuple[int, int, int]:
    # Compute the coordinates of the cell with the minimum entropy and return it alongside the state of the wave (collapsed/running)

    # Compute the probability of each pattern
    proba = np.array(frequencies) / np.sum(frequencies)

    # Compute the weight of each cell
    #weights = coefficient_matrix.astype(int) * frequencies

    # Compute the entropies for each cell
    #entropies = np.log(np.sum(weights, axis=2)) - np.sum(weights * np.log(weights), axis=2) / np.sum(weights, axis=2)
    entropies = np.sum(coefficient_matrix.astype(int) * proba, axis=2)

    # Set entropy to 0 for all collapsed cells
    entropies[np.sum(coefficient_matrix, axis=2) == 1] = 0

    if np.sum(entropies) == 0:
        return -1, -1, WAVE_COLLAPSED

    # Get all indices of minimal non-zero cells
    min_indices = np.argwhere(entropies == np.amin(entropies, initial=np.max(entropies), where=entropies > 0))

    # Choose a random index from the list of minimal indices
    min_index = min_indices[np.random.randint(0, min_indices.shape[0])]

    return min_index[0], min_index[1], RUNNING

def observe(coefficient_matrix: np.ndarray, frequencies: List[int]) -> Tuple[Tuple[int, int], np.ndarray, int]:
    # Search for the cell with the minimal entropy and collapses it,
    # based on possible patterns in the cell and there respective frequencies
    
    # Check if there is a contradiction
    if np.any(~np.any(coefficient_matrix, axis=2)):
        return (-1, -1), coefficient_matrix, CONTRADICTION
    
    # Get coordinates of the cell with min entropy
    min_entropy_pos_x, min_entropy_pos_y, status = min_entropy_coordinates(coefficient_matrix, frequencies)
    min_entropy_pos = (min_entropy_pos_x, min_entropy_pos_y)

    # Check if the wave is fully collapsed
    if status == WAVE_COLLAPSED:
        return min_entropy_pos, coefficient_matrix, WAVE_COLLAPSED

    # Collapse the cell at min_entropy_pos
    coefficient_matrix = collapse_cell(coefficient_matrix, frequencies, min_entropy_pos)

    return min_entropy_pos, coefficient_matrix, RUNNING

def collapse_cell(coefficient_matrix: np.ndarray, frequencies: List[int], min_entropy_pos: Tuple[int, int]) -> np.ndarray:
    # Collapse a cell at min_entropy_pos to a single pattern, randomly choosed based on the pattern's frequencies

    # Get indices of patterns available at min_entropy_pos
    patterns_ind = np.where(coefficient_matrix[min_entropy_pos])[0]

    # Get frequencies of the patterns at min_entropy_pos
    patterns_freq = np.array(frequencies)[patterns_ind]

    # Collapse cell to a pattern randomly, weighted by the frequencies
    chosen_pattern_ind = np.random.choice(patterns_ind, p=patterns_freq / np.sum(patterns_freq))

    # Set possibility of all patterns other than the chosen pattern to false
    coefficient_matrix[min_entropy_pos] = np.full(coefficient_matrix[min_entropy_pos[0], min_entropy_pos[1]].shape[0], False, dtype=bool)
    coefficient_matrix[min_entropy_pos[0], min_entropy_pos[1], chosen_pattern_ind] = True

    return coefficient_matrix

def propagate_cell(orig_cell: Tuple[int, int], direction: Tuple[int, int], coefficient_matrix: np.ndarray, rules: List[Dict[Tuple[int, int], Set[int]]]) -> np.ndarray:
    # Propagate from one cell to an adjacent cell according to the set of rules

    # Get the coordinates of the adjacent cell
    adjacent_cell_pos = orig_cell[0] + direction[0], orig_cell[1] + direction[1]

    # Get the patterns that are valid in the target cell, by index
    possible_patterns_in_orig_cell = np.where(coefficient_matrix[orig_cell])[0]

    # Create a vector full of False with the same shape as orig_cell
    possibilities_in_dir = np.zeros(coefficient_matrix.shape[-1], bool)

    # Accumulate all possible patterns in the direction
    for pattern in possible_patterns_in_orig_cell:
        possibilities_in_dir[list(rules[pattern][direction])] = True

    # Apply a boolean OR between the possible patterns from the orig_cell and the patterns of the target cell
    return np.multiply(possibilities_in_dir, coefficient_matrix[adjacent_cell_pos])

def propagate(min_entropy_pos: Tuple[int, int], coefficient_matrix: np.ndarray, rules: List[Dict[Tuple[int, int], Set[int]]], directions: List[Tuple[int, int]]) -> np.ndarray:
    # After collapsing the cell in the 'Observe' step, we will propagate the change to the neighbours of the cell

    # Create a queueof positions of cells to update
    propagation_queue = queue.Queue()
    propagation_queue.put(min_entropy_pos)

    # Loop until no more cells are left to update
    while not propagation_queue.empty():
        cell = propagation_queue.get()

        # Propagate to each direction of the current cell
        for direction in directions:
            adjacent_cell_pos = cell[0] + direction[0], cell[1] + direction[1]

            # If the adjacent cell is not collapsed, propagate it
            if in_matrix(adjacent_cell_pos, coefficient_matrix.shape) and not is_cell_collapsed(coefficient_matrix, adjacent_cell_pos):
                update_cell = propagate_cell(cell, direction, coefficient_matrix, rules)

                # If the propagation to the adjacent cell changed its wave, update the coefficient matrix
                if not np.array_equal(coefficient_matrix[adjacent_cell_pos], update_cell):
                    coefficient_matrix[adjacent_cell_pos] = update_cell

                    if adjacent_cell_pos not in propagation_queue.queue:
                        propagation_queue.put(adjacent_cell_pos)
    
    return coefficient_matrix


if __name__=="__main__":

    # Parse arguments
    args = get_opts()

    # Initialize the WFC algorithm
    height = args.height
    width = args.width
    scale = args.scale

    coefficient_matrix, directions, frequencies, patterns, rules = initialize(
        args.bitmap, 
        args.N, 
        height, width, 
        flip=args.flip, 
        rotate=args.rotate, 
        save_patterns=args.save_patterns, 
        print_rules=args.print_rules)

    # Initialize control parameters
    status = 1
    iteration = 0

    # If we want to save the video
    if args.save_video:
        outvid = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30.0, (int(height * scale), int(width * scale)))

    # Iterate over the steps of the algorithm: observe, collapse, propagate until the wave collapses
    while status != WAVE_COLLAPSED:
        iteration += 1

        # Observe and Collapse
        min_entropy_pos, coefficient_matrix, status = observe(coefficient_matrix, frequencies)

        if status == CONTRADICTION:
            print("The algorithm found a contradiction\n")
            exit(-1)
        
        # Display the current state of the wave function as an image
        collapsed, image = image_from_coefficients(coefficient_matrix, patterns)
        resized_image = cv2.resize(image, (int(height * scale), int(width * scale)), interpolation = cv2.INTER_AREA)
        cv2.imshow(f"Wave Function Collapse", resized_image)
        cv2.waitKey(10)
        print(f"{collapsed} cells are collapsed\n")

        if args.save_video:
            outvid.write(resized_image)

        # Propagate
        coefficient_matrix = propagate(min_entropy_pos, coefficient_matrix, rules, directions)
    
    print("The wave has collapsed!")
    
    if args.save_video:
        outvid.release()

    cv2.imshow(f"Wave Function Collapse", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
