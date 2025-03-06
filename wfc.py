import queue
import numpy as np
from typing import List, Tuple, Dict, Set, Any, Union
from utils import *
from gen_pattern import generate_patterns_and_frequencies, save_patterns
from gen_rules import get_rules, get_dirs, print_adjacency_rules

SAVE_VIDEO = True # Record a video frame by frame
SAVE_PATTERNS = True # Save the generated patterns
PRINT_RULES = True # Print all adjacency rules

# Status constants
CONTRADICTION = -2
WAVE_COLLAPSED = -1
RUNNING = 0

def initialize(params: Parameters) -> WFCProperties:
    # Initialize the WFC algorithm

    # Create WFC Properties dataclass
    properties = WFCProperties()

    # Get all possible offset directions for patterns
    properties.directions = get_dirs(params.N)

    # Get a dictionnary that map patterns with teir corresponding frequencies
    pattern_to_freq = generate_patterns_and_frequencies(params.input_path, params.N, params.flip, params.rotate)

    # Split the patterns and the frequencies
    properties.patterns, properties.frequencies = np.array(np.array([to_ndarray(tup) for tup in pattern_to_freq.keys()])), list(pattern_to_freq.values())

    if params.save_patterns:
        save_patterns(properties.patterns, properties.frequencies)

    # Get all the rules of adjacency for all patterns
    properties.rules = get_rules(properties.patterns, properties.directions)

    if params.print_rules:
        print_adjacency_rules(properties.rules)
    
    # Create the coefficient_matrix representing the wave function
    properties.coefficient_matrix = np.full((params.height, params.width, len(properties.patterns)), True, dtype=bool)

    return properties

def min_entropy_coordinates(properties: WFCProperties) -> None:
    # Compute the coordinates of the cell with the minimum entropy and return it alongside the state of the wave (collapsed/running)

    # Compute the probability of each pattern
    proba = np.array(properties.frequencies) / np.sum(properties.frequencies)

    # Compute the weight of each cell
    #weights = properties.coefficient_matrix.astype(int) * properties.frequencies

    # Compute the entropies for each cell
    #entropies = np.log(np.sum(weights, axis=2)) - np.sum(weights * np.log(weights), axis=2) / np.sum(weights, axis=2)
    entropies = np.sum(properties.coefficient_matrix.astype(int) * proba, axis=2)

    # Set entropy to 0 for all collapsed cells
    entropies[np.sum(properties.coefficient_matrix, axis=2) == 1] = 0

    # No more cell to collapse
    if np.sum(entropies) == 0:
        properties.min_entropy_pos = (-1, -1)
        properties.status = WAVE_COLLAPSED
        return

    # Get all indices of minimal non-zero cells
    min_indices = np.argwhere(entropies == np.amin(entropies, initial=np.max(entropies), where=entropies > 0))

    # Choose a random index from the list of minimal indices
    min_index = min_indices[np.random.randint(0, min_indices.shape[0])]

    # Update min entropy pos and set status as Running
    properties.min_entropy_pos = (min_index[0], min_index[1])
    properties.status = RUNNING
    return

def observe(properties: WFCProperties) -> None:
    # Search for the cell with the minimal entropy and collapses it,
    # based on possible patterns in the cell and there respective frequencies
    
    # Check if there is a contradiction
    if np.any(~np.any(properties.coefficient_matrix, axis=2)):
        properties.min_entropy_pos = (-1, -1)
        properties.status = CONTRADICTION
        return
    
    # Get coordinates of the cell with min entropy
    min_entropy_coordinates(properties)

    # Check if the wave is fully collapsed
    if properties.status == WAVE_COLLAPSED:
        return

    # Collapse the cell at min_entropy_pos
    collapse_cell(properties)
    return

def collapse_cell(properties: WFCProperties) -> None:
    # Collapse a cell at min_entropy_pos to a single pattern, randomly choosed based on the pattern's frequencies

    # Get indices of patterns available at min_entropy_pos
    patterns_ind = np.where(properties.coefficient_matrix[properties.min_entropy_pos])[0]

    # Get frequencies of the patterns at min_entropy_pos
    patterns_freq = np.array(properties.frequencies)[patterns_ind]

    # Collapse cell to a pattern randomly, weighted by the frequencies
    chosen_pattern_ind = np.random.choice(patterns_ind, p=patterns_freq / np.sum(patterns_freq))

    # Set possibility of all patterns other than the chosen pattern to false
    properties.coefficient_matrix[properties.min_entropy_pos] = np.full(properties.coefficient_matrix[properties.min_entropy_pos[0], properties.min_entropy_pos[1]].shape[0], False, dtype=bool)
    properties.coefficient_matrix[properties.min_entropy_pos[0], properties.min_entropy_pos[1], chosen_pattern_ind] = True

def propagate_cell(orig_cell: Tuple[int, int], direction: Tuple[int, int], properties: WFCProperties) -> np.ndarray:
    # Propagate from one cell to an adjacent cell according to the set of rules

    # Get the coordinates of the adjacent cell
    adjacent_cell_pos = orig_cell[0] + direction[0], orig_cell[1] + direction[1]

    # Get the patterns that are valid in the target cell, by index
    possible_patterns_in_orig_cell = np.where(properties.coefficient_matrix[orig_cell])[0]

    # Create a vector full of False with the same shape as orig_cell
    possibilities_in_dir = np.zeros(properties.coefficient_matrix.shape[-1], bool)

    # Accumulate all possible patterns in the direction
    for pattern in possible_patterns_in_orig_cell:
        possibilities_in_dir[list(properties.rules[pattern][direction])] = True

    # Apply a boolean OR between the possible patterns from the orig_cell and the patterns of the target cell
    return np.multiply(possibilities_in_dir, properties.coefficient_matrix[adjacent_cell_pos])

def propagate(properties) -> None:
    # After collapsing the cell in the 'Observe' step, we will propagate the change to the neighbours of the cell

    # Create a queueof positions of cells to update
    propagation_queue = queue.Queue()
    propagation_queue.put(properties.min_entropy_pos)

    # Loop until no more cells are left to update
    while not propagation_queue.empty():
        cell = propagation_queue.get()

        # Propagate to each direction of the current cell
        for direction in properties.directions:
            adjacent_cell_pos = cell[0] + direction[0], cell[1] + direction[1]

            # If the adjacent cell is not collapsed, propagate it
            if in_matrix(adjacent_cell_pos, properties.coefficient_matrix.shape) and not is_cell_collapsed(properties.coefficient_matrix, adjacent_cell_pos):
                update_cell = propagate_cell(cell, direction, properties)

                # If the propagation to the adjacent cell changed its wave, update the coefficient matrix
                if not np.array_equal(properties.coefficient_matrix[adjacent_cell_pos], update_cell):
                    properties.coefficient_matrix[adjacent_cell_pos] = update_cell

                    if adjacent_cell_pos not in propagation_queue.queue:
                        propagation_queue.put(adjacent_cell_pos)

def WFCSetup(params: Parameters) -> WFCProperties:
    createFolder()

    properties = initialize(params)
    create_image(params, properties)

    # If we want to save the video
    if params.record:
        properties.current_img.save(f'frames/{params.save_pathname}_{properties.iteration:05d}.png')

    return properties

# Iterate over the steps of the algorithm: observe, collapse, propagate until the wave collapses
def runStep(params: Parameters, properties: WFCProperties) -> None:
    if properties.status == WAVE_COLLAPSED:
        print("The wave has collapsed!")
        return

    properties.iteration += 1

    # Observe and Collapse
    observe(properties)

    if properties.status == CONTRADICTION:
        print("The algorithm found a contradiction\n")
        return

    # Display the current state of the wave function as an image
    collapsed = create_image(params, properties)
    # print(f"{collapsed} cells are collapsed\n")

    if params.record:
        properties.current_img.save(f'frames/{params.save_pathname}_{properties.iteration:05d}.png')

    # Propagate
    propagate(properties)

# Save the final video and image
def saveResult(params: Parameters, properties: WFCProperties) -> None:
    if params.record: createVideo(params)
    if params.save_image: properties.current_img.save('outputs/'+params.save_pathname+'.png')


if __name__=="__main__":

    params = Parameters(save_patterns=False)

    properties = WFCSetup(params)

    while properties.status != WAVE_COLLAPSED:
        runStep(params, properties)