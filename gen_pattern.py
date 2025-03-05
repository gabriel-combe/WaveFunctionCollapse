import os
import numpy as np
from utils import to_tuple
from PIL import Image as Img
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Set, Any, Union

# Generate patterns and compute their associated frequencies
def generate_patterns_and_frequencies(path: str, N:int, flip: bool=True, rotate: bool=True) -> Dict[tuple[Any, ...], int]:
    # Open the image and convert it to numpy array
    img = np.array(Img.open(path))

    # Check if the image has more than 3 channels
    if img.shape[2] > 3:
        #Reduce the number of channel to 3
        img = img[:, :, :3]
    
    patterns = patterns_from_image(img, N, flip, rotate)

    patterns, freqs = np.unique(patterns, axis=0, return_counts=True)
    pattern_to_freq = {to_tuple(patterns[i]): freqs[i] for i in range(len(freqs))}

    return pattern_to_freq

# Extract patterns from the image
def patterns_from_image(img: np.ndarray, N: int, flip: bool=True, rotate: bool=True) -> List[np.ndarray]:
    # List of patterns
    patterns = []

    for i in range(img.shape[0] - N + 1):
        for j in range(img.shape[1] - N + 1):
            patterns.append(img[i:i+N, j:j+N, :])
            index = len(patterns)-1
    
            #Optionnaly include flipped and rotated version of the patterns
            if flip:
                patterns.append(np.flip(patterns[index], axis=0))
                patterns.append(np.flip(patterns[index], axis=1))

            if rotate:
                patterns.append(np.rot90(patterns[index], k=1))
                patterns.append(np.rot90(patterns[index], k=2))
                patterns.append(np.rot90(patterns[index], k=3))
    
    return np.array(patterns, dtype='uint8')

# Save the generated patterns in the output path provided
def save_patterns(patterns: np.ndarray, freq: List[int], output_path: str = 'output') -> None:
    # Saves a list of image tiles to new image files in a given output path.

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sum_of_freq = sum(freq)
    for i in range(len(patterns)):
        fig, axs = plt.subplots()
        axs.imshow(patterns[i])
        axs.set_title(f"pattern No. {i + 1}")
        axs.set_xticks([])
        axs.set_yticks([])
        fig.suptitle(f"Number of occurrences: {freq[i]}, probability: {round(freq[i] / sum_of_freq, 2)}")
        plt.savefig(os.path.join(output_path, f'pattern_{i}.jpg'))
        plt.close()