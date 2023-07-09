import numpy as np
from utils import flip_dir
from typing import List, Tuple, Dict, Set, Any, Union

def get_dirs(N: int) -> List[Tuple[int, int]]:
    # Get the coordinates around a pattern
    dirs = [(i, j) for j in range(-N+1, N) for i in range(-N+1, N)]
    dirs.remove((0, 0))
    return dirs

def mask_with_offset(pattern: np.ndarray, offset: Tuple[int, int]) -> np.ndarray:
    # Get a subarray of a pattern based on a given offset
    xoffset, yoffset = offset
    if abs(xoffset)>len(pattern) or abs(yoffset)>len(pattern[0]):
        return np.array([[]])
    
    return pattern[max(0, xoffset):min(len(pattern)+xoffset, len(pattern)),
                    max(0, yoffset):min(len(pattern[0])+yoffset, len(pattern[0])), :]

def check_for_match(p1: np.ndarray, p2: np.ndarray, dir: Tuple[int, int]) -> bool:
    # Check if two patterns with an offset of dir match each other
    p1_offset = mask_with_offset(p1, dir)
    p2_offset = mask_with_offset(p2, flip_dir(dir))
    return np.all(np.equal(p1_offset, p2_offset))

def get_rules(patterns: np.ndarray, directions: List[Tuple[int, int]]) -> List[Dict[Tuple[int, int], Set[int]]]:
    # Create rules which are dictionaries that maps offset (x, y) to a set of indices of all patterns matching a specific pattern
    rules = [{dir: set() for dir in directions} for _ in range(len(patterns))]
    for i in range(len(patterns)):
        for d in directions:
            for j in range(i, len(patterns)):
                if check_for_match(patterns[i], patterns[j], d):
                    rules[i][d].add(j)
                    rules[j][flip_dir(d)].add(i)
    return rules

def print_adjacency_rules(rules: List[Dict[Tuple[int, int], Set[int]]]) -> None:
    for i in range(len(rules)):
        print(f"pattern number {i}: ")
        for k, v in rules[i].items():
            print(f"key: {k}, values: {v}\n")