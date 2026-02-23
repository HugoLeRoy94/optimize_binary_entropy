import torch
import itertools
import random

def generate_receptor_indices(n_units, k_sub, n_sensors):
    """
    Generates the identity of the receptors in our array.
    Since 26^5 is huge, we randomly sample 'n_sensors' unique combinations 
    (with replacement, e.g., AABBB) to simulate the array.
    """
    # 1. Generate all possible combinations (approx 142k for 26 choose 5)
    # combinations_with_replacement handles stoichiometry (AAAAA, AAAAB, etc.)
    all_combos = list(itertools.combinations_with_replacement(range(n_units), k_sub))
    
    # 2. Select a subset to simulate the "Octopus Nose"
    if n_sensors > len(all_combos):
        selected = all_combos
    else:
        selected = random.sample(all_combos, n_sensors)
        
    return torch.tensor(selected, dtype=torch.long)