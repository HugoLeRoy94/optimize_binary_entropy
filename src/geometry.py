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

def generate_targeted_receptors(n_units, k_sub, composition_targets):
    """
    Generates combinations prioritized by their complexity (number of unique subunits).
    
    Args:
        n_units (int): Total number of available sub-units.
        k_sub (int): Number of sub-units in a receptor (e.g., 5).
        composition_targets (dict): Defines how many to draw per complexity level.
                                    Format: {num_unique_units: count}
                                    Example: {1: 'all', 2: 10, 3: 5}
    """
    all_combos = list(itertools.combinations_with_replacement(range(n_units), k_sub))
    
    # 1. Bucket the combinations by the number of unique subunits
    buckets = {}
    for combo in all_combos:
        n_unique = len(set(combo))
        if n_unique not in buckets:
            buckets[n_unique] = []
        buckets[n_unique].append(combo)
        
    selected_combos = []
    
    # 2. Iterate through targets in sorted order to maintain priority in the final tensor
    for n_unique in sorted(composition_targets.keys()):
        target_k = composition_targets[n_unique]
        
        if n_unique not in buckets:
            continue # Skip if this complexity is mathematically impossible 
            
        available = buckets[n_unique]
        
        if target_k == 'all' or target_k >= len(available):
            # Take all available combinations in this bucket
            selected_combos.extend(available)
        else:
            # Randomly sample 'target_k' combinations
            selected_combos.extend(random.sample(available, target_k))
            
    return torch.tensor(selected_combos, dtype=torch.long)


def generate_cascading_receptors(n_units, k_sub, n_sensors):
    """
    Automatically prioritizes simpler combinations (homomers > 2-mixes > 3-mixes)
    until the 'n_sensors' quota is reached.
    """
    all_combos = list(itertools.combinations_with_replacement(range(n_units), k_sub))
    
    buckets = {}
    for combo in all_combos:
        n_unique = len(set(combo))
        if n_unique not in buckets:
            buckets[n_unique] = []
        buckets[n_unique].append(combo)
        
    selected_combos = []
    remaining = n_sensors
    
    for n_unique in sorted(buckets.keys()):
        if remaining <= 0:
            break
            
        available = buckets[n_unique]
        # Shuffle within the tier so we don't always bias towards lower subunit indices
        random.shuffle(available) 
        
        take_n = min(remaining, len(available))
        selected_combos.extend(available[:take_n])
        remaining -= take_n
        
    return torch.tensor(selected_combos, dtype=torch.long)