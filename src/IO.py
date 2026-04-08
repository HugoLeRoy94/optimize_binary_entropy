import torch
import os
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import glob


from src.environment import LigandEnvironment, LogNormalConcentration
from src.physics import MWCReceptor,BinaryReceptor
from objectives.loss import ExactInformationLoss

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to seamlessly handle PyTorch tensors and NumPy types."""
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)

# ==========================================
# 1. MOTHER CLASS (Path Management)
# ==========================================
class BaseIO:
    """Mother class that only handles the standardization of paths."""
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.stats_path = os.path.join(self.run_dir, "stats.csv")
        self.config_path = os.path.join(self.run_dir, "config.json")


# ==========================================
# 2. DAUGHTER CLASS 1 (Simulation / Output)
# ==========================================
class ExperimentLogger(BaseIO):
    """Write-only class used during the training simulation."""
    def __init__(self, base_path="/app/data/", experiment_name="optimize_array"):
        # 1. Generate unique timestamped folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_path, f"{experiment_name}_{timestamp}")
        
        # 2. Initialize Mother Class paths
        super().__init__(run_dir)
        
        # 3. Safely create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def save_config(self, config_dict):
        """Saves the CONF dictionary as a human-readable JSON."""
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=4, cls=CustomJSONEncoder)

    def save_stats(self, epoch, stats):
        """Appends a single epoch's stats to a CSV file."""
        stats['epoch'] = epoch
        file_exists = os.path.isfile(self.stats_path)
        
        with open(self.stats_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)

    def save_checkpoint(self, epoch, env, physics, receptor_indices, is_best=False):
        """Saves a training snapshot WITHOUT config or stats to save space."""
        checkpoint = {
            "epoch": epoch,
            "env_state": env.state_dict(),
            "physics_state": physics.state_dict(),
            "receptor_indices": receptor_indices.cpu(),
        }
        fname = f"checkpoint_epoch_{epoch:04d}.pt"
        torch.save(checkpoint, os.path.join(self.ckpt_dir, fname))
        
        if is_best:
            torch.save(checkpoint, os.path.join(self.run_dir, "best_model.pt"))


# ==========================================
# 3. DAUGHTER CLASS 2 (Analysis / Input)
# ==========================================
class ExperimentLoader(BaseIO):
    """Read-only class used for data analysis. Cannot overwrite files."""
    def __init__(self, base_path="/app/data/", experiment_name="optimize_array", exact_run_folder=None):
        
        # The Auto-Discovery Magic
        if exact_run_folder is None:
            search_pattern = os.path.join(base_path, f"{experiment_name}_*")
            matching_folders = sorted(glob.glob(search_pattern))
            
            if not matching_folders:
                raise FileNotFoundError(f"No runs found matching {search_pattern}")
            
            # Grab the last one in the sorted list (the most recent timestamp)
            run_dir = matching_folders[-1]
            print(f"Auto-discovered latest run: {os.path.basename(run_dir)}")
        else:
            run_dir = exact_run_folder
            
        super().__init__(run_dir)
        
        if not os.path.exists(self.run_dir):
            raise FileNotFoundError(f"Directory {self.run_dir} does not exist.")

    def load_config(self):
        """Reads the JSON config."""
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def load_history(self):
        """Returns the CSV stats as a Pandas DataFrame."""
        return pd.read_csv(self.stats_path)

    def load_run(self, filename="best_model.pt", map_location="cpu"):
        """Utility to load a raw checkpoint dictionary."""
        return torch.load(os.path.join(self.run_dir, filename), map_location=map_location)

    def load_all_checkpoints(self, map_location="cpu"):
        """Returns a chronological list of all saved checkpoint dictionaries."""
        files = sorted([f for f in os.listdir(self.ckpt_dir) if f.endswith('.pt')])
        return [torch.load(os.path.join(self.ckpt_dir, f), map_location=map_location) for f in files]
    
    def load_objects(self, filename="best_model.pt", device='cpu'):
        """Reconstructs fully functional PyTorch objects from the saved files."""
        # Note: You must import LigandEnvironment, Receptor, etc., in the script using this!
        
        # 1. Load the separated config and history
        c = self.load_config()
        stats_df = self.load_history()
        
        # 2. Load the weights
        ckpt = self.load_run(filename, map_location=device)
        
        # 3. Rebuild empty objects based on config
        strat = LogNormalConcentration(n_families=c['n_families'], init_mean=5.0)
        e = LigandEnvironment(c['n_units'], c['n_families'], conc_model=strat)
        p = BinaryReceptor(c['n_units'], c['k_sub'])
        l = ExactInformationLoss(k_knn=c.get('k_knn', 5)) # Use .get() for safety
        
        # 4. Inject states
        e.load_state_dict(ckpt['env_state'])
        p.load_state_dict(ckpt['physics_state'])
        
        return e.to(device), p.to(device), l.to(device), ckpt['receptor_indices'], stats_df, c