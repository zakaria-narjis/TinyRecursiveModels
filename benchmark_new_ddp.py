import os
import sys
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
import yaml

# Add current directory to path
sys.path.append(os.getcwd())

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class

# Try importing ARC from evaluators module or direct import
try:
    from evaluators.arc import ARC
except ImportError:
        print("Could not import ARC evaluator. Ensure 'evaluators/arc.py' or 'arc.py' exists.")
        sys.exit(1)

# ============================================================================
# Configuration
# ============================================================================
# /home/zakarianarjis/workspace/TinyRecursiveModels/checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_65100
# /home/zakarianarjis/workspace/TinyRecursiveModels/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_zeros_init_768/step_518071
CHECKPOINT_PATH = "/home/zakarianarjis/workspace/TinyRecursiveModels/checkpoints/Arc1concept-aug-1000-ACT-torch/pretrain_mlp_t_sudoku_zeros_init_768/step_518071"
DATA_PATH = "data/arc1concept-aug-1000" # data/sudoku-extreme-1k-aug-1000  data/arc1concept-aug-1000

# Hyperparameters to experiment with
INFERENCE_CONFIG = {
    "halt_max_steps": 2,  
    "H_cycles": 3,
    "L_cycles": 6,
    "batch_size": 1024,
}

# ============================================================================
# DDP Setup
# ============================================================================

def setup_ddp():
    """Initialize Distributed Data Parallel."""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        
        # Create a CPU group for gathering Python objects (required by ARC evaluator)
        cpu_group = dist.new_group(backend="gloo")
        
        return rank, world_size, local_rank, cpu_group
    else:
        # Fallback for single GPU/CPU debugging
        return 0, 1, 0, None

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# ============================================================================
# Helper Functions
# ============================================================================

def load_model_from_checkpoint(checkpoint_path: str, config_overrides: Dict[str, Any] = None):
    """Load model from checkpoint with optional config overrides."""
    
    # Load the config from the checkpoint directory
    config_file = os.path.join(os.path.dirname(checkpoint_path), "all_config.yaml")
    
    with open(config_file, "r") as f:
        full_config = yaml.safe_load(f)
    
    # Extract model config
    arch_config = full_config["arch"].copy()
    model_name = arch_config["name"]
    loss_config = arch_config["loss"].copy()
    loss_name = loss_config["name"]
    
    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if key in arch_config:
                arch_config[key] = value
    
    # Create model config dict (exclude name and loss)
    model_cfg = {
        k: v for k, v in arch_config.items() 
        if k not in ["name", "loss"]
    }
    
    # Add batch size (placeholder, updated later)
    model_cfg["batch_size"] = config_overrides.get("batch_size", full_config.get("global_batch_size", 64))
    
    # These will be set by the dataset metadata
    model_cfg["vocab_size"] = 0 
    model_cfg["seq_len"] = 0
    model_cfg["num_puzzle_identifiers"] = 0
    model_cfg["causal"] = False
    
    # Extract loss config kwargs (exclude name)
    loss_kwargs = {k: v for k, v in loss_config.items() if k != "name"}
    
    return full_config, model_cfg, model_name, loss_name, loss_kwargs

def strip_compiled_prefix(state_dict):
    """Remove _orig_mod. prefix from compiled model state dict."""
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('_orig_mod.'):
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def create_eval_dataset(data_path: str, batch_size: int, rank: int, world_size: int, split: str = "test"):
    """Create evaluation dataset with DDP support."""
    
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=0,
            dataset_paths=[data_path],
            global_batch_size=batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=rank,
            num_replicas=world_size,
        ),
        split=split
    )
    
    return dataset, dataset.metadata

def patch_initial_carry(model):
    """
    Monkey-patch the model's initial_carry method to ensure tensors 
    are created on the correct device (same as input batch).
    """
    original_initial_carry = model.initial_carry
    
    def initial_carry_wrapper(batch):
        # Call original method (creates tensors on CPU)
        carry = original_initial_carry(batch)
        
        # Get target device from input batch
        device = batch["inputs"].device
        
        # Move relevant tensors to device
        if hasattr(carry, "steps"):
            carry.steps = carry.steps.to(device)
        if hasattr(carry, "halted"):
            carry.halted = carry.halted.to(device)
            
        if hasattr(carry, "inner_carry"):
            if hasattr(carry.inner_carry, "z_H"):
                carry.inner_carry.z_H = carry.inner_carry.z_H.to(device)
            if hasattr(carry.inner_carry, "z_L"):
                carry.inner_carry.z_L = carry.inner_carry.z_L.to(device)
                
        return carry
    
    # Bind the wrapper to the model instance
    model.initial_carry = initial_carry_wrapper
    return model

def run_inference_arc(
    model: nn.Module,
    dataset: PuzzleDataset,
    evaluator: ARC,
    device: str = "cuda",
    max_batches: int = None,
    rank: int = 0
):
    """Run inference using the ARC evaluator."""
    
    model.eval()
    batch_count = 0
    
    evaluator.begin_eval()
    
    # Keys required by ARC evaluator
    required_keys = list(evaluator.required_outputs)
    
    with torch.no_grad():
        dataset_iter = iter(dataset)
        
        while True:
            if max_batches and batch_count >= max_batches:
                break
            
            try:
                batch_data = next(dataset_iter)
                set_name, batch, global_batch_size = batch_data
            except StopIteration:
                break
            
            batch_count += 1
            if rank == 0 and batch_count % 5 == 0:
                print(f"Processing batch {batch_count}...")
            
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Initial carry (Patched version will handle device placement)
            carry = model.initial_carry(batch)
            
            # Run inference until halt
            inference_steps = 0
            max_inference_steps = 100
            
            while True:
                # We request specific keys needed by ARC evaluator: "q_halt_logits", "preds", etc.
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, 
                    batch=batch, 
                    return_keys=required_keys
                )
                inference_steps += 1
                
                if all_finish or inference_steps >= max_inference_steps:
                    break
            
            # Update evaluator
            evaluator.update_batch(batch, preds)
            
            # Clear cache occasionally
            if batch_count % 10 == 0:
                torch.cuda.empty_cache()

    if rank == 0:
        print(f"✓ Inference complete. Processed {batch_count} batches.")

# ============================================================================
# Main Inference Function
# ============================================================================

def evaluate_with_config(
    checkpoint_path: str,
    data_path: str,
    halt_max_steps: int = 16,
    H_cycles: int = 3,
    L_cycles: int = 6,
    batch_size: int = 64,
    max_batches: int = None,
    device: str = "cuda",
    compile_model: bool = False,
    rank: int = 0,
    world_size: int = 1,
    cpu_group = None
):
    """
    Evaluate model with custom hyperparameters using ARC evaluator.
    """
    
    if rank == 0:
        print("=" * 80)
        print("EVALUATION CONFIGURATION")
        print("=" * 80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Data Path: {data_path}")
        print(f"Config: H={H_cycles}, L={L_cycles}, Halt={halt_max_steps}, BS={batch_size}")
        print(f"DDP: Rank {rank}/{world_size}")
        print("=" * 80)
    
    # Load config and prepare overrides
    config_overrides = {
        "halt_max_steps": halt_max_steps,
        "H_cycles": H_cycles,
        "L_cycles": L_cycles,
        "batch_size": batch_size,
    }
    
    full_config, model_cfg, model_name, loss_name, loss_kwargs = load_model_from_checkpoint(
        checkpoint_path, config_overrides
    )
    
    # Create dataset (with DDP sharding)
    dataset, metadata = create_eval_dataset(data_path, batch_size, rank, world_size)
    
    # Update model config with dataset metadata
    model_cfg["vocab_size"] = metadata.vocab_size
    model_cfg["seq_len"] = metadata.seq_len
    model_cfg["num_puzzle_identifiers"] = metadata.num_puzzle_identifiers
    model_cfg["batch_size"] = batch_size // world_size # Per-GPU batch size
    
    # Initialize ARC Evaluator
    evaluator = ARC(
        data_path=data_path,
        eval_metadata=metadata,
        submission_K=2, 
        aggregated_voting=True
    )

    # Load Model
    model_cls = load_model_class(model_name)
    loss_head_cls = load_model_class(loss_name)
    
    with torch.device(device):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **loss_kwargs)
        
        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = strip_compiled_prefix(state_dict)
        
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            model.load_state_dict(state_dict, strict=False)
        
        # --- PATCH: Fix initial_carry device issue ---
        model = patch_initial_carry(model)
        
        if compile_model and "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)
        
        model.eval()

    # Run Inference
    run_inference_arc(
        model, 
        dataset, 
        evaluator, 
        device=device, 
        max_batches=max_batches,
        rank=rank
    )
    
    # Gather Results
    if rank == 0:
        print("Gathering results from all ranks...")
        
    metrics = evaluator.result(
        save_path=None, 
        rank=rank,
        world_size=world_size,
        group=cpu_group 
    )
    
    if rank == 0:
        print("\n" + "=" * 80)
        print("ARC RESULTS")
        print("=" * 80)
        for k, v in metrics.items():
            print(f"{k}: {v:.5f}")
        print("=" * 80)
        
    return metrics

# ============================================================================
# Grid Search
# ============================================================================

def grid_search(rank, world_size, local_rank, cpu_group):
    """Run grid search over hyperparameters."""
    
    device = f"cuda:{local_rank}"
    
    BATCH_SIZE = 192 * 5 * 2
    
    experiments = [
        {"halt_max_steps": 1, "H_cycles": 6, "L_cycles": 12},
        # Add more experiments here
    ]
    
    all_results = []
    
    for exp_config in experiments:
        if rank == 0:
            print(f"\n\n>>> STARTING EXPERIMENT: {exp_config}")
        
        # Force barrier before start
        dist.barrier()
        
        results = evaluate_with_config(
            checkpoint_path=CHECKPOINT_PATH,
            data_path=DATA_PATH,
            batch_size=BATCH_SIZE, # Updated batch size
            max_batches=None,
            device=device,
            rank=rank,
            world_size=world_size,
            cpu_group=cpu_group,
            **exp_config
        )
        
        if rank == 0:
            results["config"] = exp_config
            all_results.append(results)
        
        # Force barrier after end
        dist.barrier()
    
    if rank == 0:
        print("\n\n" + "=" * 100)
        print("COMPARISON OF ALL EXPERIMENTS")
        print("=" * 100)
        
        for i, res in enumerate(all_results):
            cfg = res.pop("config")
            print(f"\nExp {i+1}: {cfg}")
            for k, v in res.items():
                print(f"  {k}: {v:.5f}")

        save_filename = "grid_search_arc_results.json"
        with open(save_filename, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\nSaved results to {save_filename}")

if __name__ == "__main__":
    rank, world_size, local_rank, cpu_group = setup_ddp()
    
    try:
        grid_search(rank, world_size, local_rank, cpu_group)
    finally:
        cleanup_ddp()