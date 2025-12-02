import os
import sys
import json
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List
import yaml
import time
import copy

# Add current directory to path
sys.path.append(os.getcwd())

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
from models.losses import IGNORE_LABEL_ID

# ============================================================================
# Configuration
# ============================================================================

# Use your actual paths here
CHECKPOINT_PATH = "checkpoints/Maze-30x30-hard-1k-ACT-torch/pretrain_att_maze30x30/step_65100"
DATA_PATH = "data/maze-30x30-hard-1k"

# Global DDP variables, set once
GLOBAL_RANK = 0
LOCAL_RANK = 0
WORLD_SIZE = 1

# ============================================================================
# DDP Helper Functions
# ============================================================================

def setup_ddp():
    """Initialize Distributed Data Parallel."""
    global GLOBAL_RANK, LOCAL_RANK, WORLD_SIZE
    
    if "LOCAL_RANK" in os.environ and not dist.is_initialized():
        # Initialize only if environment variables are set and not already initialized
        dist.init_process_group(backend="nccl")
        LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(LOCAL_RANK)
        GLOBAL_RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
    elif not dist.is_initialized():
        # Fallback for single GPU/No torchrun (only use first GPU)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            LOCAL_RANK = 0
        GLOBAL_RANK = 0
        WORLD_SIZE = 1
        
    # IMPORTANT: Return the currently set global variables
    return GLOBAL_RANK, LOCAL_RANK, WORLD_SIZE

def is_main_process():
    """Check if the current process is rank 0."""
    return GLOBAL_RANK == 0

# ============================================================================
# Loading Helpers (Moved logic outside of functions to use GLOBAL_RANK/WORLD_SIZE)
# ============================================================================

def load_model_from_checkpoint(checkpoint_path: str, config_overrides: Dict[str, Any] = None):
    """Load model from checkpoint with optional config overrides."""
    
    config_file = os.path.join(os.path.dirname(checkpoint_path), "all_config.yaml")
    
    with open(config_file, "r") as f:
        full_config = yaml.safe_load(f)
    
    if is_main_process():
        print("\nOriginal config from checkpoint:")
        print(f"  Data paths: {full_config.get('data_paths', 'N/A')}")
        print(f"  Global batch size: {full_config.get('global_batch_size', 'N/A')}")
    
    # Extract model config
    arch_config = full_config["arch"].copy()
    model_name = arch_config["name"]
    loss_config = arch_config["loss"].copy()
    loss_name = loss_config["name"]
    
    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if key in arch_config:
                if is_main_process():
                    print(f"Overriding {key}: {arch_config[key]} -> {value}")
                arch_config[key] = value
    
    # Create model config dict (exclude name and loss)
    model_cfg = {
        k: v for k, v in arch_config.items() 
        if k not in ["name", "loss"]
    }
    
    # Add batch size
    model_cfg["batch_size"] = config_overrides.get("batch_size", full_config.get("global_batch_size", 64))
    
    # These will be set by the dataset metadata
    model_cfg["vocab_size"] = 0  # Placeholder
    model_cfg["seq_len"] = 0  # Placeholder
    model_cfg["num_puzzle_identifiers"] = 0  # Placeholder
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


def create_eval_dataset(data_path: str, batch_size: int):
    """Create evaluation dataset with DDP sharding using global variables."""
    
    # PuzzleDataset handles sharding internally via rank/num_replicas
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=0,
            dataset_paths=[data_path],
            global_batch_size=batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            # Use global DDP variables here
            rank=GLOBAL_RANK, 
            num_replicas=WORLD_SIZE,
        ),
        split="test"
    )
    
    return dataset, dataset.metadata

# ============================================================================
# Metric & Inference Logic
# ============================================================================

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute accuracy metrics."""
    
    # Get predictions
    preds = torch.argmax(logits, dim=-1)
    
    # Create mask for valid labels (ignore IGNORE_LABEL_ID)
    valid_mask = (labels != IGNORE_LABEL_ID)
    
    # Token-level accuracy calculations
    correct_tokens = (preds == labels) & valid_mask
    num_correct_tokens = correct_tokens.sum().item()
    num_valid_tokens = valid_mask.sum().item()
    
    token_accuracy = num_correct_tokens / num_valid_tokens if num_valid_tokens > 0 else 0.0
    
    # Sequence-level exact match accuracy
    correct_per_sequence = correct_tokens.sum(dim=1) == valid_mask.sum(dim=1)
    num_correct_sequences = correct_per_sequence.sum().item()
    num_sequences = labels.shape[0]
    
    exact_accuracy = correct_per_sequence.float().mean().item()
    
    return {
        "token_accuracy": token_accuracy,
        "exact_accuracy": exact_accuracy,
        "num_correct_tokens": num_correct_tokens,
        "num_valid_tokens": num_valid_tokens,
        "num_sequences": num_sequences,
        "num_correct_sequences": num_correct_sequences,
    }

# Helper to move nested dict/object structure to device
def move_to_device(data: Any, device: torch.device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    return data

def run_inference_ddp(
    model: nn.Module,
    dataset: PuzzleDataset,
    device: torch.device,
    max_batches: int = None
) -> Dict[str, Any]:
    """Run inference on the dataset and compute distributed metrics."""
    
    model.eval()
    
    # Local accumulators
    local_metrics = {
        "num_correct_tokens": 0.0,
        "num_valid_tokens": 0.0,
        "num_correct_sequences": 0.0,
        "num_sequences": 0.0,
        "sum_inference_steps": 0.0,
        "num_batches": 0.0
    }
    
    local_per_batch_metrics = []

    batch_count = 0
    
    with torch.no_grad():
        dataset_iter = iter(dataset)
        
        while True:
            if max_batches and batch_count >= max_batches:
                break
            
            try:
                # PuzzleDataset yields (set_name, batch, global_batch_size)
                batch_data = next(dataset_iter)
                set_name, batch, _ = batch_data 
                
            except StopIteration:
                break
            except Exception as e:
                if is_main_process():
                    print(f"Error fetching batch: {e}")
                break
            
            batch_count += 1
            
            # To device
            batch = move_to_device(batch, device)
            
            # Init Carry
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            
            # Inference Loop
            inference_steps = 0
            max_inference_steps = 100 # Safety limit
            
            while True:
                # model() is expected to return tensors on the input's device (CUDA)
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, 
                    batch=batch, 
                    return_keys=["logits"]
                )
                inference_steps += 1
                
                # 'all_finish' can be a boolean or a tensor. If it's a tensor, 
                # ensure comparison is done correctly (it should be reduced to a single value here).
                if isinstance(all_finish, torch.Tensor):
                    all_finish = all_finish.all().item()
                    
                if all_finish or inference_steps >= max_inference_steps:
                    break
            
            # Compute Metrics
            logits = preds["logits"]
            labels = batch["labels"]
            
            # Ensure labels are also on the device, already handled by move_to_device(batch)
            batch_m = compute_metrics(logits, labels)
            
            # Update local accumulators
            local_metrics["num_correct_tokens"] += batch_m["num_correct_tokens"]
            local_metrics["num_valid_tokens"] += batch_m["num_valid_tokens"]
            local_metrics["num_correct_sequences"] += batch_m["num_correct_sequences"]
            local_metrics["num_sequences"] += batch_m["num_sequences"]
            local_metrics["sum_inference_steps"] += inference_steps
            local_metrics["num_batches"] += 1
            
            # Store detail for this batch (optional for full json, but good for debugging)
            detail = {
                "token_accuracy": batch_m["token_accuracy"],
                "exact_accuracy": batch_m["exact_accuracy"],
                "num_valid_tokens": batch_m["num_valid_tokens"],
                "num_sequences": batch_m["num_sequences"],
                "num_correct_sequences": batch_m["num_correct_sequences"],
                "inference_steps": inference_steps,
                "set_name": set_name
            }
            local_per_batch_metrics.append(detail)
            
            if is_main_process() and batch_count % 10 == 0:
                 print(f"Processing batch {batch_count}...")

    # ==========================================
    # AGGREGATE METRICS ACROSS ALL RANKS
    # ==========================================
    
    # Prepare tensor for all_reduce: [correct_tok, valid_tok, correct_seq, total_seq, sum_steps, num_batches]
    metrics_tensor = torch.tensor([
        local_metrics["num_correct_tokens"],
        local_metrics["num_valid_tokens"],
        local_metrics["num_correct_sequences"],
        local_metrics["num_sequences"],
        local_metrics["sum_inference_steps"],
        local_metrics["num_batches"]
    ], dtype=torch.float64, device=device)
    
    # Sum up everything from all GPUs
    if WORLD_SIZE > 1:
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
    # Extract global values
    global_correct_tok = metrics_tensor[0].item()
    global_valid_tok = metrics_tensor[1].item()
    global_correct_seq = metrics_tensor[2].item()
    global_total_seq = metrics_tensor[3].item()
    global_sum_steps = metrics_tensor[4].item()
    global_num_batches = metrics_tensor[5].item()
    
    # Compute Final Averages
    final_token_acc = global_correct_tok / global_valid_tok if global_valid_tok > 0 else 0.0
    final_exact_acc = global_correct_seq / global_total_seq if global_total_seq > 0 else 0.0
    final_avg_steps = global_sum_steps / global_num_batches if global_num_batches > 0 else 0.0
    
    return {
        "token_accuracy": final_token_acc,
        "exact_accuracy": final_exact_acc,
        "avg_inference_steps": final_avg_steps,
        "num_batches": int(global_num_batches),
        "total_valid_tokens": int(global_valid_tok),
        "total_sequences": int(global_total_seq),
        "total_correct_sequences": int(global_correct_seq),
        # This list will only contain batches processed by this specific rank.
        # We keep it for consistency but strip it before saving.
        "per_batch_metrics": local_per_batch_metrics 
    }

# ============================================================================
# Main Evaluation Function
# ============================================================================

def evaluate_with_config(
    checkpoint_path: str,
    data_path: str,
    halt_max_steps: int = 16,
    H_cycles: int = 3,
    L_cycles: int = 6,
    batch_size: int = 64,
    max_batches: int = None,
    compile_model: bool = False
):
    """Evaluate model with custom hyperparameters using DDP."""
    
    device = torch.device(f"cuda:{LOCAL_RANK}")
    
    if is_main_process():
        print("=" * 80)
        print(f"EVALUATION CONFIGURATION (DDP World Size: {WORLD_SIZE})")
        print("=" * 80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Data Path: {data_path}")
        print(f"halt_max_steps: {halt_max_steps}")
        print(f"H_cycles: {H_cycles}")
        print(f"L_cycles: {L_cycles}")
        print(f"batch_size: {batch_size}")
    
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
    
    # Create dataset
    if is_main_process():
        print("\nLoading dataset...")
    
    # IMPORTANT: The dataset creation uses the globally set rank/world_size
    dataset, metadata = create_eval_dataset(data_path, batch_size)
    
    # Update model config
    model_cfg["vocab_size"] = metadata.vocab_size
    model_cfg["seq_len"] = metadata.seq_len
    model_cfg["num_puzzle_identifiers"] = metadata.num_puzzle_identifiers
    
    if is_main_process():
        print(f"Dataset loaded. Total puzzles: {metadata.total_puzzles}. Sharded elements for this rank.")
        print("\nLoading model...")
    
    model_cls = load_model_class(model_name)
    loss_head_cls = load_model_class(loss_name)
    
    with torch.device(device):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **loss_kwargs)
        
        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = strip_compiled_prefix(state_dict)
        
        try:
            model.load_state_dict(state_dict, strict=True)
            if is_main_process(): print("✓ Weights loaded successfully (strict mode)")
        except Exception as e:
            if is_main_process(): 
                print(f"⚠ Warning: Could not load with strict=True: {e}")
            model.load_state_dict(state_dict, strict=False)
        
        if compile_model and "DISABLE_COMPILE" not in os.environ:
            if is_main_process(): print("Compiling model...")
            model = torch.compile(model)
        
        model.eval()
    
    if is_main_process():
        print("\n" + "=" * 80)
        print("RUNNING DISTRIBUTED INFERENCE")
        print("=" * 80)
    
    # Synchronize before starting timing
    if WORLD_SIZE > 1:
        dist.barrier()
        
    start_time = time.time()
    
    # Run inference on the specified device for this rank
    results = run_inference_ddp(model, dataset, device=device, max_batches=max_batches)
    
    if WORLD_SIZE > 1:
        dist.barrier()
    
    total_time = time.time() - start_time
    
    if is_main_process():
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(f"Token Accuracy: {results['token_accuracy']:.4f}")
        print(f"Exact Accuracy: {results['exact_accuracy']:.4f} ({results['total_correct_sequences']}/{results['total_sequences']})")
        print(f"Average Inference Steps: {results['avg_inference_steps']:.2f}")
        print(f"Total Batches Processed: {results['num_batches']}")
        print(f"Total sequences: {results['total_sequences']}")
        print(f"Time taken: {total_time:.2f}s")
        print("=" * 80)
        
    return results

# ============================================================================
# Grid Search
# ============================================================================

def grid_search():
    """Run grid search over hyperparameters."""
    
    # Define experiments
    experiments = [
        {"halt_max_steps": 4, "H_cycles": 3, "L_cycles": 6},
        {"halt_max_steps": 8, "H_cycles": 3, "L_cycles": 6},
        {"halt_max_steps": 16, "H_cycles": 3, "L_cycles": 6},
        {"halt_max_steps": 32, "H_cycles": 3, "L_cycles": 6},
        {"halt_max_steps": 48, "H_cycles": 3, "L_cycles": 6},
        {"halt_max_steps": 64, "H_cycles": 3, "L_cycles": 6},
    ]
    BATCH_SIZE = 256*5
    
    all_results = []
    
    if is_main_process():
        print("STARTING GRID SEARCH...")
    
    for exp_config in experiments:
        if is_main_process():
            print(f"\n\n{'=' * 100}")
            print(f"EXPERIMENT: {exp_config}")
            print(f"{'=' * 100}\n")
        
        results = evaluate_with_config(
            checkpoint_path=CHECKPOINT_PATH,
            data_path=DATA_PATH,
            batch_size=BATCH_SIZE, 
            max_batches=None,
            compile_model=False,
            **exp_config
        )
        
        # Only rank 0 collects and saves the results
        if is_main_process():
            results["config"] = exp_config
            all_results.append(results)
    
    # Save results (Rank 0 only)
    if is_main_process():
        print("\n" + "=" * 100)
        print("SAVING RESULTS")
        print("=" * 100)
        
        results_to_save = []
        for res in all_results:
            clean_res = res.copy()
            clean_res.pop('per_batch_metrics', None) 
            results_to_save.append(clean_res)
            
        # Construct filename with dataset name
        dataset_name = os.path.basename(os.path.normpath(DATA_PATH)).split('-')[0] # e.g., 'Maze' or 'Sudoku'
        save_filename = f"{dataset_name}_grid_search_results.json"
        
        print(f"Saving summary of results to {save_filename}...")
        
        try:
            with open(save_filename, "w") as f:
                json.dump(results_to_save, f, indent=4)
            print(f"✓ Successfully saved results to {save_filename}")
        except Exception as e:
            print(f"⚠ An error occurred while saving results: {e}")
            
    return all_results

if __name__ == "__main__":
    # 1. Initialize DDP/set global variables ONCE at the very start
    setup_ddp() 
    
    # 2. Run the main logic
    grid_search()
    
    # 3. torchrun handles cleanup, so we remove dist.destroy_process_group()

