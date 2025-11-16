import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
import copy
import time

# Add current directory to path
sys.path.append(os.getcwd())

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class
from models.losses import IGNORE_LABEL_ID

# ============================================================================
# Configuration
# ============================================================================

CHECKPOINT_PATH = "/home/zakarianarjis/workspace/TinyRecursiveModels/checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/pretrain_mlp_t_sudoku/step_65100"
DATA_PATH = "data/sudoku-extreme-1k-aug-1000"

# Hyperparameters to experiment with
INFERENCE_CONFIG = {
    "halt_max_steps": 16,  # Try: 8, 16, 32, 64
    "H_cycles": 3,          # Try: 1, 2, 3, 4, 5
    "L_cycles": 6,          # Try: 3, 6, 9, 12
    "batch_size": 1024,     # Try: 32, 64, 128, 256, 512, 1024
}

# ============================================================================
# Helper Functions
# ============================================================================

def load_model_from_checkpoint(checkpoint_path: str, config_overrides: Dict[str, Any] = None):
    """Load model from checkpoint with optional config overrides."""
    
    # Load the config from the checkpoint directory
    config_file = os.path.join(os.path.dirname(checkpoint_path), "all_config.yaml")
    
    import yaml
    with open(config_file, "r") as f:
        full_config = yaml.safe_load(f)
    
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


def create_eval_dataset(data_path: str, batch_size: int, split: str = "test"):
    """Create evaluation dataset."""
    
    dataset = PuzzleDataset(
        PuzzleDatasetConfig(
            seed=0,
            dataset_paths=[data_path],
            global_batch_size=batch_size,
            test_set_mode=True,
            epochs_per_iter=1,
            rank=0,
            num_replicas=1,
        ),
        split=split
    )
    
    return dataset, dataset.metadata


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute accuracy metrics."""
    
    # Get predictions
    preds = torch.argmax(logits, dim=-1)
    
    # Create mask for valid labels (ignore IGNORE_LABEL_ID)
    valid_mask = (labels != IGNORE_LABEL_ID)
    
    # Token-level accuracy
    correct_tokens = (preds == labels) & valid_mask
    token_accuracy = correct_tokens.sum().item() / valid_mask.sum().item() if valid_mask.sum().item() > 0 else 0.0
    
    # Sequence-level exact match accuracy
    # A sequence is correct if all its tokens are correct
    correct_per_sequence = correct_tokens.sum(dim=1) == valid_mask.sum(dim=1)
    exact_accuracy = correct_per_sequence.float().mean().item()
    
    return {
        "token_accuracy": token_accuracy,
        "exact_accuracy": exact_accuracy,
        "num_valid_tokens": valid_mask.sum().item(),
        "num_sequences": labels.shape[0],
        "num_correct_sequences": correct_per_sequence.sum().item(),
    }


def run_inference(
    model: nn.Module,
    dataset: PuzzleDataset,
    device: str = "cuda",
    max_batches: int = None
) -> Dict[str, Any]:
    """Run inference on the dataset and compute metrics."""
    
    model.eval()
    all_metrics = []
    batch_count = 0
    total_sequences_processed = 0
    
    with torch.no_grad():
        # Iterate through the dataset - it yields (set_name, batch, global_batch_size)
        dataset_iter = iter(dataset)
        
        while True:
            if max_batches and batch_count >= max_batches:
                break
            
            # Try to get next batch with timeout detection
            try:
                print(f"\n[DEBUG] Attempting to fetch batch {batch_count + 1}...")
                start_time = time.time()
                
                batch_data = next(dataset_iter)
                
                fetch_time = time.time() - start_time
                print(f"[DEBUG] Batch fetched in {fetch_time:.2f}s")
                
                set_name, batch, global_batch_size = batch_data
                
            except StopIteration:
                print(f"\n[DEBUG] Dataset iteration complete - no more batches")
                break
            except Exception as e:
                print(f"\n[ERROR] Exception while fetching batch: {e}")
                import traceback
                traceback.print_exc()
                break
            
            batch_count += 1
            batch_size = batch['inputs'].shape[0]
            total_sequences_processed += batch_size
            
            print(f"\nProcessing batch {batch_count}: {set_name}")
            print(f"  Batch size: {batch_size}")
            print(f"  Total sequences processed: {total_sequences_processed}")
            
            # Move batch to device
            start_time = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            print(f"  [DEBUG] Moved to device in {time.time() - start_time:.2f}s")
            
            # Initialize carry
            start_time = time.time()
            with torch.device(device):
                carry = model.initial_carry(batch)
            print(f"  [DEBUG] Initialized carry in {time.time() - start_time:.2f}s")
            
            # Run inference until all sequences halt
            inference_steps = 0
            max_inference_steps = 100  # Safety limit
            
            start_time = time.time()
            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, 
                    batch=batch, 
                    return_keys=["logits"]
                )
                inference_steps += 1
                
                if all_finish or inference_steps >= max_inference_steps:
                    if inference_steps >= max_inference_steps:
                        print(f"  WARNING: Reached max inference steps ({max_inference_steps})")
                    break
            
            inference_time = time.time() - start_time
            print(f"  Completed in {inference_steps} steps ({inference_time:.2f}s)")
            
            # Compute accuracy metrics
            start_time = time.time()
            logits = preds["logits"]
            labels = batch["labels"]
            
            batch_metrics = compute_metrics(logits, labels)
            batch_metrics["inference_steps"] = inference_steps
            batch_metrics["set_name"] = set_name
            
            all_metrics.append(batch_metrics)
            
            print(f"  Token Accuracy: {batch_metrics['token_accuracy']:.4f}")
            print(f"  Exact Accuracy: {batch_metrics['exact_accuracy']:.4f} ({batch_metrics['num_correct_sequences']}/{batch_metrics['num_sequences']})")
            print(f"  [DEBUG] Metrics computed in {time.time() - start_time:.2f}s")
            
            # Debug: Print first few predictions vs labels (only for first batch)
            if batch_count == 1:
                print("\n  Sample predictions (first sequence, first 20 tokens):")
                preds_sample = torch.argmax(logits[0, :20], dim=-1).cpu().numpy()
                labels_sample = labels[0, :20].cpu().numpy()
                print(f"    Predictions: {preds_sample}")
                print(f"    Labels:      {labels_sample}")
                print(f"    Match:       {preds_sample == labels_sample}")
            
            # Clear GPU cache periodically
            if batch_count % 10 == 0:
                torch.cuda.empty_cache()
                print(f"  [DEBUG] Cleared GPU cache")
    
    print(f"\n✓ Completed processing {batch_count} batches, {total_sequences_processed} total sequences")
    
    if not all_metrics:
        return {
            "token_accuracy": 0.0,
            "exact_accuracy": 0.0,
            "avg_inference_steps": 0.0,
            "num_batches": 0,
            "total_valid_tokens": 0,
            "total_sequences": 0,
            "per_batch_metrics": [],
        }
    
    # Aggregate metrics
    total_valid_tokens = sum(m["num_valid_tokens"] for m in all_metrics)
    total_sequences = sum(m["num_sequences"] for m in all_metrics)
    total_correct_sequences = sum(m["num_correct_sequences"] for m in all_metrics)
    
    # Weighted average for token accuracy
    weighted_token_acc = sum(
        m["token_accuracy"] * m["num_valid_tokens"] for m in all_metrics
    ) / total_valid_tokens if total_valid_tokens > 0 else 0.0
    
    # Exact accuracy (total correct / total sequences)
    exact_accuracy = total_correct_sequences / total_sequences if total_sequences > 0 else 0.0
    
    # Average inference steps
    avg_inference_steps = sum(m["inference_steps"] for m in all_metrics) / len(all_metrics) if all_metrics else 0.0
    
    return {
        "token_accuracy": weighted_token_acc,
        "exact_accuracy": exact_accuracy,
        "avg_inference_steps": avg_inference_steps,
        "num_batches": len(all_metrics),
        "total_valid_tokens": total_valid_tokens,
        "total_sequences": total_sequences,
        "total_correct_sequences": total_correct_sequences,
        "per_batch_metrics": all_metrics,
    }


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
    compile_model: bool = False
):
    """
    Evaluate model with custom hyperparameters.
    
    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to evaluation data
        halt_max_steps: Maximum number of ACT steps
        H_cycles: Number of high-level cycles
        L_cycles: Number of low-level cycles
        batch_size: Batch size for evaluation
        max_batches: Maximum number of batches to process (None for all)
        device: Device to run on
        compile_model: Whether to compile the model
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    print("=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data Path: {data_path}")
    print(f"halt_max_steps: {halt_max_steps}")
    print(f"H_cycles: {H_cycles}")
    print(f"L_cycles: {L_cycles}")
    print(f"batch_size: {batch_size}")
    print(f"max_batches: {max_batches if max_batches else 'All'}")
    print(f"Device: {device}")
    print(f"Compile model: {compile_model}")
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
    
    # Create dataset
    print("\nLoading dataset...")
    dataset, metadata = create_eval_dataset(data_path, batch_size)
    
    # Update model config with dataset metadata
    model_cfg["vocab_size"] = metadata.vocab_size
    model_cfg["seq_len"] = metadata.seq_len
    model_cfg["num_puzzle_identifiers"] = metadata.num_puzzle_identifiers
    
    print(f"Dataset loaded successfully!")
    print(f"  Total puzzles: {metadata.total_puzzles}")
    print(f"  Mean examples per puzzle: {metadata.mean_puzzle_examples:.1f}")
    print(f"  Estimated total examples: ~{int(metadata.total_puzzles * metadata.mean_puzzle_examples)}")
    print(f"  Vocab size: {metadata.vocab_size}")
    print(f"  Sequence length: {metadata.seq_len}")
    print(f"  Num puzzle identifiers: {metadata.num_puzzle_identifiers}")
    print(f"  Sets: {metadata.sets}")
    
    # Load model
    print("\nLoading model...")
    print(f"  Model class: {model_name}")
    print(f"  Loss class: {loss_name}")
    
    model_cls = load_model_class(model_name)
    loss_head_cls = load_model_class(loss_name)
    
    with torch.device(device):
        model = model_cls(model_cfg)
        print(f"\nBase model architecture:")
        print(f"  Hidden size: {model_cfg.get('hidden_size', 'N/A')}")
        print(f"  Num heads: {model_cfg.get('num_heads', 'N/A')}")
        print(f"  H_cycles: {model_cfg.get('H_cycles', 'N/A')}")
        print(f"  L_cycles: {model_cfg.get('L_cycles', 'N/A')}")
        print(f"  L_layers: {model_cfg.get('L_layers', 'N/A')}")
        
        model = loss_head_cls(model, **loss_kwargs)
        
        # Load checkpoint weights
        print(f"\nLoading weights from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # Strip _orig_mod. prefix if present (from torch.compile)
        state_dict = strip_compiled_prefix(state_dict)
        
        # Try to load with strict=True
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✓ Weights loaded successfully (strict mode)")
        except Exception as e:
            print(f"⚠ Warning: Could not load with strict=True: {e}")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"  Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"  Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        
        if compile_model and "DISABLE_COMPILE" not in os.environ:
            print("\nCompiling model...")
            model = torch.compile(model)
        
        model.eval()
    
    print(f"\n✓ Model loaded successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Run inference
    print("\n" + "=" * 80)
    print("RUNNING INFERENCE")
    print("=" * 80)
    
    results = run_inference(model, dataset, device=device, max_batches=max_batches)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Token Accuracy: {results['token_accuracy']:.4f}")
    print(f"Exact Accuracy: {results['exact_accuracy']:.4f} ({results['total_correct_sequences']}/{results['total_sequences']})")
    print(f"Average Inference Steps: {results['avg_inference_steps']:.2f}")
    print(f"Total Batches Processed: {results['num_batches']}")
    print(f"Total Sequences: {results['total_sequences']}")
    print(f"Total Valid Tokens: {results['total_valid_tokens']}")
    print("=" * 80)
    
    return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Single evaluation - process ALL batches
    results = evaluate_with_config(
        checkpoint_path=CHECKPOINT_PATH,
        data_path=DATA_PATH,
        halt_max_steps=INFERENCE_CONFIG["halt_max_steps"],
        H_cycles=INFERENCE_CONFIG["H_cycles"],
        L_cycles=INFERENCE_CONFIG["L_cycles"],
        batch_size=INFERENCE_CONFIG["batch_size"],
        max_batches=None,  # Process all batches
        compile_model=False,
    )

# ============================================================================
# Grid Search
# ============================================================================

def grid_search():
    """Run grid search over hyperparameters."""
    
    experiments = [
        {"halt_max_steps": 8, "H_cycles": 2, "L_cycles": 4},
        {"halt_max_steps": 16, "H_cycles": 3, "L_cycles": 6},
        {"halt_max_steps": 32, "H_cycles": 4, "L_cycles": 8},
        {"halt_max_steps": 64, "H_cycles": 5, "L_cycles": 10},
    ]
    
    all_results = []
    
    for exp_config in experiments:
        print(f"\n\n{'=' * 100}")
        print(f"EXPERIMENT: {exp_config}")
        print(f"{'=' * 100}\n")
        
        results = evaluate_with_config(
            checkpoint_path=CHECKPOINT_PATH,
            data_path=DATA_PATH,
            batch_size=1024,
            max_batches=None,  # Full evaluation
            **exp_config
        )
        
        results["config"] = exp_config
        all_results.append(results)
    
    # Print comparison
    print("\n\n" + "=" * 100)
    print("COMPARISON OF ALL EXPERIMENTS")
    print("=" * 100)
    
    for i, (exp, res) in enumerate(zip(experiments, all_results)):
        print(f"\nExperiment {i+1}: {exp}")
        print(f"  Token Accuracy: {res['token_accuracy']:.4f}")
        print(f"  Exact Accuracy: {res['exact_accuracy']:.4f}")
        print(f"  Avg Steps: {res['avg_inference_steps']:.2f}")
    
    return all_results

# Uncomment to run grid search
# results = grid_search()