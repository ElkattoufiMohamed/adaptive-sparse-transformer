"""
Professional training infrastructure for transformer experiments.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from evaluation.metrics import compute_efficiency_metrics
from utils.helpers import load_model, save_model, set_seed

logger = logging.getLogger(__name__)

class AdaptiveTransformerTrainer:
    """
    Professional trainer for adaptive sparse transformer experiments.
    
    Features:
    - Automatic mixed precision training
    - Comprehensive logging and metrics
    - Model checkpointing and resume capability
    - Attention pattern analysis
    - Efficiency benchmarking
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        experiment_dir: Path,
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.use_wandb = use_wandb
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        training_config = config['training']
        self.num_epochs = training_config['num_epochs']
        self.gradient_clip_norm = training_config['gradient_clip_norm']
        self.eval_steps = config['evaluation']['eval_steps']
        self.save_steps = config['evaluation']['save_steps']
        self.logging_steps = config['evaluation']['logging_steps']
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Mixed precision training
        self.use_amp = config['hardware'].get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_eval_metric = 0.0
        self.train_metrics = []
        self.eval_metrics = []
        
        # Initialize wandb if requested
        if self.use_wandb:
            self._setup_wandb()
        
        logger.info(f"Trainer initialized - Device: {device}, AMP: {self.use_amp}")
    
    def _setup_optimizer(self):
        """Initialize optimizer with proper parameter grouping."""
        training_config = self.config['training']
        
        # Separate parameters for different learning rates if needed
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': training_config['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=float(training_config['learning_rate']),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        logger.info(f"Optimizer: AdamW with LR={training_config['learning_rate']}")
    
    def _setup_scheduler(self):
        """Initialize learning rate scheduler."""
        scheduler_config = self.config.get('scheduler', {})
        schedule_type = scheduler_config.get('type', 'cosine')
        
        total_steps = len(self.train_loader) * self.num_epochs
        warmup_steps = self.config['training']['warmup_steps']
        
        if schedule_type == 'cosine':
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif schedule_type == 'linear':
            from transformers import get_linear_schedule_with_warmup
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = None
        
        logger.info(f"Scheduler: {schedule_type}, Warmup: {warmup_steps} steps")
    
    def _setup_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb_config = self.config.get('wandb', {})
        
        wandb.init(
            project=wandb_config.get('project', 'adaptive-sparse-transformer'),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('name'),
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('notes', ''),
            config=self.config
        )
        
        # Watch model for gradient tracking
        wandb.watch(self.model, log="all", log_freq=1000)
        
        logger.info("Wandb initialized")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with autocast(enabled=self.use_amp):
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                return_attention_info=True
            )
            
            logits = outputs['logits']
            loss = nn.CrossEntropyLoss()(logits, batch['labels'])
        
        # ADD DEBUG CODE HERE - Right after loss computation
        print(f"DEBUG: Raw loss = {loss.item()}")
        print(f"DEBUG: Loss is finite: {torch.isfinite(loss)}")
        print(f"DEBUG: Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        print(f"DEBUG: Logits mean: {logits.mean().item():.3f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("CRITICAL: NaN/Inf loss detected - skipping this batch")
            return {
                'loss': 0.0,  # Return dummy values
                'accuracy': 0.0,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
    }
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                
                # ADD MORE DEBUG HERE - Check gradients
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                print(f"DEBUG: Gradient norm before clipping: {total_norm}")
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            if self.gradient_clip_norm > 0:
                # ADD MORE DEBUG HERE - Check gradients  
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                print(f"DEBUG: Gradient norm before clipping: {total_norm}")
            
            self.optimizer.step()
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        self.optimizer.zero_grad()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            accuracy = (predictions == batch['labels']).float().mean()
        
        # Extract attention information
        attention_info = outputs.get('attention_info', [])
        avg_attention_stats = self._compute_attention_stats(attention_info)
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            **avg_attention_stats
    }
    
    def _compute_attention_stats(self, attention_info: List[Dict]) -> Dict[str, float]:
        """Compute average attention pattern statistics across all layers."""
        if not attention_info:
            return {}
        
        # Aggregate pattern usage across all layers
        local_ratios = []
        global_ratios = []
        sparse_ratios = []
        
        for layer_info in attention_info:
            if 'local_ratio' in layer_info:
                local_ratios.append(layer_info['local_ratio'])
                global_ratios.append(layer_info['global_ratio'])
                sparse_ratios.append(layer_info.get('sparse_ratio', 0.0))
        
        if local_ratios:
            return {
                'avg_local_ratio': np.mean(local_ratios),
                'avg_global_ratio': np.mean(global_ratios),
                'avg_sparse_ratio': np.mean(sparse_ratios)
            }
        
        return {}
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_attention_stats = []
        
        logger.info("Running evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    return_attention_info=True
                )
                
                logits = outputs['logits']
                loss = nn.CrossEntropyLoss()(logits, batch['labels'])
                
                total_loss += loss.item()
                
                # Collect predictions and labels
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                
                # Collect attention statistics
                attention_info = outputs.get('attention_info', [])
                batch_attention_stats = self._compute_attention_stats(attention_info)
                all_attention_stats.append(batch_attention_stats)
        
        # Compute metrics
        avg_loss = total_loss / len(self.eval_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Average attention statistics
        avg_attention_stats = {}
        if all_attention_stats and all_attention_stats[0]:
            for key in all_attention_stats[0].keys():
                values = [stats.get(key, 0) for stats in all_attention_stats if key in stats]
                avg_attention_stats[f"eval_{key}"] = np.mean(values)
        
        eval_metrics = {
            'eval_loss': avg_loss,
            'eval_accuracy': accuracy,
            'eval_precision': precision,
            'eval_recall': recall,
            'eval_f1': f1,
            **avg_attention_stats
        }
        
        logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return eval_metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint with metrics."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'best_eval_metric': self.best_eval_metric
        }
        
        # Save regular checkpoint
        checkpoint_path = self.experiment_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.experiment_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with metric: {metrics.get('eval_accuracy', 0):.4f}")
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self) -> Dict[str, List[float]]:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Total epochs: {self.num_epochs}")
        logger.info(f"Steps per epoch: {len(self.train_loader)}")
        logger.info(f"Total steps: {len(self.train_loader) * self.num_epochs}")
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            logger.info(f"\n📚 Epoch {epoch + 1}/{self.num_epochs}")
            
            # Training phase
            epoch_metrics = []
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            
            print(f"DEBUG: Starting epoch {epoch + 1}, about to iterate through {len(self.train_loader)} batches")  # ADD THIS
            
            for batch_idx, batch in enumerate(progress_bar):
                try:  # ADD TRY-CATCH
                    print(f"DEBUG: Processing batch {batch_idx + 1}/{len(self.train_loader)}")  # ADD THIS
                    
                    step_metrics = self.train_step(batch)
                    print(f"DEBUG: train_step completed for batch {batch_idx + 1}")  # ADD THIS
                    
                    epoch_metrics.append(step_metrics)
                    self.global_step += 1
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f"{step_metrics['loss']:.4f}",
                        'acc': f"{step_metrics['accuracy']:.3f}",
                        'lr': f"{step_metrics['learning_rate']:.2e}"
                    })
                    
                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        print(f"DEBUG: Logging at step {self.global_step}")  # ADD THIS
                        avg_metrics = self._average_metrics(epoch_metrics[-self.logging_steps:])
                        self.train_metrics.append(avg_metrics)
                        
                        if self.use_wandb:
                            wandb.log({f"train_{k}": v for k, v in avg_metrics.items()}, 
                                    step=self.global_step)
                    
                    # Evaluation
                    if self.global_step % self.eval_steps == 0:
                        print(f"DEBUG: Starting evaluation at step {self.global_step}")  # ADD THIS
                        eval_metrics = self.evaluate()
                        print(f"DEBUG: Evaluation completed at step {self.global_step}")  # ADD THIS
                        
                        self.eval_metrics.append(eval_metrics)
                        
                        # Check if best model
                        current_metric = eval_metrics.get('eval_accuracy', 0)
                        is_best = current_metric > self.best_eval_metric
                        if is_best:
                            self.best_eval_metric = current_metric
                        
                        if self.use_wandb:
                            wandb.log(eval_metrics, step=self.global_step)
                        
                        # Save checkpoint
                        if self.global_step % self.save_steps == 0:
                            print(f"DEBUG: Saving checkpoint at step {self.global_step}")  # ADD THIS
                            self.save_checkpoint(eval_metrics, is_best=is_best)
                            
                except Exception as e:  # ADD EXCEPTION HANDLING
                    print(f"ERROR: Exception at batch {batch_idx + 1}: {e}")
                    print(f"ERROR: Exception type: {type(e)}")
                    import traceback
                    traceback.print_exc()
                    raise  # Re-raise to see full error
        
        # Final evaluation
        logger.info("\n🏁 Training completed! Running final evaluation...")
        final_eval_metrics = self.evaluate()
        
        # Save final model
        self.save_checkpoint(final_eval_metrics, is_best=False)
        
        training_time = time.time() - start_time
        logger.info(f"Total training time: {training_time:.2f} seconds")
        
        # Create training summary
        summary = {
            'final_eval_metrics': final_eval_metrics,
            'best_eval_metric': self.best_eval_metric,
            'total_steps': self.global_step,
            'training_time': training_time,
            'train_metrics_history': self.train_metrics,
            'eval_metrics_history': self.eval_metrics
        }
        
        # Save summary
        summary_path = self.experiment_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.use_wandb:
            wandb.log({"final_eval_accuracy": final_eval_metrics.get('eval_accuracy', 0)})
            wandb.finish()
        
        return summary
    
    def _average_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average metrics across multiple steps."""
        if not metrics_list:
            return {}
        
        avg_metrics = {}
        for key in metrics_list[0].keys():
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                avg_metrics[key] = np.mean(values)
        
        return avg_metrics

class BaselineTransformerTrainer(AdaptiveTransformerTrainer):
    """
    Trainer for baseline transformer (same interface, no attention analysis).
    """
    
    def _compute_attention_stats(self, attention_info: List[Dict]) -> Dict[str, float]:
        """Baseline doesn't have adaptive attention stats."""
        return {}

def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    config: Dict[str, Any],
    experiment_name: str,
    device: Optional[torch.device] = None
) -> AdaptiveTransformerTrainer:
    """Factory function to create appropriate trainer."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create experiment directory
    experiment_dir = Path("results") / "experiments" / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine trainer type based on model
    model_name = model.__class__.__name__
    if "Adaptive" in model_name:
        trainer_class = AdaptiveTransformerTrainer
    else:
        trainer_class = BaselineTransformerTrainer
    
    trainer = trainer_class(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        config=config,
        device=device,
        experiment_dir=experiment_dir,
        use_wandb=config.get('wandb', {}).get('project') is not None
    )
    
    logger.info(f"Created {trainer_class.__name__} for {experiment_name}")
    
    return trainer