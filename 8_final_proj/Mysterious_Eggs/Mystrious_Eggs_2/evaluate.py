#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import tensorflow as tf
import numpy as np
from datetime import datetime

from local_data_pipeline import LocalPetDataPipeline
from utils import ModelEvaluator, SegmentationMetrics, VisualizationUtils

def find_latest_model(model_dir='results'):
    """Find the most recent trained model"""
    model_patterns = [
        f'{model_dir}/final_model_*.h5',
        'checkpoints/best_model_*.h5'
    ]
    
    all_models = []
    for pattern in model_patterns:
        all_models.extend(glob.glob(pattern))
    
    if not all_models:
        raise FileNotFoundError("No trained models found. Please train a model first.")
    
    latest_model = max(all_models, key=os.path.getctime)
    return latest_model

def load_test_dataset(img_size=(128, 128), batch_size=16):
    """Load test dataset for evaluation"""
    print("Loading local test dataset...")
    pipeline = LocalPetDataPipeline(
        dataset_path="./dataset/cat_and_dog_dataset",
        img_size=img_size, 
        batch_size=batch_size
    )
    
    try:
        # Load dataset with train/val/test splits
        train_ds, val_ds, test_ds, info = pipeline.load_dataset()
        
        # Use test set if available, otherwise validation set
        if test_ds is not None and info['splits']['test'] > 0:
            print(f"Using test set with {info['splits']['test']} samples")
            return test_ds, info
        else:
            print(f"Using validation set with {info['splits']['validation']} samples")
            return val_ds, info
            
    except Exception as e:
        print(f"Error loading local dataset: {e}")
        raise e

def evaluate_single_model(model_path, img_size=(128, 128), batch_size=16, save_dir=None):
    """Evaluate a single model"""
    
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"evaluation_{timestamp}"
    
    print(f"Evaluating model: {model_path}")
    print(f"Results will be saved to: {save_dir}")
    print(f"Using local dataset: ./dataset/cat_and_dog_dataset")
    
    # Load model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded successfully")
        print(f"Model parameters: {model.count_params():,}")
        print(f"Input shape: {model.input_shape}")
        print(f"Output shape: {model.output_shape}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Load test dataset
    try:
        test_ds, dataset_info = load_test_dataset(img_size, batch_size)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return None
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator(model, test_ds)
    results = evaluator.evaluate_model(save_dir)
    
    return results

def compare_models(model_paths, img_size=(128, 128), batch_size=16):
    """Compare multiple models"""
    print(f"Comparing {len(model_paths)} models...")
    
    results_comparison = {}
    
    for i, model_path in enumerate(model_paths):
        print(f"\n{'='*50}")
        print(f"Evaluating Model {i+1}/{len(model_paths)}")
        print(f"Path: {model_path}")
        print(f"{'='*50}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"evaluation_model_{i+1}_{timestamp}"
        
        try:
            results = evaluate_single_model(model_path, img_size, batch_size, save_dir)
            if results:
                results_comparison[model_path] = results
        except Exception as e:
            print(f"Error evaluating {model_path}: {e}")
            continue
    
    # Create comparison summary
    if results_comparison:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        print(f"{'Model':<30} {'Pixel Acc':<12} {'Mean IoU':<12} {'Dice Coeff':<12}")
        print("-" * 66)
        
        for model_path, results in results_comparison.items():
            model_name = os.path.basename(model_path)[:25] + "..." if len(os.path.basename(model_path)) > 28 else os.path.basename(model_path)
            pixel_acc = results['pixel_accuracy']['mean']
            mean_iou = results['mean_iou']['mean']
            dice_coeff = results['dice_coefficient']['mean']
            
            print(f"{model_name:<30} {pixel_acc:<12.4f} {mean_iou:<12.4f} {dice_coeff:<12.4f}")
        
        # Save comparison results
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_file = f"model_comparison_{timestamp}.json"
        
        with open(comparison_file, 'w') as f:
            # Convert results to JSON-serializable format
            serializable_results = {}
            for path, results in results_comparison.items():
                serializable_results[path] = {
                    'pixel_accuracy': results['pixel_accuracy'],
                    'mean_iou': results['mean_iou'],
                    'dice_coefficient': results['dice_coefficient'],
                    'num_samples': results['num_samples']
                }
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nComparison results saved to: {comparison_file}")
    
    return results_comparison

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained segmentation models')
    parser.add_argument('--model', type=str, help='Path to specific model file')
    parser.add_argument('--compare-all', action='store_true', help='Compare all available models')
    parser.add_argument('--img-size', type=int, nargs=2, default=[128, 128], help='Image size (height width)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--save-dir', type=str, help='Directory to save results')
    
    args = parser.parse_args()
    
    img_size = tuple(args.img_size)
    
    try:
        if args.compare_all:
            # Find all available models
            model_patterns = [
                'results/final_model_*.h5',
                'checkpoints/best_model_*.h5'
            ]
            
            all_models = []
            for pattern in model_patterns:
                all_models.extend(glob.glob(pattern))
            
            if not all_models:
                print("No trained models found for comparison.")
                print("Please train some models first using: python train.py")
                return
            
            print(f"Found {len(all_models)} models for comparison")
            compare_models(all_models, img_size, args.batch_size)
            
        elif args.model:
            # Evaluate specific model
            if not os.path.exists(args.model):
                print(f"Model file not found: {args.model}")
                return
            
            evaluate_single_model(args.model, img_size, args.batch_size, args.save_dir)
            
        else:
            # Evaluate latest model
            try:
                latest_model = find_latest_model()
                print(f"Found latest model: {latest_model}")
                evaluate_single_model(latest_model, img_size, args.batch_size, args.save_dir)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                print("\nUsage examples:")
                print("1. Train a model first: python train.py")
                print("2. Evaluate specific model: python evaluate.py --model path/to/model.h5")
                print("3. Compare all models: python evaluate.py --compare-all")
    
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()