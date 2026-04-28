"""
Evaluation Metrics
"""
import numpy as np
from typing import Dict, List, Any, Tuple


def mean_absolute_error(y_true: List[int], y_pred: List[int]) -> float:
    """Calculate Mean Absolute Error"""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def root_mean_squared_error(y_true: List[int], y_pred: List[int]) -> float:
    """Calculate Root Mean Squared Error"""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def exact_match_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Calculate Exact Match Accuracy (%)"""
    matches = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return (matches / len(y_true)) * 100 if y_true else 0.0


def calculate_counting_metrics(
    ground_truth: List[Dict[str, int]],
    predictions: List[Dict[str, int]]
) -> Dict[str, Any]:
    """
    Calculate counting accuracy metrics.
    
    Args:
        ground_truth: List of dicts with keys 'cars', 'pedestrians', 'bicycles'
        predictions: List of dicts with same keys
        
    Returns:
        Dict with MAE, RMSE, Exact Match for each category
    """
    categories = ['cars', 'pedestrians', 'bicycles']
    results = {}
    
    for cat in categories:
        y_true = [gt.get(cat, 0) for gt in ground_truth]
        y_pred = [p.get(cat, 0) for p in predictions]
        
        results[cat] = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': root_mean_squared_error(y_true, y_pred),
            'exact_match_pct': exact_match_accuracy(y_true, y_pred)
        }
    
    # Overall metrics
    all_y_true = []
    all_y_pred = []
    for cat in categories:
        all_y_true.extend([gt.get(cat, 0) for gt in ground_truth])
        all_y_pred.extend([p.get(cat, 0) for p in predictions])
    
    results['overall'] = {
        'mae': mean_absolute_error(all_y_true, all_y_pred),
        'rmse': root_mean_squared_error(all_y_true, all_y_pred),
        'exact_match_pct': exact_match_accuracy(all_y_true, all_y_pred)
    }
    
    return results


def calculate_consistency(
    atomic_responses: List[Dict[str, str]],
    composite_responses: List[str]
) -> Dict[str, float]:
    """
    Calculate consistency between atomic and composite query responses.
    """
    # This is a simplified version - in practice you'd parse both responses
    # and compare the extracted counts
    return {
        'atomic_vs_composite_agreement': 0.0  # Placeholder
    }


def calculate_description_quality(
    descriptions: List[str],
    ground_truth: List[Dict[str, int]]
) -> Dict[str, float]:
    """
    Calculate description quality metrics.
    Requires human evaluation or LLM-as-judge.
    """
    return {
        'correctness_avg': 0.0,
        'completeness_avg': 0.0,
        'hallucination_avg': 0.0
    }
