"""Post-processing utilities for OCR decoding."""
from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np
import torch


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str]
) -> List[Tuple[str, float]]:
    """CTC decode predictions with confidence scores using greedy decoding.
    
    Args:
        preds: Log-softmax predictions of shape [batch_size, time_steps, num_classes].
        idx2char: Index to character mapping.
    
    Returns:
        List of (predicted_string, confidence_score) tuples.
    """
    probs = preds.exp()
    max_probs, indices = probs.max(dim=2)
    indices_np = indices.detach().cpu().numpy()
    max_probs_np = max_probs.detach().cpu().numpy()
    
    batch_size, time_steps = indices_np.shape
    results: List[Tuple[str, float]] = []
    
    for batch_idx in range(batch_size):
        path = indices_np[batch_idx]
        probs_b = max_probs_np[batch_idx]
        
        # Group consecutive identical characters and filter blanks
        # groupby returns (key, group_iterator) pairs
        pred_chars = []
        confidences = []
        time_idx = 0
        
        for char_idx, group in groupby(path):
            group_list = list(group)
            group_size = len(group_list)
            
            if char_idx != 0:  # Skip blank
                pred_chars.append(idx2char.get(char_idx, ''))
                # Get maximum probability from this group
                group_probs = probs_b[time_idx:time_idx + group_size]
                confidences.append(float(np.max(group_probs)))
            
            time_idx += group_size
        
        pred_str = "".join(pred_chars)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        results.append((pred_str, confidence))
    
    return results
