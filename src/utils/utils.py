import os
import json
from typing import Dict, Callable, Iterable, Tuple, Any, List
import numpy as np

def load_eval_json(path):
    with open(path, "r") as f:
        return json.load(f)
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    
def get(d: Dict[str, Any], *keys, default=None):
    """Safe nested dict get: _get(d, 'a','b') == d.get('a',{}).get('b',{})..."""
    cur = d
    for k in keys:
        if isinstance(k, tuple):
            for kk in k:
                if not isinstance(cur, dict) or kk not in cur:
                    return default
                cur = cur[kk]
        else:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
    return cur

def safe_slice_pair(y_gt: List[float], y_pred: List[float], start: int) -> Tuple[np.ndarray, np.ndarray]:
    """Slice and align GT and prediction from 'start', return as numpy arrays."""
    if y_gt is None or y_pred is None:
        return np.array([]), np.array([])
    n = min(len(y_gt), len(y_pred))
    if start >= n:
        return np.array([]), np.array([])
    ygt = np.asarray(y_gt[start:n], dtype=float)
    ypd = np.asarray(y_pred[start:n], dtype=float)
    m = min(len(ygt), len(ypd))
    return ygt[:m], ypd[:m]