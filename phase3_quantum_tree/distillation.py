"""
Layer 3 prep: distill classical + quantum outputs into a student dataset.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def blend(p_teacher: float, q_quantum: float) -> float:
    """
    Blend teacher probability and quantum score using confidence-adaptive lambda.

    Lambda schedule:
      p_teacher > 0.80  -> lambda=0.85  (high confidence: trust teacher heavily)
      p_teacher > 0.40  -> lambda=0.50  (uncertain region: equal quantum influence)
      p_teacher <= 0.40 -> lambda=0.70  (very low confidence: partial quantum boost,
                                         but not full quantum dominance to avoid
                                         instability when teacher has no signal)

    All three cases of this confidence-adaptive lambda schedule (including the
    lambda=0.70 branch for p_T <= 0.40) are described in the current manuscript as a
    fixed, hand-set design choice that is not tuned per seed.
    """
    if p_teacher > 0.8:
        lam = 0.85
    elif p_teacher > 0.4:
        lam = 0.50
    else:
        lam = 0.70
    return float(lam * p_teacher + (1.0 - lam) * q_quantum)


def build_distilled_dataset(
    features: np.ndarray,
    states: List[Dict[str, float]],
    labels: np.ndarray,
    teacher_probs: np.ndarray,
    quantum_scores: np.ndarray,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for index, state in enumerate(states):
        teacher_prob = float(teacher_probs[index])
        quantum_score = float(quantum_scores[index])
        distilled_target = blend(teacher_prob, quantum_score)
        rows.append(
            {
                "int_heading": float(state["int_heading"]),
                "int_altitude": float(state["int_altitude"]),
                "int_speed": float(state["int_speed"]),
                "int_x_offset": float(state["int_x_offset"]),
                "feature_heading": float(features[index, 0]),
                "feature_altitude": float(features[index, 1]),
                "feature_speed": float(features[index, 2]),
                "feature_x_offset": float(features[index, 3]),
                "teacher_prob": teacher_prob,
                "quantum_score": quantum_score,
                "distilled_target": distilled_target,
                "hard_label": float(labels[index]),
            }
        )
    return rows
