from __future__ import annotations

import math
from collections.abc import Sequence


LANGUAGE_DETECTION_PRIOR_WEIGHT = 0.1


def select_language_candidate(
    languages: Sequence[str],
    average_log_probabilities: Sequence[float],
    detection_probabilities: Sequence[float],
) -> int:
    if not languages or not (
        len(languages) == len(average_log_probabilities) == len(detection_probabilities)
    ):
        raise ValueError("language candidate inputs must be non-empty and equal-length")

    scores = [
        average_log_probability
        + LANGUAGE_DETECTION_PRIOR_WEIGHT * math.log(max(detection_probability, 1e-8))
        for average_log_probability, detection_probability in zip(
            average_log_probabilities, detection_probabilities
        )
    ]
    return max(range(len(languages)), key=lambda index: (scores[index], -index))
