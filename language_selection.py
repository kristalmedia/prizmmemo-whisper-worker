from __future__ import annotations

import math
from collections.abc import Sequence


LANGUAGE_DETECTION_PRIOR_WEIGHT = 0.1
PRIMARY_LANGUAGE_BONUS = 0.12
CONTIGUOUS_LANGUAGE_BONUS = 0.08


def score_language_candidates(
    languages: Sequence[str],
    average_log_probabilities: Sequence[float],
    detection_probabilities: Sequence[float],
    *,
    primary_language: str,
    previous_language: str | None = None,
    continuous_with_previous: bool = False,
) -> list[dict[str, float]]:
    if not languages or not (
        len(languages) == len(average_log_probabilities) == len(detection_probabilities)
    ):
        raise ValueError("language candidate inputs must be non-empty and equal-length")
    if primary_language not in languages:
        raise ValueError("primary_language must be one of the language candidates")
    if previous_language is not None and previous_language not in languages:
        raise ValueError("previous_language must be one of the language candidates")

    scores: list[dict[str, float]] = []
    for language, average_log_probability, detection_probability in zip(
        languages,
        average_log_probabilities,
        detection_probabilities,
    ):
        detector_adjustment = LANGUAGE_DETECTION_PRIOR_WEIGHT * math.log(
            max(detection_probability, 1e-8)
        )
        primary_adjustment = (
            PRIMARY_LANGUAGE_BONUS if language == primary_language else 0.0
        )
        continuity_adjustment = (
            CONTIGUOUS_LANGUAGE_BONUS
            if continuous_with_previous and language == previous_language
            else 0.0
        )
        base_score = average_log_probability + detector_adjustment
        scores.append(
            {
                "base_score": base_score,
                "detector_adjustment": detector_adjustment,
                "primary_adjustment": primary_adjustment,
                "continuity_adjustment": continuity_adjustment,
                "selection_score": (
                    base_score + primary_adjustment + continuity_adjustment
                ),
            }
        )
    return scores


def select_language_candidate(
    languages: Sequence[str],
    average_log_probabilities: Sequence[float],
    detection_probabilities: Sequence[float],
    *,
    primary_language: str | None = None,
    previous_language: str | None = None,
    continuous_with_previous: bool = False,
) -> int:
    scores = score_language_candidates(
        languages,
        average_log_probabilities,
        detection_probabilities,
        primary_language=primary_language or languages[0],
        previous_language=previous_language,
        continuous_with_previous=continuous_with_previous,
    )
    return max(
        range(len(languages)),
        key=lambda index: (scores[index]["selection_score"], -index),
    )
