from __future__ import annotations

from typing import Any


MAX_ISOLATED_SPEAKER_WORDS = 2
MAX_ISOLATED_SPEAKER_DURATION_SEC = 1.0


def _word_text(words: list[dict[str, Any]]) -> str:
    return "".join(str(word.get("word", "")) for word in words).strip()


def _speaker_groups(
    words: list[dict[str, Any]],
    fallback_speaker: str | None,
) -> list[tuple[str | None, list[dict[str, Any]]]]:
    groups: list[tuple[str | None, list[dict[str, Any]]]] = []
    for word in words:
        speaker = word.get("speaker") or fallback_speaker
        if groups and groups[-1][0] == speaker:
            groups[-1][1].append(word)
        else:
            groups.append((speaker, [word]))
    return groups


def _group_duration(words: list[dict[str, Any]]) -> float:
    starts = [float(word["start"]) for word in words if word.get("start") is not None]
    ends = [float(word["end"]) for word in words if word.get("end") is not None]
    return max(ends) - min(starts) if starts and ends else float("inf")


def _smooth_isolated_speaker_words(
    words: list[dict[str, Any]],
    fallback_speaker: str | None,
) -> list[dict[str, Any]]:
    smoothed = [dict(word) for word in words]
    groups = _speaker_groups(smoothed, fallback_speaker)
    for index in range(1, len(groups) - 1):
        previous_speaker = groups[index - 1][0]
        speaker, group_words = groups[index]
        next_speaker = groups[index + 1][0]
        if (
            previous_speaker is not None
            and previous_speaker == next_speaker
            and speaker != previous_speaker
            and len(group_words) <= MAX_ISOLATED_SPEAKER_WORDS
            and _group_duration(group_words) <= MAX_ISOLATED_SPEAKER_DURATION_SEC
        ):
            for word in group_words:
                word["speaker"] = previous_speaker
    return smoothed


def split_segments_by_word_speaker(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    split_segments: list[dict[str, Any]] = []
    for segment in segments:
        words = segment.get("words")
        if not isinstance(words, list) or not words:
            split_segments.append(segment)
            continue

        fallback_speaker = segment.get("speaker")
        smoothed_words = _smooth_isolated_speaker_words(words, fallback_speaker)
        groups = _speaker_groups(smoothed_words, fallback_speaker)

        if len(groups) == 1:
            if smoothed_words == words:
                split_segments.append(segment)
                continue
            smoothed_segment = {**segment, "words": smoothed_words}
            if groups[0][0] is not None:
                smoothed_segment["speaker"] = groups[0][0]
            split_segments.append(smoothed_segment)
            continue

        for speaker, group_words in groups:
            starts = [float(word["start"]) for word in group_words if word.get("start") is not None]
            ends = [float(word["end"]) for word in group_words if word.get("end") is not None]
            text = _word_text(group_words)
            if not text:
                continue
            child = {
                **segment,
                "start": min(starts) if starts else float(segment["start"]),
                "end": max(ends) if ends else float(segment["end"]),
                "text": text,
                "words": group_words,
            }
            if speaker is not None:
                child["speaker"] = speaker
            split_segments.append(child)
    return split_segments
