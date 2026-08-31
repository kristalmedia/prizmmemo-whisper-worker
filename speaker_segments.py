from __future__ import annotations

from typing import Any


def _word_text(words: list[dict[str, Any]]) -> str:
    return "".join(str(word.get("word", "")) for word in words).strip()


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
        groups: list[tuple[str | None, list[dict[str, Any]]]] = []
        for word in words:
            speaker = word.get("speaker") or fallback_speaker
            if groups and groups[-1][0] == speaker:
                groups[-1][1].append(word)
            else:
                groups.append((speaker, [word]))

        if len(groups) == 1:
            split_segments.append(segment)
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
