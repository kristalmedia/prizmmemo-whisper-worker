import unittest

from speaker_segments import split_segments_by_word_speaker


class SplitSegmentsByWordSpeakerTests(unittest.TestCase):
    def test_splits_a_segment_at_a_word_level_speaker_change(self) -> None:
        result = split_segments_by_word_speaker([{
            "start": 10.0,
            "end": 14.0,
            "text": "Hello. Selamat pagi.",
            "speaker": "SPEAKER_00",
            "words": [
                {"start": 10.0, "end": 10.8, "word": " Hello.", "speaker": "SPEAKER_00"},
                {"start": 12.0, "end": 12.8, "word": " Selamat", "speaker": "SPEAKER_01"},
                {"start": 12.8, "end": 13.6, "word": " pagi.", "speaker": "SPEAKER_01"},
            ],
        }])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[0]["text"], "Hello.")
        self.assertEqual(result[1]["speaker"], "SPEAKER_01")
        self.assertEqual(result[1]["text"], "Selamat pagi.")
        self.assertEqual(result[1]["start"], 12.0)
        self.assertEqual(result[1]["end"], 13.6)

    def test_preserves_a_segment_when_words_have_one_speaker(self) -> None:
        segment = {
            "start": 0.0,
            "end": 2.0,
            "text": "One sentence.",
            "speaker": "SPEAKER_00",
            "words": [{"start": 0.0, "end": 1.5, "word": " One sentence.", "speaker": "SPEAKER_00"}],
        }
        self.assertEqual(split_segments_by_word_speaker([segment]), [segment])

    def test_preserves_segments_without_word_timestamps(self) -> None:
        segment = {"start": 0.0, "end": 2.0, "text": "Unaligned text."}
        self.assertEqual(split_segments_by_word_speaker([segment]), [segment])


if __name__ == "__main__":
    unittest.main()
