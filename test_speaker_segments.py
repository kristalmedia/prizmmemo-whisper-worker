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

    def test_smooths_an_isolated_one_word_speaker_flicker(self) -> None:
        result = split_segments_by_word_speaker([{
            "start": 90.0,
            "end": 92.0,
            "text": "completion yang failed",
            "speaker": "SPEAKER_00",
            "words": [
                {"start": 90.0, "end": 90.7, "word": " completion", "speaker": "SPEAKER_00"},
                {"start": 90.7, "end": 91.0, "word": " yang", "speaker": "SPEAKER_01"},
                {"start": 91.0, "end": 91.8, "word": " failed", "speaker": "SPEAKER_00"},
            ],
        }])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[0]["text"], "completion yang failed")
        self.assertEqual(result[0]["words"][1]["speaker"], "SPEAKER_00")

    def test_keeps_a_substantive_short_speaker_turn(self) -> None:
        result = split_segments_by_word_speaker([{
            "start": 10.0,
            "end": 14.0,
            "text": "Hello. Yes, agreed. Continue.",
            "speaker": "SPEAKER_00",
            "words": [
                {"start": 10.0, "end": 10.8, "word": " Hello.", "speaker": "SPEAKER_00"},
                {"start": 11.0, "end": 11.6, "word": " Yes,", "speaker": "SPEAKER_01"},
                {"start": 11.6, "end": 12.3, "word": " agreed.", "speaker": "SPEAKER_01"},
                {"start": 12.5, "end": 13.5, "word": " Continue.", "speaker": "SPEAKER_00"},
            ],
        }])

        self.assertEqual(len(result), 3)
        self.assertEqual(result[1]["speaker"], "SPEAKER_01")
        self.assertEqual(result[1]["text"], "Yes, agreed.")


if __name__ == "__main__":
    unittest.main()
