import unittest

import numpy as np

from output_payload import compact_segments, normalize_speaker_embeddings


class OutputPayloadTests(unittest.TestCase):
    def test_compacts_segments_to_the_application_contract(self):
        output = compact_segments([{
            "start": np.float32(1.25),
            "end": np.float64(2.5),
            "speaker": "SPEAKER_00",
            "text": " Hello",
            "words": [{"word": "Hello", "score": np.float32(0.9)}],
            "chars": [{"char": "H"}],
        }])

        self.assertEqual(output, [{
            "start": 1.25,
            "end": 2.5,
            "speaker": "SPEAKER_00",
            "text": " Hello",
        }])

    def test_normalizes_numpy_embedding_vectors(self):
        output = normalize_speaker_embeddings({
            "SPEAKER_00": np.array([0.25, -0.5], dtype=np.float32),
        })

        self.assertEqual(output, {"SPEAKER_00": [0.25, -0.5]})

    def test_rejects_non_finite_embedding_values(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            normalize_speaker_embeddings({"SPEAKER_00": [float("nan")]})


if __name__ == "__main__":
    unittest.main()

