import unittest

from language_selection import select_language_candidate


class SelectLanguageCandidateTests(unittest.TestCase):
    def test_asr_likelihood_outweighs_a_weak_detector_preference(self) -> None:
        selected = select_language_candidate(
            ["en", "ms"],
            [-0.18, -0.72],
            [0.35, 0.65],
        )
        self.assertEqual(selected, 0)

    def test_detection_probability_breaks_close_asr_scores(self) -> None:
        selected = select_language_candidate(
            ["en", "ms"],
            [-0.31, -0.30],
            [0.85, 0.15],
        )
        self.assertEqual(selected, 0)

    def test_configured_order_breaks_an_exact_tie(self) -> None:
        selected = select_language_candidate(
            ["en", "ms"],
            [-0.25, -0.25],
            [0.50, 0.50],
        )
        self.assertEqual(selected, 0)

    def test_rejects_mismatched_candidate_inputs(self) -> None:
        with self.assertRaises(ValueError):
            select_language_candidate(["en", "ms"], [-0.25], [0.50, 0.50])


if __name__ == "__main__":
    unittest.main()
