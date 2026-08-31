import unittest

from language_selection import score_language_candidates, select_language_candidate


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

    def test_primary_language_wins_an_ambiguous_candidate_pair(self) -> None:
        selected = select_language_candidate(
            ["en", "ms"],
            [-0.35, -0.26],
            [0.48, 0.52],
            primary_language="en",
        )
        self.assertEqual(selected, 0)

    def test_clear_secondary_language_evidence_beats_primary_preference(self) -> None:
        selected = select_language_candidate(
            ["en", "ms"],
            [-0.70, -0.24],
            [0.25, 0.75],
            primary_language="en",
        )
        self.assertEqual(selected, 1)

    def test_english_asr_wins_first_wrong_translation_regression(self) -> None:
        selected = select_language_candidate(
            ["en", "ms"],
            [-0.1185, -0.4747],
            [0.0012, 0.9878],
            primary_language="en",
            previous_language="ms",
            continuous_with_previous=True,
        )
        self.assertEqual(selected, 0)

    def test_english_asr_wins_second_wrong_translation_regression(self) -> None:
        selected = select_language_candidate(
            ["en", "ms"],
            [-0.1311, -0.5511],
            [0.0008, 0.9980],
            primary_language="en",
            previous_language="en",
            continuous_with_previous=True,
        )
        self.assertEqual(selected, 0)

    def test_detector_adjustment_is_bounded(self) -> None:
        scores = score_language_candidates(
            ["en", "ms"],
            [-0.30, -0.30],
            [0.0, 1.0],
            primary_language="en",
        )
        self.assertEqual(scores[0]["detector_adjustment"], -0.1)
        self.assertEqual(scores[1]["detector_adjustment"], 0.1)

    def test_continuity_breaks_a_close_score_only_for_adjacent_speech(self) -> None:
        continuous = select_language_candidate(
            ["en", "ms"],
            [-0.30, -0.24],
            [0.50, 0.50],
            primary_language="en",
            previous_language="ms",
            continuous_with_previous=True,
        )
        after_pause = select_language_candidate(
            ["en", "ms"],
            [-0.30, -0.24],
            [0.50, 0.50],
            primary_language="en",
            previous_language="ms",
            continuous_with_previous=False,
        )
        self.assertEqual(continuous, 1)
        self.assertEqual(after_pause, 0)

    def test_score_diagnostics_explain_primary_and_continuity_adjustments(self) -> None:
        scores = score_language_candidates(
            ["en", "ms"],
            [-0.30, -0.30],
            [0.50, 0.50],
            primary_language="en",
            previous_language="en",
            continuous_with_previous=True,
        )
        self.assertGreater(scores[0]["primary_adjustment"], 0)
        self.assertGreater(scores[0]["continuity_adjustment"], 0)
        self.assertEqual(scores[1]["primary_adjustment"], 0)
        self.assertEqual(scores[1]["continuity_adjustment"], 0)

    def test_rejects_mismatched_candidate_inputs(self) -> None:
        with self.assertRaises(ValueError):
            select_language_candidate(["en", "ms"], [-0.25], [0.50, 0.50])

    def test_rejects_unknown_primary_language(self) -> None:
        with self.assertRaises(ValueError):
            select_language_candidate(
                ["en", "ms"],
                [-0.25, -0.25],
                [0.50, 0.50],
                primary_language="fr",
            )


if __name__ == "__main__":
    unittest.main()
