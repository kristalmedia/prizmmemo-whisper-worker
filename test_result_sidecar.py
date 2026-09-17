import unittest
from sys import modules
from unittest.mock import MagicMock, patch

from result_sidecar import put_result_sidecar, validate_result_put_url


class ResultSidecarTests(unittest.TestCase):
    def test_url_rejects_other_hosts_and_credentials(self):
        suffix = ".r2.cloudflarestorage.com"
        self.assertIsNone(validate_result_put_url(None, suffix))
        self.assertEqual(validate_result_put_url("https://abc.r2.cloudflarestorage.com/bucket/result?sig=1", suffix),
                         "https://abc.r2.cloudflarestorage.com/bucket/result?sig=1")
        with self.assertRaises(ValueError):
            validate_result_put_url("https://abc.r2.cloudflarestorage.com.evil.test/result", suffix)
        with self.assertRaises(ValueError):
            validate_result_put_url("https://user:pass@abc.r2.cloudflarestorage.com/result", suffix)

    def test_writes_job_bound_output_without_logging_it(self):
        client = MagicMock()
        client.put.return_value.status_code = 200
        httpx = MagicMock()
        httpx.Client.return_value.__enter__.return_value = client
        with patch.dict(modules, {"httpx": httpx}):
            self.assertTrue(put_result_sidecar("https://abc.r2.cloudflarestorage.com/result", "job-1", {"meeting_id": "m-1"}))
        body = client.put.call_args.kwargs["content"]
        self.assertIn(b'"job_id":"job-1"', body)
        self.assertIn(b'"meeting_id":"m-1"', body)

    def test_refuses_missing_job_id(self):
        self.assertFalse(put_result_sidecar("https://abc.r2.cloudflarestorage.com/result", "", {}))


if __name__ == "__main__":
    unittest.main()
