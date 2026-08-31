import logging
import unittest

from log_safety import suppress_signed_request_logging


class SafeHttpLoggingTests(unittest.TestCase):
    def test_http_request_loggers_do_not_emit_info_urls(self) -> None:
        logging.getLogger("httpx").setLevel(logging.INFO)
        logging.getLogger("httpcore").setLevel(logging.INFO)
        suppress_signed_request_logging()
        self.assertEqual(logging.getLogger("httpx").level, logging.WARNING)
        self.assertEqual(logging.getLogger("httpcore").level, logging.WARNING)


if __name__ == "__main__":
    unittest.main()
