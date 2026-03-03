# test_ollama_router.py
# Lightweight tests for the Ollama routing logic.
# Run with:  python -m pytest lucidiaAI/LUCIDIA_CORE/test_ollama_router.py -v
# or:        python lucidiaAI/LUCIDIA_CORE/test_ollama_router.py

import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock

# Allow running directly from the repo root or this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ollama_router as router


class TestOllamaTriggerDetection(unittest.TestCase):
    """Verify that @mention detection works correctly."""

    def test_copilot_trigger(self):
        self.assertTrue(router.contains_ollama_trigger("@copilot explain this"))

    def test_lucidia_trigger(self):
        self.assertTrue(router.contains_ollama_trigger("Hey @lucidia, what is 2+2?"))

    def test_blackboxprogramming_trigger(self):
        self.assertTrue(router.contains_ollama_trigger("@blackboxprogramming run a check"))

    def test_case_insensitive(self):
        self.assertTrue(router.contains_ollama_trigger("@Copilot do this"))
        self.assertTrue(router.contains_ollama_trigger("@LUCIDIA answer me"))

    def test_no_trigger(self):
        self.assertFalse(router.contains_ollama_trigger("just a normal message"))

    def test_other_mention_not_a_trigger(self):
        self.assertFalse(router.contains_ollama_trigger("@someone else"))


class TestStripTriggers(unittest.TestCase):
    def test_strips_copilot(self):
        result = router._strip_triggers("@copilot explain this")
        self.assertNotIn("@copilot", result.lower())
        self.assertIn("explain this", result)

    def test_strips_multiple(self):
        result = router._strip_triggers("@lucidia @copilot hello")
        self.assertNotIn("@lucidia", result.lower())
        self.assertNotIn("@copilot", result.lower())


class TestRoute(unittest.TestCase):
    """route() must call Ollama for trigger messages and return None otherwise."""

    def test_returns_none_for_non_trigger(self):
        result = router.route("no mention here")
        self.assertIsNone(result)

    def _make_mock_response(self, text: str):
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"response": text}).encode()
        return mock_resp

    def test_routes_copilot_to_ollama(self):
        mock_resp = self._make_mock_response("42")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = router.route("@copilot what is 6*7?")
        self.assertEqual(result, "42")

    def test_routes_lucidia_to_ollama(self):
        mock_resp = self._make_mock_response("hello")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = router.route("@lucidia say hello")
        self.assertEqual(result, "hello")

    def test_routes_blackboxprogramming_to_ollama(self):
        mock_resp = self._make_mock_response("ok")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = router.route("@blackboxprogramming run task")
        self.assertEqual(result, "ok")

    def test_no_external_fallback_on_connection_error(self):
        """If Ollama is down the router must raise ConnectionError, not silently
        fall back to a cloud provider."""
        import urllib.error
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("refused"),
        ):
            with self.assertRaises(ConnectionError):
                router.route("@copilot do something")


if __name__ == "__main__":
    unittest.main(verbosity=2)
