"""Offline regressions; run with the template's and Hermes' installed dependencies."""
import asyncio
import base64
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import yaml

# Never inspect the developer's Hermes home, including during imports.
_sandbox = tempfile.TemporaryDirectory()
os.environ.update(HOME=_sandbox.name, HERMES_HOME=_sandbox.name,
                  ADMIN_PASSWORD="fabricated-test-password")
import server


class NativeConfigTest(unittest.TestCase):
    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.home = Path(home.name)
        for context in (
            patch.dict(os.environ, {"HOME": home.name, "HERMES_HOME": home.name}, clear=True),
            patch.object(server, "HERMES_HOME", home.name),
            patch.object(server, "ENV_FILE", self.home / ".env"),
            patch("socket.socket.connect", side_effect=AssertionError("Network forbidden")),
        ):
            context.start()
            self.addCleanup(context.stop)

    def config(self, model):
        (self.home / "config.yaml").write_text(yaml.safe_dump({
            "model": model, "mcp_servers": {"example": {"command": "example"}},
        }))

    def auth(self, *, pool=False):
        tokens = {"access_token": "fabricated-access", "refresh_token": "fabricated-refresh"}
        data = {"version": 2, "active_provider": "openai-codex", "providers": {}, "credential_pool": {}}
        if pool:
            data["credential_pool"]["openai-codex"] = [tokens]
        else:
            data["providers"]["openai-codex"] = {"tokens": tokens, "auth_mode": "chatgpt"}
        (self.home / "auth.json").write_text(json.dumps(data))

    def test_codex_native_auth_with_provider_default_and_pool(self):
        for pool in (False, True):
            with self.subTest(pool=pool):
                self.auth(pool=pool)
                self.config({"provider": "openai-codex", "default": ""})
                self.assertTrue(server.is_config_complete({"TELEGRAM_BOT_TOKEN": "fabricated-bot"}))
                server.write_config_yaml({})
                self.assertTrue(server.is_config_complete())
                self.assertEqual(yaml.safe_load((self.home / "config.yaml").read_text())["model"]["default"], "")
                server.write_config_yaml({}, reset_model=True)
                self.assertFalse(server.is_config_complete())

    def test_codex_cooldown_readiness_is_local_and_read_only(self):
        from hermes_cli import auth

        claims = base64.urlsafe_b64encode(json.dumps({
            "exp": 4102444800, "sub": "fabricated-account",
        }).encode()).decode().rstrip("=")
        self.auth(pool=True)
        auth_path = self.home / "auth.json"
        data = json.loads(auth_path.read_text())
        data["credential_pool"]["openai-codex"][0].update({
            "access_token": f"eyJhbGciOiJub25lIn0.{claims}.fabricated",
            "last_status": "exhausted", "last_error_code": 429,
            "last_error_reset_at": 4102444800,
        })
        auth_path.write_text(json.dumps(data))
        before = auth_path.read_bytes()
        self.config({"provider": "openai-codex", "default": ""})
        with patch.object(auth, "_probe_codex_quota_restored", wraps=auth._probe_codex_quota_restored) as probe, \
                patch("httpx.Client.send", side_effect=AssertionError("Network forbidden")) as send:
            ready = server.is_config_complete({})
            send.assert_not_called()
            probe.assert_not_called()
        self.assertTrue(ready)
        self.assertEqual(auth_path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.home.iterdir()), ["auth.json", "config.yaml"])

    def test_codex_exported_cache_is_read_only_and_rejects_expired_tokens(self):
        self.config({"provider": "openai-codex", "default": ""})
        for directory in (".codex", "custom-codex"):
            cache = self.home / directory
            cache.mkdir()
            for expiry, expected in ((4102444800, True), (1, False)):
                with self.subTest(directory=directory, expiry=expiry):
                    claims = base64.urlsafe_b64encode(json.dumps({"exp": expiry}).encode()).decode().rstrip("=")
                    auth_path = cache / "auth.json"
                    auth_path.write_text(json.dumps({"tokens": {
                        "access_token": f"eyJhbGciOiJub25lIn0.{claims}.fabricated",
                        "refresh_token": "fabricated-refresh",
                    }}))
                    before = auth_path.read_bytes()
                    with patch.dict(os.environ, {"CODEX_HOME": "" if directory == ".codex" else str(cache)}), \
                            patch("httpx.Client.send", side_effect=AssertionError("Network forbidden")) as send:
                        self.assertEqual(server.is_config_complete({}), expected)
                        send.assert_not_called()
                    self.assertEqual(auth_path.read_bytes(), before)
                    self.assertFalse((self.home / "auth.json").exists())

    def test_explicit_empty_model_blocks_native_readiness(self):
        self.auth()
        for default in ("", "native-model"):
            with self.subTest(default=default):
                self.config({"provider": "openai-codex", "default": default})
                self.assertTrue(server.is_config_complete({}))
                self.assertFalse(server.is_config_complete({"LLM_MODEL": ""}))
                server.write_config_yaml({"LLM_MODEL": ""})
                (self.home / ".env").write_text("LLM_MODEL=\n")
                self.assertFalse(server.is_config_complete())

    def test_boot_starts_native_codex_but_not_after_reset(self):
        self.auth()
        self.config({"provider": "openai-codex", "default": ""})

        async def boot():
            with patch.object(server.gw, "start", new_callable=AsyncMock) as start:
                await server.auto_start()
                await asyncio.sleep(0)
                start.assert_awaited_once()
                server.write_config_yaml({}, reset_model=True)
                start.reset_mock()
                await server.auto_start()
                await asyncio.sleep(0)
                start.assert_not_awaited()
        asyncio.run(boot())

    def test_unconfigured_and_malformed_native_auth_stay_incomplete(self):
        self.assertFalse(server.is_config_complete({}))
        self.auth()
        self.assertFalse(server.is_config_complete({}))  # Auth alone must not undo reset.
        self.config({"provider": "unconfigured-provider", "default": "native-model"})
        self.assertFalse(server.is_config_complete({}))
        self.config({"provider": "openai-codex", "default": "native-model"})
        for raw in ('{}', '[]', 'broken json', '{"providers":{"openai-codex":{"tokens":{}}}}'):
            (self.home / "auth.json").write_text(raw)
            self.assertFalse(server.is_config_complete({}))
        self.auth()
        for raw in ('[]', 'model: []', 'model: ['):
            (self.home / "config.yaml").write_text(raw)
            self.assertFalse(server.is_config_complete({}))

    def test_setup_and_native_model_with_existing_provider_key(self):
        data = {"LLM_MODEL": "setup-model", "OPENROUTER_API_KEY": "fabricated-key"}
        self.assertTrue(server.is_config_complete(data))
        self.assertFalse(server.is_config_complete({"LLM_MODEL": "setup-model"}))
        self.config({"provider": "openrouter", "default": "native-model"})
        self.assertTrue(server.is_config_complete({"OPENROUTER_API_KEY": "fabricated-key"}))
        (self.home / "auth.json").write_text(json.dumps({"providers": {
            "xai-oauth": {"tokens": {"refresh_token": "fabricated-refresh"}},
        }}))
        self.assertTrue(server.is_config_complete({"LLM_MODEL": "grok-model"}))

    def test_native_model_survives_start_and_explicit_reset(self):
        self.config({"provider": "openai-codex", "default": "native-model"})
        server.write_config_yaml({"TELEGRAM_BOT_TOKEN": "fabricated-bot"})
        config = yaml.safe_load((self.home / "config.yaml").read_text())
        self.assertEqual(config["model"], {"provider": "openai-codex", "default": "native-model"})
        server.write_config_yaml({"LLM_MODEL": "setup-model"})
        self.assertEqual(yaml.safe_load((self.home / "config.yaml").read_text())["model"]["default"], "setup-model")
        server.write_config_yaml({"LLM_MODEL": ""})
        self.assertEqual(yaml.safe_load((self.home / "config.yaml").read_text())["model"]["default"], "")
        server.write_config_yaml({}, reset_model=True)
        config = yaml.safe_load((self.home / "config.yaml").read_text())
        self.assertEqual(config["model"], {"default": ""})
        self.assertEqual(config["mcp_servers"], {"example": {"command": "example"}})


if __name__ == "__main__":
    unittest.main()
