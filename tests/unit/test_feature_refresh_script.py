"""Regression test for Windows scheduled feature refresh."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("powershell.exe"), "Windows PowerShell required")
class FeatureRefreshScriptTest(unittest.TestCase):
    def test_routine_docker_stderr_does_not_abort_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = Path(__file__).resolve().parents[2] / "scripts" / "run_feature_refresh.ps1"
            scripts = tmp_path / "scripts"
            scripts.mkdir()
            shutil.copy2(source, scripts / source.name)
            manifests = tmp_path / "manifests"
            manifests.mkdir()
            (manifests / "nflverse_sync.latest.json").write_text(
                '{"retrieved_at_utc":"2026-09-22T16:00:00Z"}', encoding="utf-8"
            )
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            (fake_bin / "docker.cmd").write_text(
                "@echo off\r\necho routine container status 1>&2\r\nexit /b 0\r\n",
                encoding="ascii",
            )
            env = os.environ.copy()
            env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]

            result = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(scripts / source.name), "-CompletedWeek", "2",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                "routine container status",
                (tmp_path / "logs" / "feature-refresh.log").read_text(encoding="utf-16"),
            )

    def test_routine_docker_stderr_does_not_abort_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = Path(__file__).resolve().parents[2] / "scripts" / "run_scheduled_slot.ps1"
            scripts = tmp_path / "scripts"
            scripts.mkdir()
            shutil.copy2(source, scripts / source.name)
            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            (fake_bin / "docker.cmd").write_text(
                '@echo off\r\necho %*>>"%FAKE_DOCKER_CALLS%"\r\n'
                "echo routine container status 1>&2\r\nexit /b 0\r\n",
                encoding="ascii",
            )
            calls = tmp_path / "docker-calls.txt"
            env = os.environ.copy()
            env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
            env["FAKE_DOCKER_CALLS"] = str(calls)

            result = subprocess.run(
                [
                    "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    str(scripts / source.name), "-Slot", "wednesday_0900",
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("pilot preflight", calls.read_text())
            self.assertIn("pilot capture", calls.read_text())
