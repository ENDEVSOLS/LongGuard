"""Unit tests for scripts/canary_issue.py."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from scripts.canary_issue import (
    format_comment_body,
    format_issue_body,
    main,
    report_canary_failure,
)


def test_format_issue_body():
    body = format_issue_body(
        run_id="123456",
        run_url="https://github.com/ENDEVSOLS/LongGuard/actions/runs/123456",
        canary_target="latest",
        commit_sha="abcdef12",
        summary="ImportError: No module named 'crewai.hooks.dispatch'",
        timestamp="2026-11-13 03:00:00 UTC",
    )
    assert "[123456](https://github.com/ENDEVSOLS/LongGuard/actions/runs/123456)" in body
    assert "`latest`" in body
    assert "ImportError: No module named 'crewai.hooks.dispatch'" in body


def test_format_comment_body():
    body = format_comment_body(
        run_id="789012",
        run_url="https://github.com/ENDEVSOLS/LongGuard/actions/runs/789012",
        canary_target="minimum",
        commit_sha="12345678",
        summary="Test failure on lowest-direct",
        timestamp="2026-11-14 03:00:00 UTC",
    )
    assert "Recurring Canary Failure Detected" in body
    assert "[789012]" in body
    assert "`minimum`" in body
    assert "Test failure on lowest-direct" in body


def test_dry_run_mode(capsys):
    res = report_canary_failure(
        repo="ENDEVSOLS/LongGuard",
        token="mock-token",
        run_id="100",
        run_url="https://github.com/test",
        canary_target="latest",
        commit_sha="abcd",
        summary="mock failure",
        dry_run=True,
    )
    assert res["status"] == "dry_run_success"
    captured = capsys.readouterr()
    assert "[DRY-RUN]" in captured.out


def test_no_token_mode(capsys):
    res = report_canary_failure(
        repo="ENDEVSOLS/LongGuard",
        token=None,
        run_id="100",
        run_url="https://github.com/test",
        canary_target="latest",
        commit_sha="abcd",
        summary="mock failure",
        dry_run=False,
    )
    assert res["status"] == "skipped"
    captured = capsys.readouterr()
    assert "Warning: GITHUB_TOKEN not provided" in captured.err


def test_creates_new_issue_when_none_exist():
    def mock_urlopen(req, *args, **kwargs):
        mock_resp = MagicMock()
        if "issues?state=open" in req.full_url and req.get_method() == "GET":
            mock_resp.read.return_value = json.dumps([]).encode("utf-8")
        elif req.full_url.endswith("/issues") and req.get_method() == "POST":
            mock_resp.read.return_value = json.dumps({"number": 42}).encode("utf-8")
        else:
            mock_resp.read.return_value = b"{}"
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch.object(urllib.request, "urlopen", side_effect=mock_urlopen):
        res = report_canary_failure(
            repo="ENDEVSOLS/LongGuard",
            token="valid-token",
            run_id="101",
            run_url="https://github.com/ENDEVSOLS/LongGuard/actions/runs/101",
            canary_target="latest",
            commit_sha="deadbeef",
            summary="Traceback error in langgraph resume",
            api_url="https://api.github.com",
            dry_run=False,
        )
        assert res["status"] == "created"
        assert res["issue_number"] == 42


def test_updates_existing_issue_when_open_issue_exists():
    def mock_urlopen(req, *args, **kwargs):
        mock_resp = MagicMock()
        if "issues?state=open" in req.full_url and req.get_method() == "GET":
            payload = [{"number": 15, "title": "Old failure"}]
            mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        elif "/issues/15/comments" in req.full_url and req.get_method() == "POST":
            mock_resp.read.return_value = json.dumps({"id": 999}).encode("utf-8")
        else:
            mock_resp.read.return_value = b"{}"
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch.object(urllib.request, "urlopen", side_effect=mock_urlopen):
        res = report_canary_failure(
            repo="ENDEVSOLS/LongGuard",
            token="valid-token",
            run_id="102",
            run_url="https://github.com/ENDEVSOLS/LongGuard/actions/runs/102",
            canary_target="minimum",
            commit_sha="feedface",
            summary="Second failure in lowest-direct",
            api_url="https://api.github.com",
            dry_run=False,
        )
        assert res["status"] == "updated"
        assert res["issue_number"] == 15


def test_cli_execution_with_summary_file(tmp_path):
    summary_file = tmp_path / "failure.log"
    summary_file.write_text("Detailed failure log from runner", encoding="utf-8")

    with patch.object(
        sys,
        "argv",
        [
            "canary_issue.py",
            "--dry-run",
            "--summary-file",
            str(summary_file),
            "--target",
            "latest",
        ],
    ):
        main()
