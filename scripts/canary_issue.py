"""Manage GitHub issues for upstream compatibility canary failures.

Checks if an open issue with the 'compat-break' label already exists:
- If exists: appends a comment with the latest run details (deduplication).
- If not: creates a new issue with 'compat-break' label.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast


def query_open_compat_issues(
    repo: str, token: str, api_url: str = "https://api.github.com"
) -> list[dict[str, Any]]:
    """Query GitHub API for open issues with the 'compat-break' label."""
    url = f"{api_url}/repos/{repo}/issues?state=open&labels=compat-break"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "LongGuard-Canary-Bot",
        },
        method="GET",
    )
    with urllib.request.urlopen(req) as resp:
        return cast(list[dict[str, Any]], json.loads(resp.read().decode("utf-8")))


def create_issue(
    repo: str,
    token: str,
    title: str,
    body: str,
    labels: list[str],
    api_url: str = "https://api.github.com",
) -> dict[str, Any]:
    """Create a new GitHub issue."""
    url = f"{api_url}/repos/{repo}/issues"
    data = json.dumps({"title": title, "body": body, "labels": labels}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "LongGuard-Canary-Bot",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return cast(dict[str, Any], json.loads(resp.read().decode("utf-8")))


def add_issue_comment(
    repo: str,
    token: str,
    issue_number: int,
    comment: str,
    api_url: str = "https://api.github.com",
) -> dict[str, Any]:
    """Append a comment to an existing GitHub issue."""
    url = f"{api_url}/repos/{repo}/issues/{issue_number}/comments"
    data = json.dumps({"body": comment}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "LongGuard-Canary-Bot",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return cast(dict[str, Any], json.loads(resp.read().decode("utf-8")))


def format_issue_body(
    run_id: str,
    run_url: str,
    canary_target: str,
    commit_sha: str,
    summary: str,
    timestamp: str,
) -> str:
    """Format markdown body for a newly opened canary failure issue."""
    return f"""## ⚠️ Upstream Compatibility Canary Failure

The upstream compatibility canary detected an integration failure or package incompatibility.

- **Workflow Run:** [{run_id}]({run_url})
- **Canary Target:** `{canary_target}`
- **Triggered Commit:** `{commit_sha}`
- **Reported At:** `{timestamp}`

### Failure Summary
```
{summary}
```

### Action Items
1. Inspect the [workflow run logs]({run_url}) to identify failing dependencies.
2. Determine if upstream introduced breaking changes, dropped APIs, or updated constraints.
3. Update dependency bounds in `pyproject.toml` or adapt integration code accordingly.
4. Close this issue once upstream compatibility is verified and green.
"""


def format_comment_body(
    run_id: str,
    run_url: str,
    canary_target: str,
    commit_sha: str,
    summary: str,
    timestamp: str,
) -> str:
    """Format markdown body for an update comment on an existing issue."""
    return f"""### ⚠️ Recurring Canary Failure Detected

- **Workflow Run:** [{run_id}]({run_url})
- **Canary Target:** `{canary_target}`
- **Triggered Commit:** `{commit_sha}`
- **Reported At:** `{timestamp}`

```
{summary}
```
"""


def report_canary_failure(
    repo: str,
    token: str | None,
    run_id: str,
    run_url: str,
    canary_target: str,
    commit_sha: str,
    summary: str,
    api_url: str = "https://api.github.com",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the issue management logic: query, deduplicate, and report."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    clean_summary = summary.strip() or "No failure summary provided."

    if dry_run:
        print("[DRY-RUN] Simulating canary failure issue reporting:")
        print(f"  Repository: {repo}")
        print(f"  Target:     {canary_target}")
        print(f"  Run URL:    {run_url}")
        print(f"  Summary:    {clean_summary[:80]}...")
        return {"status": "dry_run_success", "action": "simulated"}

    if not token:
        print("Warning: GITHUB_TOKEN not provided; skipping issue creation.", file=sys.stderr)
        return {"status": "skipped", "reason": "no_token"}

    try:
        existing_issues = query_open_compat_issues(repo, token, api_url=api_url)
    except urllib.error.HTTPError as exc:
        print(f"Error querying GitHub issues: HTTP {exc.code} - {exc.reason}", file=sys.stderr)
        raise

    if existing_issues:
        target_issue = existing_issues[0]
        issue_number = target_issue["number"]
        comment_body = format_comment_body(
            run_id=run_id,
            run_url=run_url,
            canary_target=canary_target,
            commit_sha=commit_sha,
            summary=clean_summary,
            timestamp=timestamp,
        )
        add_issue_comment(repo, token, issue_number, comment_body, api_url=api_url)
        print(f"Updated existing issue #{issue_number} with recurring canary failure details.")
        return {"status": "updated", "issue_number": issue_number}
    else:
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        title = f"[Canary Alert] Upstream Framework Compatibility Breakage ({date_str})"
        body = format_issue_body(
            run_id=run_id,
            run_url=run_url,
            canary_target=canary_target,
            commit_sha=commit_sha,
            summary=clean_summary,
            timestamp=timestamp,
        )
        labels = ["compat-break", "ci"]
        new_issue = create_issue(repo, token, title, body, labels, api_url=api_url)
        issue_number = new_issue.get("number")
        print(f"Created new compat-break issue #{issue_number}: {title}")
        return {"status": "created", "issue_number": issue_number}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "ENDEVSOLS/LongGuard"),
        help="GitHub repository (owner/name)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub authorization token",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("GITHUB_RUN_ID", "local"),
        help="GitHub Actions run ID",
    )
    parser.add_argument(
        "--server-url",
        default=os.environ.get("GITHUB_SERVER_URL", "https://github.com"),
        help="GitHub server base URL",
    )
    parser.add_argument(
        "--commit-sha",
        default=os.environ.get("GITHUB_SHA", "HEAD"),
        help="Commit SHA triggering the canary run",
    )
    parser.add_argument(
        "--target",
        default="unknown",
        help="Canary target matrix job (e.g. 'latest', 'minimum')",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Short text failure summary",
    )
    parser.add_argument(
        "--summary-file",
        help="Path to a file containing full failure log / summary",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print intended actions without making actual GitHub API calls",
    )

    args = parser.parse_args()

    summary_text = args.summary
    if args.summary_file:
        summary_path = Path(args.summary_file)
        if summary_path.is_file():
            summary_text = summary_path.read_text(encoding="utf-8")

    run_url = f"{args.server_url}/{args.repo}/actions/runs/{args.run_id}"

    report_canary_failure(
        repo=args.repo,
        token=args.token,
        run_id=args.run_id,
        run_url=run_url,
        canary_target=args.target,
        commit_sha=args.commit_sha,
        summary=summary_text,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
