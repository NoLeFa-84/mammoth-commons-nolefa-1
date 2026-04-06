#!/usr/bin/env python3
"""
Sum stars of a list of GitHub repositories and write a Shields.io badge.

Usage (called from the workflow):
    python sum_stars.py repo1 repo2/reother ...

The script:
* reads the repo slugs from the command line,
* fetches each repo's `stargazers_count` via the GitHub REST API,
* writes <total> to a badge SVG named `stars-total.svg` in the repo root.
"""

import os
import sys
import json
import urllib.request
from urllib.error import HTTPError


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def api_get(url: str, token: str) -> dict:
    """GET a GitHub API endpoint, return decoded JSON."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def fetch_stars(repo: str, token: str) -> int:
    """Return star count for a single repo (owner/name)."""
    try:
        data = api_get(f"https://api.github.com/repos/{repo}", token)
        return data.get("stargazers_count", 0)
    except HTTPError as e:
        print(f"⚠️  Could not fetch {repo}: {e.code} {e.reason}", file=sys.stderr)
        return 0


def make_shields_svg(label: str, message: str, color: str = "brightgreen") -> str:
    """
    Return a simple Shields.io style SVG.
    This is the same format that `https://img.shields.io/badge/…` returns,
    but we embed it locally so the badge stays static.
    """
    import base64, hashlib

    # Use Shields.io API to generate the image – easiest & always up‑to‑date.
    url = f"https://img.shields.io/badge/{urllib.parse.quote(label)}-{urllib.parse.quote(message)}-{color}.svg"
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


# -------------------------------------------------
# Main
# -------------------------------------------------
if __name__ == "__main__":
    # The GITHUB_TOKEN secret is automatically available to the workflow.
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("❌  GITHUB_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)

    repos = sys.argv[1:]  # e.g. ["owner1/repoA", "owner2/repoB"]
    if not repos:
        print("❌  No repositories supplied", file=sys.stderr)
        sys.exit(1)

    total = sum(fetch_stars(r, token) for r in repos)
    print(f"✨ Total stars: {total}")

    # Create the badge SVG (label = "stars", message = total)
    badge_svg = make_shields_svg("stars", str(total))
    badge_path = os.path.join(
        os.getenv("GITHUB_WORKSPACE", "."), "stars-mammoth-badge.svg"
    )
    with open(badge_path, "w", encoding="utf-8") as f:
        f.write(badge_svg)

    # Let the workflow know the value (optional, for logs)
    print(f"::set-output name=total::{total}")
