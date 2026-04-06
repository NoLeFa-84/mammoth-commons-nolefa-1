#!/usr/bin/env python3
import os
import sys
import json
import urllib.request
from urllib.error import HTTPError


def api_get(url: str, token: str) -> dict:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def fetch_stars(repo: str, token: str) -> int:
    try:
        data = api_get(f"https://api.github.com/repos/{repo}", token)
        ret = data.get("stargazers_count", 0)
        print(f"⭐ {repo} stars: {ret}")
        return ret
    except HTTPError as e:
        print(f"⚠️ {repo}  skipped: {e.code} {e.reason}", file=sys.stderr)
        return 0


def make_shields_svg(label: str, message: str, color: str = "brightgreen") -> str:
    url = f"https://img.shields.io/badge/{urllib.parse.quote(label)}-{urllib.parse.quote(message)}-{color}.svg"
    with urllib.request.urlopen(url) as resp:
        return resp.read().decode("utf-8")


def local_svg_template(label: str, message: str, color: str = "#ffa") -> str:
    label_w = max(40, len(label) * 6 + 10)
    message_w = max(40, len(message) * 6 + 10)
    total_w = label_w + message_w

    bg_label = "#555"
    bg_message = color

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="20">
  <linearGradient id="b" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <mask id="a"><rect width="{total_w}" height="20" rx="3" fill="#fff"/></mask>
  <g mask="url(#a)">
    <rect width="{label_w}" height="20" fill="{bg_label}"/>
    <rect x="{label_w}" width="{message_w}" height="20" fill="{bg_message}"/>
    <rect width="{total_w}" height="20" fill="url(#b)"/>
  </g>
  <g fill="#fff" text-anchor="middle"
     font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{label_w/2}" y="15" fill="#010101" fill-opacity=".3">{label}</text>
    <text x="{label_w/2}" y="14">{label}</text>
    <text x="{label_w + message_w/2}" y="15" fill="#010101" fill-opacity=".3">{message}</text>
    <text x="{label_w + message_w/2}" y="14">{message}</text>
  </g>
</svg>"""


def write_badge(svg: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


if __name__ == "__main__":
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("❌  GITHUB_TOKEN env var not set", file=sys.stderr)
        sys.exit(1)
    repos = sys.argv[1:]
    if not repos:
        print("❌  No repositories supplied", file=sys.stderr)
        sys.exit(1)
    total = sum(fetch_stars(r, token) for r in repos)
    print(f"⭐ Total stars: {total}")

    badge_svg = local_svg_template("⭐ MAI-BIAS star total", str(total))
    workspace = os.getenv("GITHUB_WORKSPACE", ".")
    out_file = os.path.join(workspace, "stars-mammoth-badge.svg")
    write_badge(badge_svg, out_file)
    print(f"📝  Badge written to {out_file}")

    with open(os.getenv("GITHUB_OUTPUT", "/dev/null"), "a") as out:
        out.write(f"total={total}\n")
