# Copyright 2026 Divyam.ai
# SPDX-License-Identifier: Apache-2.0

"""Decide the version a release publishes, and refuse one that cannot be.

The version is the one requested. When none is, it is the project's version
on main if that is not yet published (a bump made on main by hand, or one an
earlier run committed and tagged but failed to publish), and otherwise the
next patch after the latest published. Either way it must be new to the index
and higher than everything already there: PyPI never accepts a version twice,
and a release below the latest would never be installed by default.

Usage: release_version.py --package NAME --index URL --current VERSION
       [--requested VERSION]
Prints ``version=<v>`` and ``normalized=<v>`` lines for GITHUB_OUTPUT.
"""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request

from packaging.version import InvalidVersion, Version

_SEMVER = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$")


def _published(index: str, package: str) -> list[Version]:
    url = f"{index.rstrip('/')}/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            releases = json.load(response).get("releases", {})
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return []
        raise
    versions: list[Version] = []
    for value in releases:
        try:
            versions.append(Version(value))
        except InvalidVersion:
            continue
    return versions


def _fail(message: str) -> None:
    print(f"::error::{message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--current", required=True)
    parser.add_argument("--requested", default="")
    args = parser.parse_args()

    published = _published(args.index, args.package)
    latest = max(published, default=None)
    if args.requested:
        if not _SEMVER.match(args.requested):
            _fail(f"Invalid version {args.requested!r}: expected X.Y.Z or X.Y.Z-pre")
        version = args.requested
    elif Version(args.current) not in published and (
        latest is None or Version(args.current) > latest
    ):
        version = args.current
    else:
        base = max(Version(args.current), latest or Version("0"))
        version = f"{base.major}.{base.minor}.{base.micro + 1}"
    candidate = Version(version)

    if candidate in published:
        _fail(f"{args.package} {candidate} is already published on {args.index}")
    if latest is not None and candidate <= latest:
        _fail(f"{candidate} is not higher than {latest}, the latest on {args.index}")
    print(
        f"Latest on {args.index}: {latest or 'none'}; releasing {version}",
        file=sys.stderr,
    )
    print(f"version={version}")
    print(f"normalized={candidate}")


if __name__ == "__main__":
    main()
