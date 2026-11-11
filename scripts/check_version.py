"""Verify version consistency across project metadata, source, lockfile, and built artifacts."""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path


def get_project_version(root: Path) -> str:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["version"]


def get_init_version(root: Path) -> str:
    init_content = (root / "src" / "longguard" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_content, re.MULTILINE)
    if not match:
        raise ValueError("Could not find __version__ in src/longguard/__init__.py")
    return match.group(1)


def get_lock_version(root: Path) -> str:
    lock_file = root / "uv.lock"
    if not lock_file.exists():
        raise FileNotFoundError("uv.lock not found")
    lock_data = tomllib.loads(lock_file.read_text(encoding="utf-8"))
    for pkg in lock_data.get("package", []):
        if pkg.get("name") == "longguard":
            return pkg.get("version", "")
    raise ValueError("longguard package not found in uv.lock")


def check_version_sync(root: Path, tag: str | None = None, require_changelog: bool = False) -> str:
    pyproject_ver = get_project_version(root)
    init_ver = get_init_version(root)
    lock_ver = get_lock_version(root)

    errors = []
    if pyproject_ver != init_ver:
        errors.append(
            f"Version mismatch: pyproject.toml ({pyproject_ver}) "
            f"!= src/longguard/__init__.py ({init_ver})"
        )
    if pyproject_ver != lock_ver:
        errors.append(
            f"Version mismatch: pyproject.toml ({pyproject_ver}) != uv.lock ({lock_ver})"
        )

    if tag:
        expected_tag = f"v{pyproject_ver}"
        if tag != expected_tag:
            errors.append(
                f"Tag mismatch: specified tag '{tag}' != expected tag '{expected_tag}'"
            )

    is_stable = not bool(re.search(r"\.dev\d+", pyproject_ver))
    if require_changelog or is_stable:
        changelog_path = root / "CHANGELOG.md"
        if not changelog_path.exists():
            errors.append(f"CHANGELOG.md not found at {changelog_path}")
        else:
            changelog = changelog_path.read_text(encoding="utf-8")
            heading = f"## [{pyproject_ver}]"
            if heading not in changelog:
                errors.append(
                    f"Missing changelog entry '{heading}' in CHANGELOG.md "
                    f"for version {pyproject_ver}"
                )

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        raise ValueError(f"{len(errors)} version consistency check(s) failed: {'; '.join(errors)}")

    return pyproject_ver


def check_distribution_artifacts(root: Path, dist_dir: Path, expected_version: str) -> None:
    if not dist_dir.exists():
        raise FileNotFoundError(f"Distribution directory {dist_dir} does not exist.")

    wheels = list(dist_dir.glob(f"longguard-{expected_version}*.whl"))
    sdists = list(dist_dir.glob(f"longguard-{expected_version}*.tar.gz"))

    if not wheels:
        raise ValueError(
            f"No .whl packages matching version {expected_version} found in {dist_dir}"
        )
    if not sdists:
        raise ValueError(
            f"No .tar.gz packages matching version {expected_version} found in {dist_dir}"
        )

    all_wheels = list(dist_dir.glob("*.whl"))
    if len(all_wheels) > len(wheels):
        print(
            f"WARNING: Found {len(all_wheels) - len(wheels)} other wheel(s) in {dist_dir}",
            file=sys.stderr,
        )

    errors = []

    # Check wheel packages
    for wheel in wheels:
        with zipfile.ZipFile(wheel, "r") as zf:
            namelist = set(zf.namelist())
            if "longguard/py.typed" not in namelist:
                errors.append(f"{wheel.name}: missing PEP 561 marker 'longguard/py.typed'")
            if "longguard/__init__.py" not in namelist:
                errors.append(f"{wheel.name}: missing 'longguard/__init__.py'")

            forbidden_prefixes = ("tests/", "site/", "scripts/", "benchmarks/")
            for name in namelist:
                if any(name.startswith(prefix) for prefix in forbidden_prefixes):
                    errors.append(f"{wheel.name}: contains forbidden file '{name}'")

            # Check wheel metadata version
            metadata_name = f"longguard-{expected_version}.dist-info/METADATA"
            if metadata_name not in namelist:
                errors.append(f"{wheel.name}: missing metadata file '{metadata_name}'")
            else:
                metadata_content = zf.read(metadata_name).decode("utf-8", errors="replace")
                pattern = rf"^Version:\s*{re.escape(expected_version)}$"
                if not re.search(pattern, metadata_content, re.MULTILINE):
                    errors.append(
                        f"{wheel.name}: METADATA does not state Version: {expected_version}"
                    )

    # Check sdist packages
    for sdist in sdists:
        with tarfile.open(sdist, "r:gz") as tf:
            names = [m.name for m in tf.getmembers()]
            forbidden_substrings = (
                "/site/",
                "release-evidence.json",
                "release-coverage.json",
                ".pytest_cache",
            )
            for name in names:
                if any(sub in name for sub in forbidden_substrings):
                    errors.append(f"{sdist.name}: contains forbidden entry '{name}'")

            pkg_info = next((name for name in names if name.endswith("/PKG-INFO")), None)
            if not pkg_info:
                errors.append(f"{sdist.name}: missing PKG-INFO")
            else:
                extracted = tf.extractfile(pkg_info)
                if extracted:
                    content = extracted.read().decode("utf-8", errors="replace")
                    pattern = rf"^Version:\s*{re.escape(expected_version)}$"
                    if not re.search(pattern, content, re.MULTILINE):
                        errors.append(
                            f"{sdist.name}: PKG-INFO does not state Version: {expected_version}"
                        )

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        raise ValueError(
            f"{len(errors)} distribution artifact check(s) failed: {'; '.join(errors)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root",
    )
    parser.add_argument("--tag", type=str, help="Tag to verify (e.g., v0.2.2)")
    parser.add_argument(
        "--require-changelog", action="store_true", help="Always require changelog entry"
    )
    parser.add_argument(
        "--check-dist", action="store_true", help="Verify built artifacts in dist/"
    )
    parser.add_argument("--dist-dir", type=Path, default=None, help="Custom dist directory")
    args = parser.parse_args()

    root = args.root.resolve()
    try:
        version = check_version_sync(root, tag=args.tag, require_changelog=args.require_changelog)
        print(f"Version consistency check passed: {version}")

        if args.check_dist:
            dist_dir = (args.dist_dir or (root / "dist")).resolve()
            check_distribution_artifacts(root, dist_dir, version)
            print(f"Distribution artifact check passed in {dist_dir}")

    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
