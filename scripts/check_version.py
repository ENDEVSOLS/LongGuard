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


def compare_wheels(candidate_wheel: Path, target_wheel: Path) -> None:
    """Compare candidate and target wheels to verify zero runtime behavior divergence.

    The wheels must contain identical files and identical contents for all runtime
    modules. Only version identifiers (__version__, METADATA, RECORD) are permitted
    to differ.
    """
    if not candidate_wheel.exists():
        raise FileNotFoundError(f"Candidate wheel does not exist: {candidate_wheel}")
    if not target_wheel.exists():
        raise FileNotFoundError(f"Target wheel does not exist: {target_wheel}")

    errors = []

    def normalize_name(name: str) -> str:
        parts = name.split("/", 1)
        if (
            len(parts) == 2
            and parts[0].startswith("longguard-")
            and parts[0].endswith(".dist-info")
        ):
            return f"[DIST-INFO]/{parts[1]}"
        return name

    with (
        zipfile.ZipFile(candidate_wheel, "r") as zf_cand,
        zipfile.ZipFile(target_wheel, "r") as zf_targ,
    ):
        cand_map = {normalize_name(name): name for name in zf_cand.namelist()}
        targ_map = {normalize_name(name): name for name in zf_targ.namelist()}

        cand_norm_set = set(cand_map.keys())
        targ_norm_set = set(targ_map.keys())

        missing_in_target = cand_norm_set - targ_norm_set
        extra_in_target = targ_norm_set - cand_norm_set

        if missing_in_target:
            errors.append(
                f"Target wheel missing files present in candidate: {sorted(missing_in_target)}"
            )
        if extra_in_target:
            errors.append(
                f"Target wheel contains unexpected extra files: {sorted(extra_in_target)}"
            )

        for norm_name in sorted(cand_norm_set & targ_norm_set):
            cand_raw = zf_cand.read(cand_map[norm_name])
            targ_raw = zf_targ.read(targ_map[norm_name])

            if norm_name == "longguard/__init__.py":
                cand_text = re.sub(
                    r'^__version__\s*=\s*["\'].*["\']',
                    "",
                    cand_raw.decode("utf-8"),
                    flags=re.MULTILINE,
                )
                targ_text = re.sub(
                    r'^__version__\s*=\s*["\'].*["\']',
                    "",
                    targ_raw.decode("utf-8"),
                    flags=re.MULTILINE,
                )
                if cand_text != targ_text:
                    errors.append(
                        "Runtime divergence in longguard/__init__.py beyond __version__ declaration"
                    )
            elif norm_name.startswith("[DIST-INFO]/"):
                dist_sub = norm_name[len("[DIST-INFO]/") :]
                if dist_sub in ("METADATA", "RECORD"):
                    continue
                if cand_raw != targ_raw:
                    errors.append(f"Metadata file {dist_sub} differs between candidate and target")
            else:
                if cand_raw != targ_raw:
                    errors.append(
                        f"Runtime module {norm_name} differs between candidate and target"
                    )

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        raise ValueError(
            f"{len(errors)} wheel comparison check(s) failed: {'; '.join(errors)}"
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
    parser.add_argument(
        "--compare-wheel",
        type=Path,
        default=None,
        help="Path to candidate wheel to verify zero divergence against target wheel in dist/",
    )
    parser.add_argument(
        "--target-wheel",
        type=Path,
        default=None,
        help="Path to target wheel to compare (defaults to built wheel in dist/)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    try:
        version = check_version_sync(root, tag=args.tag, require_changelog=args.require_changelog)
        print(f"Version consistency check passed: {version}")

        if args.check_dist:
            dist_dir = (args.dist_dir or (root / "dist")).resolve()
            check_distribution_artifacts(root, dist_dir, version)
            print(f"Distribution artifact check passed in {dist_dir}")

        if args.compare_wheel:
            target_wheel = args.target_wheel
            if not target_wheel:
                dist_dir = (args.dist_dir or (root / "dist")).resolve()
                wheels = list(dist_dir.glob(f"longguard-{version}*.whl"))
                if not wheels:
                    raise ValueError(
                        f"No built wheel matching version {version} found in {dist_dir} to compare"
                    )
                target_wheel = wheels[0]
            compare_wheels(args.compare_wheel.resolve(), target_wheel.resolve())
            print(
                f"Wheel diff verification passed: {args.compare_wheel.name} == {target_wheel.name}"
            )

    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
