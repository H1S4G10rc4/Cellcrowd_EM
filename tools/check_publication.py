"""Check source-only Git snapshots without printing possible private contents.

Default: the staged index. --history: every commit reachable from a local ref.
The allowlist and all scanned bytes are read from the checked Git snapshot.
"""
import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {"private", "private_data", "by_image_tag", "outputs", "results",
                   "synthetic_inputs", "package", "data", "backups", ".venv", ".cache"}
TEXT_SUFFIXES = {".py", ".md", ".tex", ".txt", ".yml"}
SPECIAL_FILES = {"LICENSE", "Makefile", ".gitignore", ".gitattributes",
                 ".githooks/pre-commit", ".githooks/pre-push"}
PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"/[Hh]ome/[^/\s]+/|/[Uu]sers/[^/\s]+/|[A-Z]:\\Users\\"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def allowed_path(path):
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        return False
    if any(part.lower() in FORBIDDEN_PARTS for part in p.parts):
        return False
    return path in SPECIAL_FILES or p.name == "Makefile" or p.suffix in TEXT_SUFFIXES


def check_snapshot(entries, read):
    """Return reason codes only; do not print matched strings or unknown paths."""
    errors = []
    if "PUBLIC_FILES.txt" not in entries:
        return ["missing_public_allowlist"]
    try:
        lines = read("PUBLIC_FILES.txt").decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return ["invalid_public_allowlist"]
    allowed = {line for line in lines if line and not line.startswith("#")}
    if len(allowed) != len(lines) or any(not allowed_path(p) for p in allowed):
        errors.append("invalid_public_allowlist")
    if set(entries) != allowed:
        errors.append("file_set_differs_from_public_allowlist")
    for path, mode in entries.items():
        if not allowed_path(path):
            errors.append("data_or_unsupported_path")
        if mode not in {"100644", "100755"}:
            errors.append("link_or_unsupported_file_mode")
        data = read(path)
        if len(data) > 500_000:
            errors.append("oversized_file")
        try:
            value = data.decode("utf-8")
        except UnicodeDecodeError:
            errors.append("non_text_content")
            continue
        if any(ord(c) < 32 and c not in "\n\r\t" for c in value):
            errors.append("binary_control_characters")
        if any(pattern.search(path + "\n" + value) for pattern in PATTERNS):
            errors.append("possible_credential_or_personal_path")
    return sorted(set(errors))


def snapshot(revision=None):
    if revision is None:
        entries = {}
        for row in git("ls-files", "--stage", "-z").split(b"\0"):
            if row:
                info, path = row.split(b"\t", 1)
                mode, _, stage = info.decode().split()
                if stage != "0":
                    raise ValueError("Unmerged files are not publishable")
                entries[path.decode()] = mode
        return check_snapshot(entries, lambda p: git("show", ":" + p))
    entries = {}
    for row in git("ls-tree", "-r", "-z", revision).split(b"\0"):
        if row:
            info, path = row.split(b"\t", 1)
            entries[path.decode()] = info.decode().split()[0]
    return check_snapshot(entries, lambda p: git("show", revision + ":" + p))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args()
    revisions = git("rev-list", "--all").decode().splitlines() if args.history else [None]
    if not revisions:
        raise SystemExit("No committed history to check; omit --history to inspect staged files.")
    errors = set()
    for rev in revisions:
        errors.update(snapshot(rev))
    if errors:
        print("Publication blocked: " + ", ".join(sorted(errors)), file=sys.stderr)
        return 1
    print(f"Publication check passed: {len(revisions)} source-only snapshot(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
