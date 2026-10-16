"""A17: pristine export of UB-GOLD a96137b + the minimal `# CHANGE` edits needed to import/run it (NOTES.md §2).

  python3 -I -B make_src.py            # creates $GLAD_DATA_ROOT/ubgold_versions/src_a96137b (refuses if present)
  python3 -I -B make_src.py --verify   # re-checks an existing tree against PATCHES.json (no writes)

Steps: `git archive a96137b | tar -x` into a fresh directory; COMMIT file; each patch asserts the exact pristine line
(CRLF/LF kept, line count kept); PATCHES.json (in the tree and next to this script) records file, line, pristine and
patched text, pristine/patched sha256 per patched file, and a digest of the whole tree; finally the tree is made
read-only. Pure standard library.
"""
import argparse
import hashlib
import io
import json
import os
import os.path as osp
import stat
import subprocess
import sys
import tarfile
import time

REPO = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_versions/UB-GOLD_full")
COMMIT = "a96137b0f67dbb1f97cb1867c77849dd1bf5032f"
DEST = (os.environ.get("GLAD_DATA_ROOT", "glad_data") + "/ubgold_versions/src_a96137b")
HERE = osp.dirname(osp.abspath(__file__))
TAG = "# CHANGE (A17 a96137b): "

# (file, 1-based line, exact pristine content without line ending, replacement without line ending)
PATCHES = [
    ("GAOOD/detector/mybase.py", 34, "       ",
     "        pass  " + TAG + "empty abstract body is a SyntaxError in Python 3 (PR #3 f0c6a69 added the same)"),
    ("GAOOD/detector/mybase.py", 38, "      ",
     "        pass  " + TAG + "empty abstract body is a SyntaxError in Python 3 (PR #3 f0c6a69 added the same)"),
    ("GAOOD/detector/mybase.py", 42, "       ",
     "        pass  " + TAG + "empty abstract body is a SyntaxError in Python 3 (PR #3 f0c6a69 added the same)"),
    ("GAOOD/detector/mybase.py", 343, "       ",
     "        pass  " + TAG + "empty abstract body is a SyntaxError in Python 3 (PR #3 f0c6a69 added the same)"),
    ("GAOOD/detector/mybase.py", 347, "       ",
     "        pass  " + TAG + "empty abstract body is a SyntaxError in Python 3 (PR #3 f0c6a69 added the same)"),
    ("dataloader/data_loader.py", 423, " 1",
     "    " + TAG + "stray ' 1' line removed: IndentationError in Python 3 (no-op expression; PR #3 deleted it)"),
]
EXPECT_DEF = {("GAOOD/detector/mybase.py", 34): "def process_graph(self, data):",
              ("GAOOD/detector/mybase.py", 38): "def fit(self, data,label=None):",
              ("GAOOD/detector/mybase.py", 42): "def decision_function(self, data, label=None):",
              ("GAOOD/detector/mybase.py", 343): "def init_model(self, **kwargs):",
              ("GAOOD/detector/mybase.py", 347): "def forward_model(self, data):"}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    with open(p, "rb") as f:
        return sha256_bytes(f.read())


def tree_digest(root):
    """sha256 over (relative path, file sha256) of every regular file, sorted; COMMIT/PATCHES.json excluded."""
    h = hashlib.sha256()
    n = 0
    for dp, dn, fn in os.walk(root):
        dn.sort()
        for f in sorted(fn):
            p = osp.join(dp, f)
            rel = osp.relpath(p, root)
            if rel in ("COMMIT", "PATCHES.json"):
                continue
            h.update(rel.encode() + b"\0" + sha256_file(p).encode() + b"\n")
            n += 1
    return h.hexdigest(), n


def git(*a, binary=False):
    out = subprocess.run(["git", "-C", REPO] + list(a), capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode()


def split_lines(b):
    return b.split(b"\n")   # keeps a trailing '\r' on CRLF lines


def apply(dest):
    by_file = {}
    for f, ln, old, new in PATCHES:
        by_file.setdefault(f, []).append((ln, old, new))
    recs = []
    for f, items in by_file.items():
        p = osp.join(dest, f)
        raw = open(p, "rb").read()
        blob = git("show", f"{COMMIT}:{f}", binary=True)
        assert raw == blob, f"{f}: export differs from git blob"
        lines = split_lines(raw)
        n0 = len(lines)
        for ln, old, new in items:
            cur = lines[ln - 1]
            eol = b"\r" if cur.endswith(b"\r") else b""
            body = cur[:-1] if eol else cur
            assert body.decode() == old, (f, ln, body, old)
            if (f, ln) in EXPECT_DEF:
                prev = lines[ln - 2].rstrip(b"\r").decode().strip()
                assert prev == EXPECT_DEF[(f, ln)], (f, ln, prev)
            lines[ln - 1] = new.encode() + eol
        new_raw = b"\n".join(lines)
        assert len(split_lines(new_raw)) == n0
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        with open(p, "wb") as fh:
            fh.write(new_raw)
        import ast
        ast.parse(new_raw.decode())          # the patched file must parse now
        recs.append(dict(file=f, lines=[dict(line=ln, pristine=old, patched=new) for ln, old, new in items],
                         sha256_pristine=sha256_bytes(raw), sha256_patched=sha256_bytes(new_raw),
                         git_blob=git("rev-parse", f"{COMMIT}:{f}").strip()))
    return recs


def make_readonly(root):
    for dp, dn, fn in os.walk(root):
        for f in fn:
            os.chmod(osp.join(dp, f), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for dp, dn, fn in os.walk(root, topdown=False):
        os.chmod(dp, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)


def verify(dest):
    P = json.load(open(osp.join(dest, "PATCHES.json")))
    ok = True
    for r in P["patched_files"]:
        s = sha256_file(osp.join(dest, r["file"]))
        print(f"{r['file']}: patched sha256 {'OK' if s == r['sha256_patched'] else 'MISMATCH ' + s}")
        ok &= s == r["sha256_patched"]
    d, n = tree_digest(dest)
    print(f"tree digest ({n} files): {'OK' if d == P['tree_sha256_patched'] else 'MISMATCH ' + d}")
    ok &= d == P["tree_sha256_patched"]
    # every unpatched file equals its git blob
    patched = {r["file"] for r in P["patched_files"]}
    names = git("ls-tree", "-r", "--name-only", COMMIT).split("\n")
    bad = [f for f in names if f and f not in patched
           and sha256_file(osp.join(dest, f)) != sha256_bytes(git("show", f"{COMMIT}:{f}", binary=True))]
    print(f"unpatched files equal to git blobs: {'OK' if not bad else bad}")
    ok &= not bad
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dest", default=DEST)
    a = ap.parse_args()
    if a.verify:
        sys.exit(0 if verify(a.dest) else 1)
    assert git("rev-parse", "a96137b^{commit}").strip() == COMMIT
    if osp.exists(a.dest):
        sys.exit(f"{a.dest} exists; refusing to overwrite (use --verify)")
    os.makedirs(a.dest)
    arc = git("archive", "--format=tar", COMMIT, binary=True)
    with tarfile.open(fileobj=io.BytesIO(arc)) as t:
        t.extractall(a.dest, filter="tar")
    names = [f for f in git("ls-tree", "-r", "--name-only", COMMIT).split("\n") if f]
    d0, n0 = tree_digest(a.dest)
    assert n0 == len(names), (n0, len(names))
    recs = apply(a.dest)
    d1, n1 = tree_digest(a.dest)
    meta = dict(commit=COMMIT, commit_date=git("show", "-s", "--format=%ci", COMMIT).strip(), source_repo=REPO,
                remote=git("remote", "get-url", "origin").strip(), created=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
                created_by="prelim/evalpaper/version_a96137b/make_src.py",
                make_src_sha256=sha256_file(osp.abspath(__file__)), n_files=n0,
                tree_sha256_pristine=d0, tree_sha256_patched=d1, patched_files=recs,
                not_patched_note="data_loader.py l.19 'from GOOD import register' is never executed (the harness "
                                 "extracts single functions with ast); every other file is byte-identical to git.")
    with open(osp.join(a.dest, "COMMIT"), "w") as f:
        f.write(f"{COMMIT}\n{meta['commit_date']}\nexported {meta['created']} with git archive from {REPO} "
                f"({meta['remote']}); # CHANGE edits listed in PATCHES.json\n")
    for p in (osp.join(a.dest, "PATCHES.json"), osp.join(HERE, "PATCHES.json")):
        with open(p, "w") as f:
            json.dump(meta, f, indent=1)
    make_readonly(a.dest)
    print(json.dumps({k: v for k, v in meta.items() if k != "patched_files"}, indent=1))
    for r in recs:
        print(r["file"], [x["line"] for x in r["lines"]], r["sha256_pristine"][:12], "->", r["sha256_patched"][:12])
    assert verify(a.dest)


if __name__ == "__main__":
    main()
