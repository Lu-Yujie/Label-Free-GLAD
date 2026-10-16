"""Rebuild adapter files that contain code from upstream repositories without a licence.

Some of our adapter files for MUSE, GLADMamba, HC-GLAD, ProtoGLAD and SUL/GUL reuse lines of the official code. We do
not redistribute those lines. Instead, each such file is shipped as a recipe (evalpaper/ext/<M>/rebuild/*.recipe)
that holds only our own lines plus line references (file, line number, indentation) into the upstream repository at
the commit we evaluated. This script reads the fetched clone in evalpaper/ext/<M>/src/, rebuilds every target file
and checks it against the sha256 stored in the recipe. It stops with an error if any check fails.

  python3 scripts/rebuild_from_upstream.py evalpaper/ext/MUSE/rebuild/*.recipe     # from the repository root

The fetch scripts (evalpaper/ext/<M>/fetch_src.sh) call it after cloning. Standard library only.

Recipe format (one header block, then operations):
  #target <path>        file to write, relative to the repository root
  #sha256 <hex>         sha256 of the rebuilt file
  #commit <hex>         upstream commit the line references point into
  #final-newline 0|1
  O <n>                 the next n lines are ours, copied as they are
  U <file> <start> <n> <indents>   n consecutive upstream lines from <file> (relative to src/), starting at line
                        <start>; <indents> is a JSON list with the leading whitespace of each line (an int = that
                        many spaces, a string = literal whitespace, null = an empty line)
  S <file> <line> <indent> <suffix>   one upstream line with our text appended (<suffix> is a JSON string)
"""
import hashlib
import json
import os
import os.path as osp
import subprocess
import sys

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))


def die(msg):
    sys.stderr.write("REBUILD FAILED: " + msg + "\n")
    sys.exit(1)


def content(line):
    return line.rstrip("\r\n").strip()


def ws(x):
    return " " * x if isinstance(x, int) else x


def rebuild(recipe_path):
    src_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(recipe_path))), "src")
    lines = open(recipe_path, encoding="utf-8").read().split("\n")
    head, i = {}, 0
    while i < len(lines) and lines[i].startswith("#"):
        k, _, v = lines[i][1:].partition(" ")
        head[k] = v.strip()
        i += 1
    for k in ("target", "sha256", "commit", "final-newline"):
        if k not in head:
            die(f"{recipe_path}: header '#{k}' missing")
    if not osp.isdir(src_dir):
        die(f"{src_dir} not found; run the fetch script of this method first")
    try:
        head_commit = subprocess.run(["git", "-C", src_dir, "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()
    except OSError:
        head_commit = ""
    if head_commit and head_commit != head["commit"]:
        die(f"{src_dir} is at {head_commit}, the recipe needs {head['commit']}")
    cache = {}

    def up(f):
        if f not in cache:
            p = osp.join(src_dir, f)
            if not osp.exists(p):
                die(f"{recipe_path}: upstream file {f} not found in {src_dir}")
            cache[f] = open(p, encoding="utf-8").read().splitlines()
        return cache[f]

    out = []
    while i < len(lines):
        op = lines[i]
        i += 1
        if not op:
            continue
        kind, rest = op[0], op[2:]
        if kind == "O":
            n = int(rest)
            out.extend(lines[i:i + n])
            i += n
        elif kind == "U":
            f, start, n, ind = rest.split(" ", 3)
            start, n, ind = int(start), int(n), json.loads(ind)
            src = up(f)
            if len(ind) != n or start + n - 1 > len(src):
                die(f"{recipe_path}: bad U operation '{op[:80]}'")
            for k in range(n):
                c = content(src[start - 1 + k])
                if ind[k] is None:
                    if c:
                        die(f"{recipe_path}: {f}:{start + k} is not empty as expected")
                    out.append("")
                else:
                    out.append(ws(ind[k]) + c)
        elif kind == "S":
            f, ln, ind, suffix = rest.split(" ", 3)
            src = up(f)
            out.append(ws(json.loads(ind)) + content(src[int(ln) - 1]) + json.loads(suffix))
        else:
            die(f"{recipe_path}: unknown operation '{op[:40]}'")
    text = "\n".join(out) + ("\n" if head["final-newline"] == "1" else "")
    data = text.encode("utf-8")
    got = hashlib.sha256(data).hexdigest()
    if got != head["sha256"]:
        die(f"{head['target']}: sha256 {got} != expected {head['sha256']}")
    dst = osp.join(ROOT, head["target"])
    os.makedirs(osp.dirname(dst), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(data)
    print(f"rebuilt {head['target']} (sha256 {got[:16]}..., OK)")


def main():
    if len(sys.argv) < 2:
        die("usage: rebuild_from_upstream.py RECIPE [RECIPE ...]")
    for p in sys.argv[1:]:
        rebuild(p)


if __name__ == "__main__":
    main()
