"""Run a study script with its machine-specific data paths mapped to $GLAD_DATA_ROOT.

Some files are released byte for byte because their sha256 is fixed in the paper or in the study's frozen records
(README.md lists them). A few of them contain absolute data paths of the machine the study ran on, all of the form
<prefix>/grail_data/<rest>, and the per-run records in the released archives store data paths as
$GLAD_DATA_ROOT/<rest>. This wrapper maps both kinds of path to $GLAD_DATA_ROOT/<rest> (the first only when it does
not exist on your machine) and then runs the script unchanged:

  export GLAD_DATA_ROOT=/path/to/glad_data
  python -B scripts/portable_run.py evalpaper/stats/glad_stats.py --final

Paths that exist are never changed, and nothing else about the script is changed. Our other (non-frozen) files read
GLAD_DATA_ROOT directly and do not need the wrapper, but running them through it does no harm.
"""
import builtins
import glob
import io
import os
import runpy
import sys
import tempfile

sys.dont_write_bytecode = True

KEY = "/grail_data"
ROOT = os.environ.get("GLAD_DATA_ROOT")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = [os.path.abspath(x) + os.sep for x in (ROOT, REPO, tempfile.gettempdir()) if x]   # never remapped
ALWAYS = os.environ.get("GLAD_REMAP_ALWAYS") == "1"     # map machine paths even where they exist (for testing)


def remap(p):
    if ROOT is None or isinstance(p, (bytes, int)):
        return p
    try:
        s = os.fspath(p)
    except TypeError:
        return p
    if not isinstance(s, str):
        return p
    for var in ("$GLAD_DATA_ROOT/", "${GLAD_DATA_ROOT}/"):
        if s.startswith(var):
            return os.path.join(ROOT, s[len(var):])
    if not s.startswith("/") or any((s + os.sep).startswith(k) for k in KEEP):
        return p
    i = s.find(KEY + "/")
    if i < 0:
        if not s.endswith(KEY):
            return p
        rest = ""
    else:
        rest = s[i + len(KEY) + 1:]
    if not ALWAYS and _exists(s):
        return p
    return os.path.join(ROOT, rest) if rest else ROOT


_stat = os.stat


def _exists(s):
    try:
        _stat(s)
        return True
    except (OSError, ValueError):
        return False


def _wrap1(f):
    def g(path, *a, **k):
        return f(remap(path), *a, **k)
    g.__wrapped__ = f
    return g


def _wrap2(f):
    def g(src, dst, *a, **k):
        return f(remap(src), remap(dst), *a, **k)
    g.__wrapped__ = f
    return g


def install():
    builtins.open = _wrap1(builtins.open)
    io.open = builtins.open
    for name in ("stat", "lstat", "listdir", "scandir", "makedirs", "mkdir", "remove", "unlink", "rmdir", "access",
                 "chmod", "utime"):
        if hasattr(os, name):
            setattr(os, name, _wrap1(getattr(os, name)))
    for name in ("replace", "rename"):
        setattr(os, name, _wrap2(getattr(os, name)))
    glob.glob = _wrap1(glob.glob)
    glob.iglob = _wrap1(glob.iglob)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if ROOT is None:
        sys.exit("set GLAD_DATA_ROOT first (see README.md)")
    script = os.path.abspath(sys.argv[1])
    install()
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(script))
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
