"""crop_board, the locator's box, and boards-on-by-default."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import boards as B
from claude_backend import parse_box
from notes_tools import NotesToolContext, build_handlers, build_tools

root = Path(tempfile.mkdtemp(prefix="crop-"))
img = root / "board.jpg"          # a full-resolution still, as stored on disk
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi",
                "-i", "testsrc=size=1920x1080:duration=1:rate=1",
                "-frames:v", "1", str(img)], check=True)
assert B._probe(img) == (1920, 1080)

# --- the crop ---------------------------------------------------------------
out = B.zoom(img, root / "c1.jpg", (0.5, 0.2, 0.5, 0.55))
assert out, "a legitimate crop failed"
path, gain = out
w, h = B._probe(path)
# Half of a 1920 still is 960 real pixels; the whole still would have been
# squeezed to 1568, i.e. 0.82x. So the crop is worth about 1.2x.
assert w == 960, f"the crop must be native pixels, got {w}"
assert 1.1 < gain < 1.3, f"gain against the whole still: {gain}"
print(f"crop: 960x{h} native, {gain:.2f}x the detail of the whole still")

# It must NEVER upscale — that is interpolation, and interpolation invents no
# chalk. The crop only avoids throwing detail away.
tiny, tiny_gain = B.zoom(img, root / "c2.jpg", (0.0, 0.0, 0.1, 0.1))
assert B._probe(tiny)[0] == 192, f"a 10% crop of 1920 is 192px, got {B._probe(tiny)[0]}"
assert tiny_gain > 1
big, _ = B.zoom(img, root / "c3.jpg", (0.0, 0.0, 1.0, 1.0))
assert B._probe(big)[0] == B.ZOOM_WIDTH, "capped at the vision ceiling"
print("never upscales; capped at the vision ceiling")

# a box off the edge is clamped, degenerate boxes are refused
assert B.zoom(img, root / "c4.jpg", (0.8, 0.8, 0.5, 0.5)) is not None
assert B.zoom(img, root / "c5.jpg", (0.5, 0.5, 0.01, 0.5)) is None
assert B.zoom(img, root / "c6.jpg", (2.0, 0.0, 0.5, 0.5)) is None

# Stills are stored bigger than we ever send, so crops have pixels to take.
assert B.SNAPSHOT_WIDTH > B.ZOOM_WIDTH

# --- the tool ---------------------------------------------------------------
ctx = NotesToolContext(refs_dir=root / "refs", diagrams_dir=root / "dgm",
                       boards=[{"id": 7, "path": img, "best_at": 30.0,
                                "intervals": [[0, 60]], "revisits": 0}])
assert {"crop_board", "check_diagram"} <= {t["name"] for t in build_tools(ctx)}
h = build_handlers(ctx)
r = h["crop_board"]({"board": 7, "x": 0.5, "y": 0.0, "width": 0.5,
                     "height": 1.0})
assert not r.is_error and [b["type"] for b in r.content] == ["text", "image"]
assert "native resolution" in r.content[0]["text"]
assert h["crop_board"]({"board": 99, "x": 0, "y": 0,
                        "width": 1, "height": 1}).is_error
bad = h["crop_board"]({"board": 7, "x": 0, "y": 0, "width": 0.001, "height": 1})
assert bad.is_error and "unusable" in bad.content
assert (root / "dgm" / "crops" / "crop-001.jpg").exists()
print("tool ok: native crop returned as an image, bad input rejected")

bare = NotesToolContext(refs_dir=root / "refs", diagrams_dir=root / "dgm")
assert "crop_board" not in {t["name"] for t in build_tools(bare)}
assert "crop_board" not in build_handlers(bare)

# --- the locator's reply ----------------------------------------------------
box, why = parse_box('{"x":0.5,"y":0.18,"width":0.48,"height":0.5,'
                     '"note":"right panel"}')
assert box and box["x"] == 0.5 and box["note"] == "right panel" and not why
box2, _ = parse_box('Sure:\n```json\n{"x":0,"y":0,"width":1,"height":1}\n```')
assert box2 and box2["width"] == 1.0
none1, why1 = parse_box('{"error":"no diagram on this board"}')
assert none1 is None and "no diagram" in why1
none2, why2 = parse_box("I could not find it")
assert none2 is None and why2
none3, why3 = parse_box('{"x":0.1,"y":0.2}')
assert none3 is None and "missing" in why3
print("box parsing ok: fences, errors and malformed boxes all handled")

# --- segmentation is on by default ------------------------------------------
src = (REPO / "build_course.py").read_text()
assert '"--no-boards"' in src and 'action="store_false"' in src
# --boards survives only as a suppressed no-op: without it argparse's prefix
# matching would read a leftover `--boards` as `--boards-color`.
assert 'dest="boards_legacy"' in src and "args.boards_legacy" in src

p = argparse.ArgumentParser()
p.add_argument("--no-boards", dest="boards", action="store_false")
p.add_argument("--boards-color", action="store_true")
p.add_argument("--boards", dest="boards_legacy", action="store_true")
for argv, want in ((["--boards"], (True, False, True)),
                   ([], (True, False, False)),
                   (["--no-boards"], (False, False, False)),
                   (["--boards-color"], (True, True, False))):
    a = p.parse_args(argv)
    assert (a.boards, a.boards_color, a.boards_legacy) == want, argv
print("boards default on (--no-boards to skip); a stale --boards is inert")

print("\nALL OK")
