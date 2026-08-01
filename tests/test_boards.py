"""Board segmentation: the ink metric, pan compensation, and the zoom merge.

Most of these encode a bug that actually happened. The per-frame quantile,
the inverted phase-correlation sign, the flapping revisit and the drifting
baseline were all found by measurement against real lectures, not by
reasoning; so were the two camera styles that motivate the zoom gate.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import boards as B

W, H, FPS = 320, 180, 5
root = Path(tempfile.mkdtemp(prefix="boards-"))


def blank():
    return np.full((H, W, 3), 40, np.uint8)          # dark green board


def scribble(img, seed, n=40):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x, y = rng.integers(4, W - 12), rng.integers(4, H - 6)
        img[y:y + 2, x:x + 10] = 230                 # a stroke of chalk
    return img


board_a = scribble(blank(), 1)
board_a2 = scribble(board_a.copy(), 2, n=25)         # more writing, same board
panned = np.roll(board_a2, 60, axis=1)               # camera moves right
board_b = scribble(blank(), 9, n=45)                 # after an erasure
board_c = scribble(blank(), 21, n=50)

# A, more on A, pan off and back (must NOT split), a different board B, back
# to A (a real revisit), then C.
script = [(30, board_a), (25, board_a2), (15, panned), (20, board_a2),
          (35, board_b), (25, board_a2), (30, board_c)]

raw = root / "raw.rgb"
with open(raw, "wb") as f:
    person_x = 0
    for secs, base in script:
        for _ in range(secs * FPS):
            fr = base.copy()
            person_x = (person_x + 7) % (W - 30)     # a lecturer pacing
            fr[H // 3:, person_x:person_x + 26] = 90
            f.write(fr.tobytes())
video = root / "lecture.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                "-i", str(raw), "-pix_fmt", "yuv420p", str(video)], check=True)
print(f"synthetic lecture: {sum(s for s, _ in script)}s")

# --- the lecturer must be removed by the temporal median --------------------
frames = B.sample(video, fps=1, width=96)
med = B._rolling_median(frames, 15)
assert np.abs(frames[20] - med[20]).mean() > 2, \
    "the median is not removing the moving figure"

# --- the ink threshold is global, so density actually varies ----------------
thr = B.ink_threshold(med)
dens = np.array([B._ink(m, thr).mean() for m in med])
assert dens.max() > dens.min() * 1.3, \
    "density is ~constant — a per-frame quantile would do this"
print(f"lecturer removed; ink density {dens.min():.4f}–{dens.max():.4f}")

# --- containment is asymmetric: adding keeps it, erasing kills it -----------
ma, ma2, mb = (B._ink(x.mean(axis=2).astype(np.float32), thr)
               for x in (board_a, board_a2, board_b))
assert B._containment(ma, ma2) > 0.8, "adding writing must keep the board"
assert B._containment(ma, mb) < 0.5, "a fresh board must not match"

# --- a camera pan is motion, not an erasure (the sign was once inverted) ----
dy, dx = B._shift(ma2, B._ink(np.roll(board_a2, 60, axis=1).mean(axis=2)
                              .astype(np.float32), thr))
assert abs(abs(dx) - 60) <= 6, f"pan not recovered: {dx}"
print(f"containment asymmetric; pan recovered as dx={dx}")

# --- the zoom matcher -------------------------------------------------------
def mk(seed, n=60):
    r = np.random.default_rng(seed)
    m = np.zeros((90, 160), bool)
    for _ in range(n):
        y, x = r.integers(4, 86), r.integers(4, 150)
        m[y:y + 2, x:x + 8] = True
    return m


one, other = mk(1), mk(9)
for s in (0.6, 0.7, 0.8, 1.25, 1.5, 1.75, 2.0):
    score, _ = B.same_under_zoom(one, B._rescale(one, s))
    assert score >= B.ZOOM_MATCH, f"a {s}x reframe scored {score:.2f}"
assert B.same_under_zoom(one, other)[0] < B.ZOOM_MATCH
print("zoom matcher: reframes 0.6x–2x match, an unrelated board does not")

# --- but it is only ever applied to a camera that moves ---------------------
# A locked-off camera shows global motion in a fraction of a percent of
# samples, an operated one in tens of percent. This is what keeps the merge
# away from static lectures, where it destroys correct boards.
# A static camera watching a board being written on: the content grows, but
# nothing translates. (Cycling unrelated masks is NOT this — phase
# correlation between two different boards peaks wherever it likes, which is
# how this test first failed.)
base = mk(1)
still = []
for i in range(40):
    m = base.copy()
    r = np.random.default_rng(100 + i)
    for _ in range(i):                       # a few more strokes each sample
        y, x = r.integers(4, 86), r.integers(4, 150)
        m[y:y + 2, x:x + 8] = True
    still.append(m)
assert B.camera_motion(still) < B.MOVING_CAMERA, \
    f"a static camera must read still, got {B.camera_motion(still):.2f}"
# The same board, reframed: content identical, position not.
moving = [np.roll(still[i], (0, 15 * (i % 2)), axis=(0, 1)) for i in range(40)]
assert B.camera_motion(moving) >= B.MOVING_CAMERA, \
    f"a panning camera must read moving, got {B.camera_motion(moving):.2f}"
print("camera motion separates a locked-off camera from an operated one")


class _B:                                    # a Board stand-in for merge_zoomed
    def __init__(self, m, lo, hi):
        self.mask, self.hires, self.image = m, None, None
        self.intervals, self.peak_ink, self.peak_at = [[lo, hi]], 0.1, lo

    @property
    def seconds(self):
        return sum(b - a for a, b in self.intervals)


pair = [_B(one, 0, 60), _B(B._rescale(one, 1.5), 60, 120)]
assert len(B.merge_zoomed(pair, B.ZOOM_MATCH)) == 1, "a reframe must fold"
assert B.merge_zoomed(pair, B.ZOOM_MATCH)[0].intervals == [[0, 120]], \
    "folding two views must join their intervals"
assert len(B.merge_zoomed([_B(one, 0, 60), _B(other, 60, 120)],
                          B.ZOOM_MATCH)) == 2, "distinct boards must survive"
print("merge folds a reframe, joins intervals, keeps distinct boards apart")

# --- end to end --------------------------------------------------------------
res = B.analyse(video, root / "out", fps=1, window=15, min_seconds=5,
                progress=lambda m: None)
ids = [b["id"] for b in res["boards"]]
assert 3 <= len(ids) <= 5, f"expected ~3 real boards, got {len(ids)}"
revisited = [b for b in res["boards"] if b["revisits"] >= 1]
assert revisited, "returning to an earlier board was not recognised"
spans = revisited[0]["intervals"]
# The point is that the board was come back to *after another board*, so
# somewhere in its intervals there is a gap long enough to hold one. Which
# gap is not the property under test — short gaps are the debounced pan.
gaps = [b[0] - a[1] for a, b in zip(spans, spans[1:])]
assert len(spans) >= 2 and max(gaps) > 20, \
    f"no gap long enough for the intervening board: {spans}"
assert (root / "out" / "boards.json").exists()
assert all((root / "out" / b["image"]).exists()
           for b in res["boards"] if b["image"])
for b in res["boards"]:
    for lo, hi in b["intervals"]:
        assert hi > lo
print(f"end to end: {len(ids)} boards, revisit spans {spans}")

print("\nALL OK")
