"""
boards.py — find the blackboards in a lecture video and when each was current.

The note-writer's only real input is a linear transcript, so everything the
lecturer *drew* is lost. This module recovers the structure: it segments the
video into board states, so a later pass can read each board once, in full,
instead of sampling frames blindly.

How it works (no ML, no extra dependencies — ffmpeg for pixels, numpy for the
rest):

  1. Sample the video at ~1 fps, downscaled and by default greyscale.
  2. Remove the lecturer with a sliding temporal median: they move, the board
     does not, so the median over a window of seconds is the board alone.
  3. Reduce each median frame to an *ink mask* — pixels of high local
     contrast. This is invariant to whether the board is dark with chalk or
     white with marker, and to overall lighting.
  4. Track boards by ink *containment* rather than similarity. If the ink of
     the current board is still (mostly) present in a later frame, it is the
     same board — whether the lecturer added to it, or the camera panned away
     and came back. When the old ink disappears, the board was erased and a
     new one begins. Frames are motion-compensated first (phase correlation),
     so a camera pan does not read as an erasure.

That containment test is what handles the two awkward cases: writing more on
an earlier board, and the camera revisiting one. Both produce another
*interval* on an existing board rather than a new board.

The best snapshot of a board is its ink peak — the moment just before it was
erased, when it is most complete.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SAMPLE_FPS = 1.0
SAMPLE_WIDTH = 192          # analysis resolution; snapshots are full-res
MEDIAN_WINDOW = 15          # seconds of temporal median (lecturer removal)
INK_QUANTILE = 0.985        # local-contrast quantile that counts as ink
MIN_INK = 0.002             # below this the board is essentially blank
KEEP = 0.55                 # old ink still present => same board
REVISIT = 0.55              # ...including after the camera looked elsewhere
MIN_SECONDS = 20.0          # ignore boards that were current only briefly
DEBOUNCE = 4                # samples a change must persist before it is a cut


VERIFY_WIDTH = 512          # resolution for confirming a board is the same one
VERIFY_SPAN = 10.0
SCREEN = 0.45               # cheap low-res score needed before verifying


@dataclass
class Board:
    id: int
    intervals: list = field(default_factory=list)   # [start, end] seconds
    peak_ink: float = 0.0
    peak_at: float = 0.0
    mask: np.ndarray | None = None                  # ink at its fullest
    hires: np.ndarray | None = None                 # ...at verification size
    image: str | None = None

    @property
    def seconds(self) -> float:
        return sum(b - a for a, b in self.intervals)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "intervals": [[round(a, 1), round(b, 1)] for a, b in self.intervals],
            "seconds": round(self.seconds, 1),
            "revisits": max(0, len(self.intervals) - 1),
            "best_at": round(self.peak_at, 1),
            "ink": round(self.peak_ink, 4),
            "image": self.image,
        }


def _probe(video: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
         str(video)], capture_output=True, text=True).stdout.strip()
    w, h = (out.split("\n")[0].split("x") + ["1920", "1080"])[:2]
    return int(w), int(h)


def sample(video: Path, fps: float = SAMPLE_FPS, width: int = SAMPLE_WIDTH,
           color: bool = False) -> np.ndarray:
    """Downscaled frames as (N, H, W) greyscale or (N, H, W, 3) colour."""
    src_w, src_h = _probe(video)
    height = max(2, int(round(width * src_h / src_w / 2)) * 2)
    pix, chans = ("rgb24", 3) if color else ("gray", 1)
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"fps={fps},scale={width}:{height}",
         "-pix_fmt", pix, "-f", "rawvideo", "-"],
        capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()[:400]}")
    frame_bytes = width * height * chans
    n = len(proc.stdout) // frame_bytes
    arr = np.frombuffer(proc.stdout[:n * frame_bytes], dtype=np.uint8)
    shape = (n, height, width, chans) if color else (n, height, width)
    return arr.reshape(shape).astype(np.float32)


def _rolling_median(frames: np.ndarray, window: int) -> np.ndarray:
    """The board without the lecturer: they move through a window of seconds,
    the writing does not, so the per-pixel median keeps only the board."""
    n = len(frames)
    half = max(1, window // 2)
    out = np.empty_like(frames)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = np.median(frames[lo:hi], axis=0)
    return out


def _contrast(frame: np.ndarray) -> np.ndarray:
    """Local contrast — high where something is written, whatever the board
    colour or the lighting."""
    if frame.ndim == 3:
        frame = frame.mean(axis=2)
    gy, gx = np.gradient(frame)
    return np.hypot(gx, gy)


def ink_threshold(medians: np.ndarray, quantile: float = INK_QUANTILE) -> float:
    """One contrast threshold for the whole video.

    It must be global: a per-frame quantile would mark the same fraction of
    every frame as ink by construction, so a full board and a wiped one would
    score identically and no erasure could ever be detected."""
    sample_idx = np.linspace(0, len(medians) - 1, min(60, len(medians)))
    mags = np.concatenate([_contrast(medians[int(i)]).ravel()
                           for i in sample_idx])
    return float(max(np.quantile(mags, quantile), 4.0))


def _ink(frame: np.ndarray, threshold: float) -> np.ndarray:
    return _contrast(frame) > threshold


def _shift(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """Integer (dy, dx) by phase correlation, such that rolling `b` by
    (dy, dx) brings it back into register with `a` — so a camera pan reads as
    motion rather than as the board having been erased."""
    fa = np.fft.rfft2(a.astype(np.float32))
    fb = np.fft.rfft2(b.astype(np.float32))
    denom = np.abs(fa) * np.abs(fb)
    denom[denom == 0] = 1e-9
    corr = np.fft.irfft2(fa * np.conj(fb) / denom, s=a.shape)
    dy, dx = np.unravel_index(int(np.argmax(corr)), corr.shape)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return int(dy), int(dx)


def _containment(old: np.ndarray, new: np.ndarray,
                 align: bool = True) -> float:
    """How much of `old`'s ink survives in `new` (0..1).

    Deliberately asymmetric: adding writing to a board leaves containment
    high, while erasing it drops sharply. That asymmetry is what separates
    'the same board, with more on it' from 'a new board'."""
    if not old.any():
        return 0.0
    direct = float((old & new).sum() / old.sum())
    # Alignment costs two FFTs, so the per-frame "is this still the same
    # board" check runs unaligned: a pan simply ends the interval, and the
    # aligned comparison then happens once, when deciding whether the new
    # view is actually a board we have seen before. A large pan drives
    # direct containment to nearly zero, so it must not be an early-out.
    if not align or direct >= 0.9:
        return direct
    dy, dx = _shift(old, new)
    if not (dy or dx):
        return direct
    rolled = np.roll(new, (dy, dx), axis=(0, 1))
    return max(direct, float((old & rolled).sum() / old.sum()))


MOVING_CAMERA = 0.05        # fraction of samples showing global motion


def camera_motion(masks: list, step: int = 5) -> float:
    """Fraction of sampled moments at which the whole frame has moved.

    The lecturer is already gone — these are lecturer-free medians — so any
    global translation is the camera. This is what decides whether the
    zoom-merge pass is safe to run: it is only ever needed where an operator
    reframes, and it is only ever harmful where the camera is bolted down.
    Measured on this course the two styles are a hundred-fold apart, 0.2%
    against 24%, so the test does not need to be delicate."""
    moved = total = 0
    for a, b in zip(masks[::step], masks[step::step]):
        if not (a.any() and b.any()):
            continue
        dy, dx = _shift(a, b)
        total += 1
        moved += abs(dy) + abs(dx) > 3
    return moved / total if total else 0.0


def merge_zoomed(boards: list, threshold: float = 0.72) -> list:
    """Fold together consecutive boards that are one board at two zooms.

    Gated on camera_motion, because it is right for one camera style and
    wrong for the other.

    The problem is real: one of the two lecturers in the test course is
    filmed by an operator who reframes constantly, and because the per-frame
    test compensates for translation but not scale, every zoom reads as a
    fresh board. That lecturer averages 79 boards a lecture against the
    other's 28, and consecutive "boards" are demonstrably the same writing
    at two magnifications.

    But this fix misidentifies the signal. Scoring every consecutive pair on
    two real lectures:

        Whitlock (21 boards, all genuine)   0.60 … 0.91
        Ostrand (108 boards, many dupes)    0.22 … 1.00

    No threshold separates them. Genuinely distinct boards score as high as
    0.91 because consecutive boards on a multi-panel slate legitimately
    share most of their ink — the metric is a containment score, and
    containment is high for "same board, more written" and "one panel
    erased" as well as for "same board, zoomed". At 0.72 this merged 51 of
    Ostrand's 108 and also 11 of Whitlock's 21, which were correct.

    Rather than sharpen the metric, the caller asks a question the metric
    cannot: does this camera move at all? Where it does not, no reframe can
    have happened and the pass is skipped entirely. That leaves the static
    lectures exactly as they were and the mobile ones merged — imperfect,
    since an Ostrand board wrongly merged is still possible, but the failure
    is bounded to the lectures that need the pass at all.

    A sharper metric would still be better: require the best-fitting scale
    to be clearly different from 1 AND the fit there to beat the fit at
    scale 1, testing "this content, rescaled" rather than "this content,
    still mostly here"."""
    if not boards:
        return boards
    out = [boards[0]]
    for b in boards[1:]:
        prev = out[-1]
        if prev.mask is None or b.mask is None:
            out.append(b)
            continue
        score, scale = same_under_zoom(prev.mask, b.mask)
        if score < threshold:
            out.append(b)
            continue
        # One board. Keep the fuller view as the record of it, since that is
        # what the snapshot will be taken from.
        prev.intervals = sorted(prev.intervals + b.intervals)
        if b.peak_ink > prev.peak_ink:
            prev.peak_ink, prev.peak_at = b.peak_ink, b.peak_at
            prev.mask, prev.hires = b.mask, b.hires
    for b in out:                       # re-merge intervals made adjacent
        merged = []
        for lo, hi in b.intervals:
            if merged and lo - merged[-1][1] <= 1e-6:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        b.intervals = merged
    return out


ZOOM_SCALES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0)
ZOOM_MATCH = 0.72           # agreement to call two boards one reframe. Only
                            # applied where the camera actually moves — see
                            # camera_motion and merge_zoomed.
ZOOM_MIN_OVERLAP = 0.25     # ...of the smaller frame, or the match means nothing


def _rescale(mask: np.ndarray, s: float) -> np.ndarray:
    """Nearest-neighbour resample of a boolean mask. Good enough: we are
    matching chalk blobs, not resampling a photograph."""
    h, w = mask.shape
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    yi = np.clip((np.arange(nh) / s).astype(int), 0, h - 1)
    xi = np.clip((np.arange(nw) / s).astype(int), 0, w - 1)
    return mask[yi][:, xi]


def _dilate(mask: np.ndarray, r: int = 2) -> np.ndarray:
    """Fatten a mask by r pixels.

    Chalk strokes are one or two pixels wide at analysis resolution, so a
    rescale that is a few percent off — which any finite set of trial scales
    will be — slides them clean past each other and a true match scores as a
    miss. Comparing against a fattened copy tolerates that, and is cheaper
    and steadier than searching a finer grid of scales."""
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy or dx:
                out |= np.roll(mask, (dy, dx), axis=(0, 1))
    return out


def _on_canvas(mask: np.ndarray, shape: tuple) -> tuple:
    """Centre a mask on a canvas; also return which of the canvas it covers.

    Centred, because a zoom is about the middle of the frame — anchoring at
    a corner throws the correspondence away before alignment can find it."""
    H, W = shape
    h, w = mask.shape
    data = np.zeros(shape, bool)
    valid = np.zeros(shape, bool)
    y0, x0 = (H - h) // 2, (W - w) // 2
    ys, ye = max(0, y0), min(H, y0 + h)
    xs, xe = max(0, x0), min(W, x0 + w)
    if ye > ys and xe > xs:
        data[ys:ye, xs:xe] = mask[ys - y0:ye - y0, xs - x0:xe - x0]
        valid[ys:ye, xs:xe] = True
    return data, valid


def _overlap_score(old: np.ndarray, new: np.ndarray) -> float:
    """How much of `old`'s ink survives in `new`, counting only the part of
    `old` that `new` actually shows.

    This is the whole difference between a zoom and an erasure. When the
    camera zooms in, most of the old board is off-frame — not wiped — and
    plain containment reads that missing ink as erased. Restricting to the
    shared footprint asks the question that has an answer: of the writing
    both views can see, how much is still there?"""
    if not old.any() or not new.any():
        return 0.0
    shape = (max(old.shape[0], new.shape[0]), max(old.shape[1], new.shape[1]))
    a, a_valid = _on_canvas(old, shape)
    b, b_valid = _on_canvas(new, shape)
    dy, dx = _shift(a, b)
    best = 0.0
    for rolled, valid in (((b, b_valid) if not (dy or dx) else
                           (np.roll(b, (dy, dx), axis=(0, 1)),
                            np.roll(b_valid, (dy, dx), axis=(0, 1)))),
                          (b, b_valid)):
        shared = a_valid & valid
        seen = a & shared
        if seen.sum() < ZOOM_MIN_OVERLAP * a.sum() or not seen.any():
            continue        # too little of the old board is in shot to judge
        best = max(best, float((seen & _dilate(rolled)).sum() / seen.sum()))
    return best


def same_under_zoom(old: np.ndarray, new: np.ndarray,
                    scales: tuple = ZOOM_SCALES) -> tuple[float, float]:
    """(best score, scale) for `new` being `old` seen at a different zoom.

    Both directions are tried, because a zoom in and a zoom out are the same
    event seen from either end: zooming in, the new view is a magnified crop
    of the old; zooming out, the old is a magnified crop of the new."""
    best, best_s = _overlap_score(old, new), 1.0
    for s in scales:
        score = max(_overlap_score(old, _rescale(new, s)),
                    _overlap_score(_rescale(old, 1 / s), new))
        if score > best:
            best, best_s = score, s
    return best, best_s


def _median_at(video: Path, at: float, width: int = VERIFY_WIDTH,
               span: float = VERIFY_SPAN) -> np.ndarray | None:
    """Lecturer-free greyscale frame at `at`, at verification resolution."""
    src_w, src_h = _probe(video)
    height = max(2, int(round(width * src_h / src_w / 2)) * 2)
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{max(0.0, at - span / 2):.2f}",
         "-i", str(video), "-t", f"{span:.2f}",
         "-vf", f"fps=1,scale={width}:{height}",
         "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True)
    n = len(p.stdout) // (width * height)
    if p.returncode != 0 or n < 1:
        return None
    a = np.frombuffer(p.stdout[:n * width * height], np.uint8)
    return np.median(a.reshape(n, height, width).astype(np.float32), axis=0)


class _Verifier:
    """Decides whether two moments show the same board, at a resolution where
    the answer is meaningful.

    At the coarse analysis size a chalk stroke is about a pixel, so the ink
    mask degenerates into 'there is writing around here' and two unrelated
    but equally busy boards score alike — measured on a real lecture, an
    unrelated pair scored 0.75 against a genuine revisit's 0.78. At 512px the
    same pair scores 0.16 against 0.55. Verification is rare (once per cut),
    so it can afford the extra frames."""

    def __init__(self, video: Path, width: int = VERIFY_WIDTH):
        # Never upscale: enlarging a low-resolution source adds no detail,
        # and fattens strokes until the contrast mask means less, not more.
        self.width = min(width, _probe(video)[0])
        self.video, self.threshold = video, None
        self.calls = 0

    def mask(self, at: float) -> np.ndarray | None:
        frame = _median_at(self.video, at, self.width)
        if frame is None:
            return None
        if self.threshold is None:
            self.threshold = float(max(np.quantile(_contrast(frame),
                                                   INK_QUANTILE), 4.0))
        return _ink(frame, self.threshold)

    def same(self, board: "Board", at: float, need: float) -> bool:
        self.calls += 1
        if board.hires is None:
            board.hires = self.mask(board.peak_at)
        # `at` is the moment the view changed, so a window centred there
        # straddles the change and medians two different boards together.
        # Sample the window that starts once the new view has settled.
        other = self.mask(at + VERIFY_SPAN / 2 + 1.0)
        if board.hires is None or other is None:
            return False
        return _containment(board.hires, other) >= need


def segment(video: Path, *, fps: float = SAMPLE_FPS, color: bool = False,
            window: int = MEDIAN_WINDOW, keep: float = KEEP,
            revisit: float = REVISIT, min_seconds: float = MIN_SECONDS,
            width: int = SAMPLE_WIDTH, debounce: int = DEBOUNCE,
            zoom_match: float = ZOOM_MATCH,
            progress=print) -> tuple[list[Board], dict]:
    """Segment the video into boards. Returns (boards, diagnostics)."""
    frames = sample(video, fps=fps, width=width, color=color)
    progress(f"  sampled {len(frames)} frames at {fps} fps "
             f"({'colour' if color else 'greyscale'}, {width}px wide)")
    medians = _rolling_median(frames, int(window * fps))
    thresh = ink_threshold(medians)
    masks = [_ink(m, thresh) for m in medians]
    density = np.array([m.mean() for m in masks])
    progress(f"  ink density: median {np.median(density):.4f}, "
             f"range {density.min():.4f}–{density.max():.4f} "
             f"(threshold {thresh:.1f})")

    boards: list[Board] = []
    current: Board | None = None
    open_at = 0.0
    cuts = revisits = 0
    pending_cut = 0
    track: np.ndarray | None = None      # the view currently on screen
    verifier = _Verifier(video)

    for i, mask in enumerate(masks):
        t = i / fps
        if density[i] < MIN_INK:            # blank board / cutaway shot
            if current is not None:
                current.intervals.append([open_at, t])
                current = None
            pending_cut = 0
            continue

        if current is not None:
            # Continuity is judged against the view actually on screen, not
            # against the board's fullest moment: after a revisit or a pan we
            # may be looking at a partial view, and comparing to the peak
            # would cut, rematch and cut again every few frames.
            if _containment(track, mask, align=False) >= keep:
                # The baseline only ever advances to a *fuller* board. If it
                # followed every frame it would drift: consecutive frames
                # barely differ, so a board erased and rewritten gradually
                # would track as one continuous board for half the lecture.
                if density[i] > current.peak_ink:
                    current.peak_ink, current.peak_at = density[i], t
                    current.mask = track = mask
                pending_cut = 0
                continue
            # A board change takes a few seconds — wiping, panning, walking
            # across the lens. Reacting to the first differing frame carves
            # out one-second "boards" from the blur mid-transition, which
            # then compete for later matches. Require the change to persist.
            pending_cut += 1
            if pending_cut < debounce:
                continue
            current.intervals.append([open_at, t - (debounce - 1) / fps])
            current = None
            pending_cut = 0
            cuts += 1

        # Does this match a board we have seen before (camera came back, or
        # the lecturer returned to it)? Screen cheaply, then confirm at a
        # resolution where the comparison actually means something.
        ranked = sorted(((_containment(b.mask, mask), b) for b in boards),
                        key=lambda kv: -kv[0])
        match = None
        for score, b in ranked[:3]:
            if score < SCREEN:
                break
            if verifier.same(b, t, revisit):
                match = b
                break
        if match is not None:
            current, open_at, track = match, t, mask
            revisits += 1
        else:
            current = Board(id=len(boards) + 1, peak_ink=density[i],
                            peak_at=t, mask=mask)
            boards.append(current)
            open_at, track = t, mask

    if current is not None:
        current.intervals.append([open_at, len(masks) / fps])

    # Two intervals separated only by the transition we debounced away are
    # one stretch, not a revisit; a real revisit has another board in between.
    gap = (debounce + 2) / fps
    for b in boards:
        merged = []
        for lo, hi in sorted(b.intervals):
            if merged and lo - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        b.intervals = merged

    kept = [b for b in boards if b.seconds >= min_seconds]
    before_zoom = len(kept)
    # Only where an operator actually reframes. On a locked-off camera the
    # merge has nothing to find and measurably destroys correct boards.
    motion = camera_motion(masks)
    if motion >= MOVING_CAMERA:
        progress(f"  camera moves in {motion:.0%} of samples — merging "
                 f"reframes of the same board")
        kept = merge_zoomed(kept, zoom_match)
    for n, b in enumerate(kept, 1):
        b.id = n
    diag = {"frames": len(frames), "fps": fps, "colour": color,
            "boards_found": len(boards), "boards_kept": len(kept),
            "zoom_merged": before_zoom - len(kept),
            "cuts": cuts, "reopens": revisits,
            # After merging, a revisit is a board with a genuine gap in it.
            "revisits": sum(max(0, len(b.intervals) - 1) for b in kept),
            "verifications": verifier.calls,
            "ink_median": round(float(np.median(density)), 5)}
    return kept, diag


SNAPSHOT_WIDTH = 3840       # i.e. "the source, whatever it is" — capped at it
                            # below, never upscaled. Stills are stored at full
                            # resolution because crops are taken from them: a
                            # crop of an already-downscaled still is
                            # interpolation, and interpolation invents no
                            # chalk. Sending the whole still is what gets
                            # downscaled, and that happens at the API.
SNAPSHOT_SPAN = 12.0        # seconds of frames to median over


def snapshot(video: Path, at: float, dest: Path,
             span: float = SNAPSHOT_SPAN, width: int = SNAPSHOT_WIDTH) -> bool:
    """A clean, readable still of the board at `at`.

    Not a single frame: the lecturer is usually standing in front of what
    they just wrote, and that is exactly the part worth reading. Taking the
    per-pixel median of a dozen seconds around the moment removes them and
    leaves the board — the same trick used for the analysis, at full size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    src_w, src_h = _probe(video)
    width = min(width, src_w)
    height = max(2, int(round(width * src_h / src_w / 2)) * 2)
    start = max(0.0, at - span / 2)
    grab = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-i", str(video),
         "-t", f"{span:.2f}", "-vf", f"fps=1,scale={width}:{height}",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True)
    frame_bytes = width * height * 3
    n = len(grab.stdout) // frame_bytes if grab.returncode == 0 else 0
    if n >= 3:
        stack = np.frombuffer(grab.stdout[:n * frame_bytes], dtype=np.uint8)
        stack = stack.reshape(n, height, width, 3)
        clean = np.median(stack, axis=0).astype(np.uint8)
        enc = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-i", "-",
             "-frames:v", "1", "-q:v", "2", str(dest)],
            input=clean.tobytes(), capture_output=True)
        if enc.returncode == 0 and dest.exists():
            return True
    # Fall back to a plain still (short clip, or ffmpeg unhappy).
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(video),
         "-frames:v", "1", "-q:v", "2", str(dest)], capture_output=True)
    return r.returncode == 0 and dest.exists()


ZOOM_WIDTH = 1568           # the vision long-edge ceiling: past it, the API
                            # rescales anyway, so sending more is waste
MIN_ZOOM_FRACTION = 0.04    # a box smaller than this is a misread, not a crop


def zoom(image: Path, dest: Path, box: tuple[float, float, float, float],
         width: int = ZOOM_WIDTH) -> tuple[Path, float] | None:
    """Crop a fractional box out of a board still, at native resolution.

    The point of a crop is that the whole still gets downscaled to the vision
    model's long-edge ceiling, so a chalk arrowhead ends up a pixel or two
    wide. Cropping first means those same pixels arrive un-downscaled. What
    it does NOT do is add detail: this never scales *up*, because upscaling a
    crop is interpolation and interpolation invents no chalk. It only avoids
    throwing detail away — which is why the stills on disk are full
    resolution.

    Returns (path, effective magnification against sending the whole still),
    or None if the box is unusable."""
    x, y, w, h = (float(v) for v in box)
    # Clamp rather than reject: a model's box is approximate by nature.
    x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
    w, h = min(max(w, 0.0), 1.0 - x), min(max(h, 0.0), 1.0 - y)
    if w < MIN_ZOOM_FRACTION or h < MIN_ZOOM_FRACTION:
        return None
    src_w, src_h = _probe(image)
    cw, ch = max(16, int(src_w * w)), max(16, int(src_h * h))
    cx, cy = int(src_w * x), int(src_h * y)
    out_w = min(width, cw)              # never upscale
    out_h = max(2, int(round(out_w * ch / cw / 2)) * 2)
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(image),
         "-vf", f"crop={cw}:{ch}:{cx}:{cy},scale={out_w}:{out_h}:flags=lanczos",
         "-q:v", "2", str(dest)], capture_output=True)
    if r.returncode != 0 or not dest.exists():
        return None
    # What the crop bought: pixels-per-chalk-stroke against the whole still
    # sent at the same ceiling.
    whole = min(width, src_w) / src_w
    return dest, (out_w / cw) / whole


def analyse(video: Path, out_dir: Path, *, color: bool = False,
            progress=print, **kw) -> dict:
    """Segment, save a snapshot per board, and write boards.json."""
    video, out_dir = Path(video), Path(out_dir)
    progress(f"Segmenting boards in {video.name}…")
    boards, diag = segment(video, color=color, progress=progress, **kw)
    out_dir.mkdir(parents=True, exist_ok=True)
    for b in boards:
        name = f"board-{b.id:02d}.jpg"
        if snapshot(video, b.peak_at, out_dir / name):
            b.image = name
    result = {"video": str(video), **diag,
              "boards": [b.to_dict() for b in boards]}
    (out_dir / "boards.json").write_text(json.dumps(result, indent=2))
    progress(f"  {len(boards)} board(s) kept "
             f"({diag['revisits']} revisit(s)) -> {out_dir / 'boards.json'}")
    return result


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="Output dir (default: <video dir>/boards)")
    ap.add_argument("--color", action="store_true",
                    help="Analyse in colour (default: greyscale)")
    ap.add_argument("--fps", type=float, default=SAMPLE_FPS)
    ap.add_argument("--window", type=int, default=MEDIAN_WINDOW,
                    help="Seconds of temporal median used to remove the lecturer")
    ap.add_argument("--keep", type=float, default=KEEP)
    ap.add_argument("--revisit", type=float, default=REVISIT)
    ap.add_argument("--min-seconds", type=float, default=MIN_SECONDS)
    a = ap.parse_args()
    video = Path(a.video)
    out = Path(a.out) if a.out else video.parent / "boards"
    res = analyse(video, out, color=a.color, fps=a.fps, window=a.window,
                  keep=a.keep, revisit=a.revisit, min_seconds=a.min_seconds)
    for b in res["boards"]:
        spans = " ".join(f"{a0/60:.0f}-{b0/60:.0f}m" for a0, b0 in b["intervals"])
        print(f"  board {b['id']:>2}  {b['seconds']/60:5.1f} min  "
              f"peak {b['ink']:.4f} @ {b['best_at']/60:5.1f}m  "
              f"{'revisited ' if b['revisits'] else ''}{spans}")


if __name__ == "__main__":
    main()
