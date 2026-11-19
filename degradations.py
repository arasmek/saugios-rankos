"""Video degradations shared by the pose-estimator notebooks.

The test definitions live in `tests.txt` (JSON) and are read into a variable called TESTS. The filter functions below
are the ones from `mediapipe_classifier.ipynb`, so a test name and level mean the same degradation for every model.
Scenario names are built exactly like in the MediaPipe notebooks ("resolution_96", "noise_10_seed0", ...).
"""
import hashlib
import itertools
import json
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path

import cv2
import numpy as np


def ffmpeg():
    """ffmpeg from PATH, otherwise the copy that comes with the imageio-ffmpeg package."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def read_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()
    if not frames:
        raise IOError(f"Could not read any frames from {path}")
    info = Path(str(path) + ".json")
    if info.exists():
        fps = json.loads(info.read_text())["fps"]
    return frames, float(fps)


def even(frames):
    """Most codecs need an even width and height."""
    h, w = frames[0].shape[:2]
    return [f[: h - h % 2, : w - w % 2] for f in frames]


def encode(frames, fps, path, codec, options):
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-an", "-c:v", codec, *map(str, options), str(path)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), capture_output=True, check=True)


def f_resolution(frames, fps, p, rng):
    h, w = frames[0].shape[:2]
    new_h = int(p["height"]) // 2 * 2
    new_w = round(w * new_h / h / 2) * 2
    return [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames], fps


def f_fps(frames, fps, p, rng):
    new_fps = float(p["fps"])
    n = max(1, round(len(frames) * new_fps / fps))
    return [frames[min(int(i * fps / new_fps), len(frames) - 1)] for i in range(n)], new_fps


def f_codec(frames, fps, p, rng):
    """Compress with a real codec and decode again -> genuine compression artefacts."""
    frames = even(frames)
    h, w = frames[0].shape[:2]
    options = ["-pix_fmt", p.get("pix_fmt", "yuv420p")]
    if "crf" in p:
        options += ["-crf", p["crf"]]
    if "bitrate" in p:
        options += ["-b:v", p["bitrate"]]
    ext = {"libvpx-vp9": ".webm", "mjpeg": ".avi"}.get(p["codec"], ".mp4")
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / f"video{ext}"
        encode(frames, fps, file, p["codec"], options)
        raw = subprocess.run([ffmpeg(), "-loglevel", "error", "-i", str(file), "-fps_mode", "passthrough",
                              "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)), fps


def f_blur(frames, fps, p, rng):
    k = int(p["size"]) | 1
    return [cv2.GaussianBlur(f, (k, k), 0) for f in frames], fps


def f_brightness(frames, fps, p, rng):
    return [cv2.convertScaleAbs(f, alpha=1.0, beta=float(p["change"])) for f in frames], fps


def f_noise(frames, fps, p, rng):
    s = float(p["sigma"])
    return [np.clip(f + rng.normal(0, s, f.shape), 0, 255).astype(np.uint8) for f in frames], fps


def f_frame_drop(frames, fps, p, rng):
    """Randomly lost frames are replaced by the previous frame (like a stuttering stream)."""
    out = [frames[0]]
    for f in frames[1:]:
        out.append(out[-1] if rng.random() < float(p["probability"]) else f)
    return out, fps


def f_occlusion(frames, fps, p, rng):
    """A black box (size = fraction of the frame) at a random fixed position."""
    h, w = frames[0].shape[:2]
    bh, bw = int(h * p["size"]), int(w * p["size"])
    y, x = rng.integers(0, h - bh + 1), rng.integers(0, w - bw + 1)
    out = [f.copy() for f in frames]
    for f in out:
        f[y:y + bh, x:x + bw] = 0
    return out, fps


FILTERS = {"resolution": f_resolution, "fps": f_fps, "codec": f_codec, "blur": f_blur,
           "brightness": f_brightness, "noise": f_noise, "frame_drop": f_frame_drop, "occlusion": f_occlusion}
RANDOM_FILTERS = {"noise", "frame_drop", "occlusion"}   # different result every time -> one scenario per seed


def apply_filters(frames, fps, filters, rng):
    for spec in filters:
        params = {k: v for k, v in spec.items() if k != "type"}
        frames, fps = FILTERS[spec["type"]](frames, fps, params, rng)
    return frames, fps


def fill_level(obj, level):
    """Replace the word "LEVEL" in the filter settings with the current level."""
    if isinstance(obj, dict):
        return {k: fill_level(v, level) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill_level(v, level) for v in obj]
    return level if obj == "LEVEL" else obj


def load_tests(path="tests.txt"):
    """Read the test definitions (JSON) and check them, so a typo fails here and not hours into a run."""
    tests = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, t in tests.items():
        if not isinstance(t.get("filters"), list) or not t["filters"] or not isinstance(t.get("levels"), list):
            raise ValueError(f"{path}: test {name!r} needs a non-empty \"filters\" list and a \"levels\" list")
        for spec in t["filters"]:
            if spec.get("type") not in FILTERS:
                raise ValueError(f"{path}: test {name!r} uses unknown filter type {spec.get('type')!r}, "
                                 f"available: {sorted(FILTERS)}")
    return tests


def short_hash(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:6]


ORIGINAL = {"scenario": "original", "test": "original", "level": "original", "level_index": -1,
            "order": "", "seed": None, "filters": []}


def make_scenarios(tests, run_tests, seeds):
    """The original video first, then one scenario per level (and per seed for random filters, per order for
    tests with "all_orders"). Same names and fields as `make_scenarios` in the MediaPipe notebooks."""
    unknown = [t for t in run_tests if t not in tests]
    if unknown:
        raise KeyError(f"RUN_TESTS names not found in the tests file: {unknown}, available: {list(tests)}")
    scenarios = [ORIGINAL]
    for test in run_tests:
        t = tests[test]
        for i, level in enumerate(t["levels"]):
            filters = fill_level(t["filters"], level)
            orders = list(itertools.permutations(filters)) if t.get("all_orders") else [filters]
            is_random = any(f["type"] in RANDOM_FILTERS for f in filters)
            for order in orders:
                for seed in (seeds if is_random else [None]):
                    name = f"{test}_{level}"
                    if t.get("all_orders"):
                        name += "_" + "-then-".join(f["type"] for f in order)
                    if seed is not None:
                        name += f"_seed{seed}"
                    scenarios.append({"scenario": name, "test": test, "level": level, "level_index": i,
                                      "order": " > ".join(f["type"] for f in order), "seed": seed,
                                      "filters": list(order)})
    return scenarios


def scenario_folder(s):
    """Folder name of a scenario. The hash of its settings means edited test parameters never reuse old results."""
    return s["scenario"] if not s["filters"] else f"{s['scenario']}_{short_hash([s['filters'], s['seed']])}"


def scenario_rng(s, video_id):
    """Same random numbers per scenario and video as in the MediaPipe notebooks (video_id = "<sign>__<video>")."""
    return np.random.default_rng([s["seed"] or 0, zlib.crc32(video_id.encode())])
