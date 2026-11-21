"""Sign recognition under degradation, the same way for every pose model.

ONE classifier is trained on the clean ("original") keypoints of the training videos of ALL models together, then
used unchanged on the held-out videos in every degraded scenario of every model. Because the classifier and the
videos are the same, differences between the models come from the pose estimator alone.

Input: the unified keypoint files `keypoints/kp_<model>_<scenario>/<sign>/<video>.npz` (written by
`mediapipe_to_keypoints.ipynb` and `sapiens_to_keypoints.ipynb`) with `kp2d` [T, 52, 2] in pixels, `conf` [T, 52]
and `joint_names`. The settings come from the "classifier" part of `eval_setup.txt`.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from degradations import ORIGINAL

BODY_POINTS = ["L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
HANDS = {side: [f"{side}_{f}_{i}" for f in FINGERS for i in range(1, 5)] for side in "LR"}
KEYS = ["model", "scenario", "test", "level", "level_index", "order", "seed"]


def keypoint_file(kp_root, model, scenario, video):
    """keypoints/kp_<model>_<scenario>/<sign>/<video>.npz"""
    return Path(kp_root) / f"kp_{model}_{scenario['scenario']}" / video["sign"] / f"{Path(video['path']).stem}.npz"


def sign_features(file, frames=16, min_conf=0.3, hand_found_share=0.5):
    """One fixed-length feature vector per video. Shoulders, elbows, wrists and the finger joints of both hands
    (face left out), relative to the shoulders (camera position and distance do not matter), at `frames` evenly
    spaced moments of the sign (fps does not matter). A joint below `min_conf` (or not found) counts as missing = 0.
    Plus one "hand found" flag per hand and moment: at least `hand_found_share` of its 20 joints are present."""
    d = np.load(file)
    ix = {str(n): i for i, n in enumerate(d["joint_names"])}
    ok = np.isfinite(d["kp2d"]).all(-1) & (d["conf"] >= min_conf)
    kp = np.where(ok[..., None], d["kp2d"].astype(np.float64), np.nan)
    a, b = ix["L_Shoulder"], ix["R_Shoulder"]
    middle = (kp[:, a] + kp[:, b]) / 2
    width = np.linalg.norm(kp[:, a] - kp[:, b], axis=-1)
    moments = np.linspace(0, len(kp) - 1, frames).round().astype(int)
    points = [ix[n] for n in BODY_POINTS + HANDS["L"] + HANDS["R"]]
    if np.isfinite(middle).any():
        origin, scale = np.nanmedian(middle, axis=0), max(float(np.nanmedian(width)), 1e-6)
        xy = np.clip(np.nan_to_num((kp[moments][:, points] - origin) / scale), -10, 10)
    else:                                               # no shoulders found in any frame: nothing to measure
        xy = np.zeros((frames, len(points), 2))
    found = [ok[moments][:, [ix[n] for n in HANDS[s]]].mean(1) >= hand_found_share for s in "LR"]
    return np.r_[xy.ravel(), np.array(found, float).ravel()]


def make_classifier(name):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    models = {"random_forest": lambda: RandomForestClassifier(n_estimators=300, random_state=0),
              "svm":           lambda: make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10)),
              "knn":           lambda: KNeighborsClassifier(n_neighbors=1),
              "logreg":        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))}
    if name not in models:
        raise ValueError(f"classifier name must be one of {list(models)}, got {name!r}")
    return models[name]()


def split_videos(y, test_share, seed=0):
    """Hold out test_share of the videos of every sign (at least 1, and at least 1 stays for training).
    Returns the train and test positions in y. Fixed seed -> the same split every run."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for sign in sorted(set(y)):
        idx = rng.permutation(np.flatnonzero(y == sign))
        n_test = int(np.clip(round(test_share * len(idx)), 1, len(idx) - 1))
        test += list(idx[:n_test])
        train += list(idx[n_test:])
    return np.array(sorted(train)), np.array(sorted(test))


def evaluate(videos, scenarios, models, cfg, kp_root="keypoints"):
    """Train ONE classifier on the clean keypoints of the training videos of all `models`, then test it on the
    held-out videos in every scenario of every model.
    Returns (predictions, split, skipped): one prediction row per (model, scenario, held-out video); which video is
    train or test; and the scenarios left out because some model has no keypoints for them yet."""
    per_sign = Counter(v["sign"] for v in videos)
    single = sorted(s for s, n in per_sign.items() if n < 2)
    if single:
        print(f"skipped (only one video): {single}")
    vids = [v for v in videos if v["sign"] not in single]
    y = np.array([v["sign"] for v in vids])
    if len(set(y)) < 2:
        raise ValueError("The classifier needs at least 2 signs with 2 or more videos each")
    train, test = split_videos(y, cfg["test_share"])
    split = pd.DataFrame({"video": [v["video"] for v in vids], "sign": y,
                          "set": np.where(np.isin(np.arange(len(vids)), test), "test", "train")})

    def missing(model, scenario, positions):
        return [str(keypoint_file(kp_root, model, scenario, vids[i])) for i in positions
                if not keypoint_file(kp_root, model, scenario, vids[i]).exists()]

    usable, skipped = [], {}
    for s in scenarios:
        need = np.arange(len(vids)) if not s["filters"] else test      # clean: train and test videos, degraded: test only
        lacking = {m: missing(m, s, need) for m in models}
        lacking = {m: f for m, f in lacking.items() if f}
        if not lacking:
            usable.append(s)
        elif not s["filters"]:
            m = next(iter(lacking))
            raise FileNotFoundError(f"No clean keypoints for {m} ({len(lacking[m])} files missing, first: {lacking[m][0]}). "
                                    f"Run the pose notebook and its *_to_keypoints notebook first.")
        else:
            skipped[s["scenario"]] = sorted(lacking)

    feat = {}
    for m in models:
        for s in usable:
            positions = np.arange(len(vids)) if not s["filters"] else test
            computed = {i: sign_features(keypoint_file(kp_root, m, s, vids[i]), cfg["frames"], cfg["min_conf"],
                                         cfg["hand_found_share"]) for i in positions}
            feat[m, s["scenario"]] = computed
        print(f"\r[{m}] features done", end="")
    print()

    clean = ORIGINAL["scenario"]
    X = np.vstack([np.array([feat[m, clean][i] for i in train]) for m in models])
    classifier = make_classifier(cfg["name"]).fit(X, np.tile(y[train], len(models)))      # the one classifier

    rows = []
    for m in models:
        for s in usable:
            predicted = classifier.predict(np.array([feat[m, s["scenario"]][i] for i in test]))
            for i, p in zip(test, predicted):
                rows.append({"model": m, **{k: s[k] for k in ("scenario", "test", "level", "level_index", "order", "seed")},
                             "sign": y[i], "video": vids[i]["video"], "predicted": p, "correct": bool(p == y[i])})
    print(f"{cfg['name']}: 1 classifier trained on {len(train)} clean videos x {len(models)} models, tested on "
          f"{len(test)} held-out videos in {len(usable)} scenarios x {len(models)} models, {len(set(y))} signs "
          f"(chance = {1 / len(set(y)):.2f})")
    return pd.DataFrame(rows), split, skipped


def summarize(predictions, limit):
    """Accuracy per model and scenario, its change against that model's own original accuracy, and per test the
    most degraded level that still keeps `limit` of the original accuracy."""
    df = predictions.copy()
    df["level"] = df["level"].astype(str)
    summary = (df.groupby(KEYS, dropna=False)["correct"].mean().rename("accuracy").reset_index()
                 .sort_values(["model", "level_index", "test", "order", "seed"]))
    base = summary[summary.test == "original"].set_index("model")["accuracy"]
    summary["accuracy_delta"] = summary.accuracy - summary.model.map(base)
    summary["accuracy_retained"] = summary.accuracy / summary.model.map(base).replace(0, np.nan)

    curves = (summary[summary.test != "original"]
              .groupby(["model", "test", "order", "level_index", "level"], dropna=False)
              [["accuracy", "accuracy_retained"]].mean().reset_index())
    rows = []
    for (model, test, order), g in curves.groupby(["model", "test", "order"], dropna=False):
        g = g.sort_values("level_index")
        ok = [bool(r >= limit) for r in g.accuracy_retained]
        bad = ok.index(False) if False in ok else None
        rows.append({"model": model, "test": test, "order": order, "limit": f"accuracy >= {limit:.0%} of original",
                     "original_accuracy": round(base[model], 3),
                     "last_ok_level": g.level.iloc[-1] if bad is None else (g.level.iloc[bad - 1] if bad else "none"),
                     "first_bad_level": "none" if bad is None else g.level.iloc[bad],
                     "accuracy_there": None if bad is None else round(g.accuracy.iloc[bad], 3)})
    return summary, pd.DataFrame(rows)
