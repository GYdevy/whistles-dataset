"""
Pure DSP + Rule-Based Whistle Detector
No ML
Multi-match evaluation
"""

import json
import tempfile
import subprocess
import os
from dataclasses import dataclass
import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d
import soundfile as sf


# ============================================================
# CONFIG
# ============================================================

VIDEO_DIR = r"E:\Volleyballey\videos"
GT_PATH = r"E:\Volleyballey\detector_slop\whistles_all_reanchored.json"
FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"
DATASET_DIR = r"E:\Volleyballey\whistle_dataset"
SNIPPET_SEC = 0.6
FP_SAFE_MARGIN = 1.0  # seconds away from any GT whistle

ANCHOR_TOLERANCE = 0.6


@dataclass
class Config:
    sr: int = 22050
    n_fft: int = 2048
    hop: int = 128
    whistle_low: float = 3700
    whistle_high: float = 4300
    min_frames: int = 15
    max_gap_frames: int = 12


cfg = Config()


# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio_from_video(video_path, target_sr):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    subprocess.run([
        FFMPEG, "-y", "-i", video_path,
        "-ac", "1",
        "-ar", str(target_sr),
        "-vn", tmp_path
    ], stdout=subprocess.DEVNULL,
       stderr=subprocess.DEVNULL,
       check=True)

    y, _ = librosa.load(tmp_path, sr=target_sr)
    os.remove(tmp_path)
    return y


# ============================================================
# STAGE A — ROI DETECTOR
# ============================================================

def detect_active_frames(y):

    S = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop)
    mag = np.abs(S)

    freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)
    band_mask = (freqs >= cfg.whistle_low) & (freqs <= cfg.whistle_high)

    band_mag = mag[band_mask]
    band_energy = band_mag.mean(axis=0)

    flatness = librosa.feature.spectral_flatness(S=mag)[0]

    band_peak = np.max(band_mag, axis=0)
    band_mean = np.mean(band_mag, axis=0)
    sharpness = band_peak / (band_mean + 1e-8)

    # Normalize
    band_energy = (band_energy - np.median(band_energy)) / (np.std(band_energy) + 1e-8)
    sharpness = (sharpness - np.median(sharpness)) / (np.std(sharpness) + 1e-8)
    flatness = (flatness - np.median(flatness)) / (np.std(flatness) + 1e-8)

    band_energy = uniform_filter1d(band_energy, size=5)
    sharpness = uniform_filter1d(sharpness, size=5)
    flatness = uniform_filter1d(flatness, size=5)

    score = 1.0 * band_energy + 1.0 * sharpness - 1.2 * flatness

    START_TH = 0.75
    CONTINUE_TH = 0.25

    active = []
    in_whistle = False

    for i, s in enumerate(score):
        if not in_whistle:
            if s > START_TH:
                in_whistle = True
                active.append(i)
        else:
            if s > CONTINUE_TH:
                active.append(i)
            else:
                in_whistle = False

    return active


# ============================================================
# GROUPING
# ============================================================

def group_frames(active):

    if not active:
        return []

    groups = []
    cur = [active[0]]

    for f in active[1:]:
        if f - cur[-1] <= cfg.max_gap_frames:
            cur.append(f)
        else:
            groups.append(cur)
            cur = [f]

    groups.append(cur)
    return groups


def extract_candidates(groups):

    detections = []
    pad_sec = 0.35
    pad_frames = int((pad_sec * cfg.sr) / cfg.hop)

    for g in groups:
        if len(g) < cfg.min_frames:
            continue

        start_frame = max(0, g[0] - pad_frames)
        end_frame = g[-1] + pad_frames

        start_sec = start_frame * cfg.hop / cfg.sr
        end_sec = end_frame * cfg.hop / cfg.sr

        detections.append((start_sec, end_sec))

    return detections


# ============================================================
# REFINEMENT
# ============================================================

def refine_candidates(y, detections):

    refined = []

    for start, end in detections:

        s0 = int(start * cfg.sr)
        s1 = int(end * cfg.sr)
        seg = y[s0:s1]

        if len(seg) < cfg.n_fft:
            continue

        S = librosa.stft(seg, n_fft=cfg.n_fft, hop_length=cfg.hop)
        mag = np.abs(S)

        freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)
        mask = (freqs >= cfg.whistle_low) & (freqs <= cfg.whistle_high)

        if not np.any(mask):
            continue

        # Use peak energy instead of mean band energy
        band_peak = mag[mask].max(axis=0)
        band_peak = uniform_filter1d(band_peak, size=5)

        # Take robust center (top 5% strongest frames)
        th = np.percentile(band_peak, 95)
        strong_idxs = np.where(band_peak >= th)[0]

        if len(strong_idxs) > 0:
            center_frame = int(np.median(strong_idxs))
        else:
            center_frame = int(np.argmax(band_peak))

        t_peak = (s0 + center_frame * cfg.hop) / cfg.sr

        pre_sec = 1.0
        post_sec = 0.6

        new_start = max(0.0, t_peak - pre_sec)
        new_end = min(len(y) / cfg.sr, t_peak + post_sec)

        refined.append((new_start, new_end, t_peak))

    return refined


# ============================================================
# FEATURE EXTRACTION (STAGE B/C)
# ============================================================

def extract_window_features(y, start, end):

    s0 = int(start * cfg.sr)
    s1 = int(end * cfg.sr)
    seg = y[s0:s1]

    if len(seg) < cfg.n_fft:
        return None

    # STFT
    S = librosa.stft(seg, n_fft=cfg.n_fft, hop_length=cfg.hop)
    mag = np.abs(S)

    freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)
    band_mask = (freqs >= cfg.whistle_low) & (freqs <= cfg.whistle_high)

    if not np.any(band_mask):
        return None

    S_w = mag[band_mask]
    freqs_w = freqs[band_mask]

    # ------------------------------------------------
    #  Narrow-band concentration ratio
    # ------------------------------------------------
    mean_spectrum = S_w.mean(axis=1)
    peak_idx = np.argmax(mean_spectrum)
    center_freq = freqs_w[peak_idx]

    narrow_mask = (freqs >= center_freq - 150) & (freqs <= center_freq + 150)

    if np.any(narrow_mask):
        narrow_energy = mag[narrow_mask].mean()
        band_energy_full = mag[band_mask].mean() + 1e-8
        narrow_ratio = narrow_energy / band_energy_full
    else:
        narrow_ratio = 0.0
    # ------------------------------------------------
    # Band ratio (energy concentration)
    # ------------------------------------------------
    band_energy = S_w.mean()
    total_energy = mag.mean() + 1e-8
    band_ratio = band_energy / total_energy

    # ------------------------------------------------
    # Frequency stability (Hz)
    # ------------------------------------------------
    if S_w.shape[1] < 3:
        freq_std = np.inf
    else:
        peak_bins = np.argmax(S_w, axis=0)
        peak_freqs = freqs_w[peak_bins]
        freq_std = np.std(peak_freqs)

    # ------------------------------------------------
    # Band flatness
    # ------------------------------------------------
    flatness_band = librosa.feature.spectral_flatness(S=S_w)[0].mean()

    # ------------------------------------------------
    # Peak prominence
    # ------------------------------------------------
    band_peak = S_w.max(axis=0)
    band_mean = S_w.mean(axis=0) + 1e-8
    peak_prominence = np.mean(band_peak / band_mean)

    return {
        "band_ratio": band_ratio,
        "freq_std": freq_std,
        "flatness_band": flatness_band,
        "peak_prominence": peak_prominence,
        "band_energy": band_energy,
        "narrow_ratio": narrow_ratio,
    }

def suppress_close_centers(detections, min_gap_sec=0.7):

    detections = sorted(detections, key=lambda x: x[2])  # sort by t_peak
    filtered = []

    for d in detections:

        center = d[2]

        if not filtered:
            filtered.append(d)
            continue

        prev_center = filtered[-1][2]

        if abs(center - prev_center) > min_gap_sec:
            filtered.append(d)

    return filtered


# ============================================================
# RULE-BASED SIFTER
# ============================================================

def rule_based_sifter(detections, y, stats):

    accepted = []


    for start, end, t_peak in detections:

        feats = extract_window_features(y, start, end)
        if feats is None:
            continue

        # ---- Z SCORE NORMALIZATION ----
        z_band_ratio = (feats["band_ratio"] - stats["band_ratio"]["median"]) / stats["band_ratio"]["std"]
        z_freq_std = (feats["freq_std"] - stats["freq_std"]["median"]) / stats["freq_std"]["std"]
        z_flat = (feats["flatness_band"] - stats["flatness_band"]["median"]) / stats["flatness_band"]["std"]
        z_prom = (feats["peak_prominence"] - stats["peak_prominence"]["median"]) / stats["peak_prominence"]["std"]

        # Hard reject only extreme noise
        #if z_flat > 2.5:
            #continue

        z_narrow = (feats["narrow_ratio"] - stats["narrow_ratio"]["median"]) / stats["narrow_ratio"]["std"]

        score = (
                1.0 * z_band_ratio +
                1.3 * z_prom +
                1.0 * z_narrow -
                0.8 * z_flat
        )

        if score > -0.2:
            accepted.append((start, end, t_peak))
            continue

        # rescue tonal events
        if (
                feats["band_ratio"] > stats["band_ratio"]["median"] * 0.6 and
                feats["peak_prominence"] > stats["peak_prominence"]["median"] * 0.6
        ):
            accepted.append((start, end, t_peak))


    accepted = suppress_close_centers(accepted, min_gap_sec=0.6)

    return accepted

def compute_match_stats(detections, y):

    all_feats = []

    for start, end, t_peak in detections:
        feats = extract_window_features(y, start, end)
        if feats:
            all_feats.append(feats)

    if not all_feats:
        return None

    stats = {}

    keys = all_feats[0].keys()

    for k in keys:
        arr = np.array([f[k] for f in all_feats])
        stats[k] = {
            "median": np.median(arr),
            "std": np.std(arr) + 1e-8
        }

    return stats
# ============================================================
# FEATURE DISTRIBUTION DEBUG
# ============================================================

def analyze_feature_distributions(detections, y, gt):

    tp_features = []
    fp_features = []

    for start, end, t_peak in detections:

        feats = extract_window_features(y, start, end)
        if feats is None:
            continue

        is_tp = False
        for g in gt:
            anchor = g["t_anchor"]
            if (start - ANCHOR_TOLERANCE) <= anchor <= (end + ANCHOR_TOLERANCE):
                is_tp = True
                break

        if is_tp:
            tp_features.append(feats)
        else:
            fp_features.append(feats)

    print("\n==== FEATURE STATS ====")

    def summarize(name):
        tp_vals = np.array([f[name] for f in tp_features])
        fp_vals = np.array([f[name] for f in fp_features])

        print(f"\nFeature: {name}")
        print(" TP median:", np.median(tp_vals))
        print(" FP median:", np.median(fp_vals))
        print("TP 10%:", np.percentile(tp_vals, 10))
        print("TP 50%:", np.percentile(tp_vals, 50))
        print("TP 90%:", np.percentile(tp_vals, 90))

        print("FP 10%:", np.percentile(fp_vals, 10))
        print("FP 50%:", np.percentile(fp_vals, 50))
        print("FP 90%:", np.percentile(fp_vals, 90))

    summarize("band_ratio")
    summarize("freq_std")
    summarize("flatness_band")
    summarize("peak_prominence")
    summarize("band_energy")


# ============================================================
# EVALUATION
# ============================================================

def evaluate_candidate_hits(detections, gt):

    matched = 0
    offsets = []
    missed = []

    for g in gt:

        anchor = g["t_anchor"]
        found = False

        for start, end, t_peak in detections:
            if (start - ANCHOR_TOLERANCE) <= anchor <= (end + ANCHOR_TOLERANCE):
                offsets.append(t_peak - anchor)
                matched += 1
                found = True
                break

        if not found:
            missed.append(anchor)

    recall = matched / len(gt)
    return recall, offsets, missed


# ============================================================
# MATCH EVALUATION
# ============================================================

def evaluate_match(match_id, all_gt):

    print(f"\n==================== {match_id} ====================")

    video_path = os.path.join(VIDEO_DIR, f"{match_id}.mp4")
    y = load_audio_from_video(video_path, cfg.sr)

    active = detect_active_frames(y)
    groups = group_frames(active)

    stage1 = extract_candidates(groups)
    refined = refine_candidates(y, stage1)

    # sort refined by band_energy
    refined_sorted = sorted(
        refined,
        key=lambda d: extract_window_features(y, d[0], d[1])["band_energy"],
        reverse=True
    )

    top_k = int(0.3 * len(refined_sorted))
    refined_top = refined_sorted[:top_k]

    stats = compute_match_stats(refined_top, y)
    accepted = rule_based_sifter(refined, y, stats)

    gt_filtered = [g for g in all_gt if g["match_id"] == match_id]

    analyze_feature_distributions(refined, y, gt_filtered)

    recall, offsets, missed = evaluate_candidate_hits(accepted, gt_filtered)
    generate_dataset_snippets(match_id, y, accepted, gt_filtered)
    explosion = len(accepted) / len(gt_filtered)

    print("GT:", len(gt_filtered))
    print("Stage1:", len(stage1))
    print("Accepted:", len(accepted))
    print("Recall:", round(recall, 3))
    print("Explosion:", round(explosion, 2))
    print("Missed:", len(missed))

    if offsets:
        offsets = np.array(offsets)
        print("Median abs offset:", round(np.median(np.abs(offsets)), 3))
        print("90th percentile:", round(np.percentile(np.abs(offsets), 90), 3))

    return {
        "match": match_id,
        "recall": recall,
        "explosion": explosion
    }

def generate_dataset_snippets(match_id, y, detections, gt):

    pos_dir = os.path.join(DATASET_DIR, "pos")
    neg_dir = os.path.join(DATASET_DIR, "neg")

    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    half_len = SNIPPET_SEC / 2
    sr = cfg.sr

    gt_anchors = [g["t_anchor"] for g in gt]

    # ------------------------------------
    # 🔥 Count existing files for match
    # ------------------------------------
    existing_pos = [
        f for f in os.listdir(pos_dir)
        if f.startswith(f"{match_id}_pos_")
    ]
    existing_neg = [
        f for f in os.listdir(neg_dir)
        if f.startswith(f"{match_id}_neg_")
    ]

    pos_count = len(existing_pos)
    neg_count = len(existing_neg)

    print(f"{match_id}: continuing from pos={pos_count}, neg={neg_count}")

    # ------------------------------------
    # Generate new snippets from DSP detections
    # ------------------------------------
    for start, end, t_peak in detections:

        s0 = int((t_peak - half_len) * sr)
        s1 = int((t_peak + half_len) * sr)

        if s0 < 0 or s1 > len(y):
            continue

        snippet = y[s0:s1]

        # Check TP
        is_tp = any(abs(t_peak - anchor) <= ANCHOR_TOLERANCE for anchor in gt_anchors)

        if is_tp:
            out_path = os.path.join(
                pos_dir,
                f"{match_id}_pos_{pos_count}.wav"
            )
            sf.write(out_path, snippet, sr)
            pos_count += 1

        else:
            # skip FPs too close to GT
            if any(abs(t_peak - anchor) <= FP_SAFE_MARGIN for anchor in gt_anchors):
                continue

            out_path = os.path.join(
                neg_dir,
                f"{match_id}_neg_{neg_count}.wav"
            )
            sf.write(out_path, snippet, sr)
            neg_count += 1

    print(f"{match_id}: saved total pos={pos_count}, neg={neg_count}")
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    with open(GT_PATH) as f:
        all_gt = json.load(f)

    match_ids = sorted(set(g["match_id"] for g in all_gt))

    results = []

    for m in match_ids:
        r = evaluate_match(m, all_gt)
        results.append(r)

    print("\n================ SUMMARY ================")
    for r in results:
        print(r)