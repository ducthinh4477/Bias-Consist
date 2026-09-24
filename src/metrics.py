import re
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import wasserstein_distance
from sklearn import metrics as M


def ovr_roc(labels: np.ndarray, probs: np.ndarray):
    """
    Calculate the One-vs-Rest (OvR) Receiver Operating Characteristic (ROC) and Area Under the ROC Curve (AUROC) for each class.

    Parameters:
    labels (np.ndarray): Array of true class labels. Shape should be (n_samples,).
    probs (np.ndarray): Array of predicted probabilities for each class. Shape should be (n_samples, n_classes).

    Returns:
    tuple: A tuple containing:
        - aurocs (list): List of AUROC values for each class.
        - fprs (list): List of false positive rates for each class.
        - tprs (list): List of true positive rates for each class.
        - ths (list): List of thresholds for each class.
        - ovr_macro_auroc (float): Macro-averaged AUROC for the OvR setting.
    """
    num_classes = probs.shape[1]
    labels_one_hot = np.eye(num_classes)[labels]
    fprs, tprs, ths = [], [], []

    # Why OvR with macro avg: https://chatgpt.com/share/677e448d-5bc0-8006-b9b5-081427b02857
    ovr_macro_auroc = M.roc_auc_score(labels_one_hot, probs, multi_class="ovr", average="macro")

    # Calculate OvR ROC and AUROC for each class
    for i in range(num_classes):
        fpr_class, tpr_class, ths_class = M.roc_curve(labels_one_hot[:, i], probs[:, i])
        ths_class = np.nan_to_num(ths_class, posinf=1.0)  # replace inf with max value
        ths_class = np.concatenate(([1], ths_class, [0]))  # add 0 and 1 thresholds
        fpr_class = np.concatenate(([0], fpr_class, [1]))  # add 0 and 1 fpr
        tpr_class = np.concatenate(([0], tpr_class, [1]))  # add 0 and 1 tpr
        fprs.append(fpr_class)
        tprs.append(tpr_class)
        ths.append(ths_class)

    return fprs, tprs, ths, ovr_macro_auroc


def ovr_prc(labels: np.ndarray, probs: np.ndarray):
    """
    Calculate the One-vs-Rest (OvR) Precision-Recall Curve (PRC) and the mean Average Precision (mAP) for a multi-class classification problem.

    Args:
        labels (np.ndarray): Array of true class labels with shape (n_samples,).
        probs (np.ndarray): Array of predicted probabilities with shape (n_samples, n_classes).

    Returns:
        tuple: A tuple containing:
            - precs (list of np.ndarray): List of precision values for each class.
            - recs (list of np.ndarray): List of recall values for each class.
            - ths (list of np.ndarray): List of threshold values for each class.
            - ovr_macro_ap (float): The mean Average Precision (mAP) score.
    """
    num_classes = probs.shape[1]
    labels_one_hot = np.eye(num_classes)[labels]
    precs, recs, ths = [], [], []

    # The same as mAP (mean Average Precision)
    ovr_macro_ap = M.average_precision_score(labels_one_hot, probs, average="macro")

    # Calculate OvR PRC for each class
    for i in range(num_classes):
        prec_class, rec_class, ths_class = M.precision_recall_curve(labels_one_hot[:, i], probs[:, i])
        ths_class = np.nan_to_num(ths_class, posinf=1.0)  # replace inf with max value
        ths_class = np.concatenate(([1], ths_class, [0]))  # add 0 and 1 thresholds
        prec_class = np.concatenate(([0], prec_class, [1]))  # add 0 and 1 precision
        rec_class = np.concatenate(([1], rec_class, [0]))  # add 0 and 1 recall
        precs.append(prec_class)
        recs.append(rec_class)
        ths.append(ths_class)

    return precs, recs, ths, ovr_macro_ap


def calculate_eer(y_true: np.ndarray, y_score: np.ndarray, return_threshold: bool = False):
    """
    Returns the equal error rate (EER) and the threshold at which EER occurs
    for a binary classifier output.

    Args:
        y_true (np.ndarray): True binary labels.
        y_score (np.ndarray): Target scores, can either be probability estimates of the positive class,
                              confidence values, or non-thresholded measure of decisions.
                              Assumes shape (n_samples, 2) where column 1 is the positive class score.

    Returns:
        tuple: A tuple containing:
            - eer (float): The Equal Error Rate.
            - threshold (float): The threshold at which EER occurs. Returns NaN if EER calculation fails.
    """
    y_score = np.asarray(y_score)
    if y_score.ndim == 1:
        score_pos = y_score
    elif y_score.ndim == 2 and y_score.shape[1] >= 2:
        score_pos = y_score[:, 1]
    else:
        score_pos = y_score.ravel()
    fpr, tpr, thresholds = M.roc_curve(y_true, score_pos, pos_label=1)
    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except ValueError:
        eer = np.nan

    if return_threshold:
        return eer, float(interp1d(fpr, thresholds)(eer))

    return eer


def calculate_tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_targets: list = [0.01, 0.05]):
    """
    Calculate True Positive Rate (TPR) at specified False Positive Rate (FPR) levels for binary classification.

    Args:
        y_true (np.ndarray): True binary labels (0 or 1).
        y_score (np.ndarray): Predicted probabilities or scores, shape (n_samples, 2), where column 1 is for positive class.
        fpr_targets (list): List of FPR targets (e.g., [0.01, 0.05] for 1% and 5%).

    Returns:
        list: List of TPR values corresponding to the specified FPR targets. If a target FPR is out of range, NaN is returned for that target.
    """
    fpr, tpr, _ = M.roc_curve(y_true, y_score[:, 1], pos_label=1)

    results = []
    for target in fpr_targets:
        if target < fpr.min() or target > fpr.max():
            results.append(np.nan)
        else:
            results.append(np.interp(target, fpr, tpr))

    return results


def compute_wasserstein1_metrics(probs: np.ndarray, labels: np.ndarray):
    is_real = labels == 0
    is_fake = labels == 1

    if is_real.any() and is_fake.any():
        #! Compute Wasserstein-1 distance for inter-class separability
        # These W1(u, v) reflect how well the model separates the two classes
        # u ~ P(p(y=0|x) | y=0)
        # v ~ P(p(y=0|x) | y=1)
        W1_sep_real = wasserstein_distance(probs[is_real, 0], probs[is_fake, 0])

        # u ~ P(p(y=1|x) | y=0)
        # v ~ P(p(y=1|x) | y=1)
        W1_sep_fake = wasserstein_distance(probs[is_real, 1], probs[is_fake, 1])

        #! Compute Wasserstein-1 distance for intra-sample confidence margin
        # These W1(u, v) reflect how confident the model is about its predictions
        # u ∼ P(p(y=0∣x) ∣ y=0)
        # v ∼ P(p(y=1∣x) ∣ y=0)
        W1_conf_real = wasserstein_distance(probs[is_real, 0], probs[is_real, 1])

        # u ∼ P(p(y=0∣x) ∣ y=1)
        # v ∼ P(p(y=1∣x) ∣ y=1)
        W1_conf_fake = wasserstein_distance(probs[is_fake, 0], probs[is_fake, 1])

        return W1_sep_real, W1_sep_fake, W1_conf_real, W1_conf_fake

    return -1, -1, -1, -1


def infer_label_from_path(path_or_name: str) -> Optional[int]:
    """
    Infer ground-truth binary label (0 = Real, 1 = Fake) from file path or file name.
    
    Args:
        path_or_name: File path or file name string.

    Returns:
        0 for real, 1 for fake, or None if unknown.
    """
    lower = str(path_or_name).lower().replace("\\", "/")
    parts = lower.split("/")

    # Check directory path segments
    for p in parts[:-1]:
        if any(term in p for term in ("real", "youtube-real", "celeb-real", "actors", "original", "pristine")):
            return 0
        if any(term in p for term in ("df", "fake", "deepfake", "f2f", "fs", "nt", "manipulated", "celeb-synthesis", "face2face", "faceswap", "neuraltextures")):
            return 1

    filename = parts[-1]
    stem = filename.rsplit(".", 1)[0]

    # Keyword check in filename
    if any(k in stem for k in ("real", "original", "pristine")):
        return 0
    if any(k in stem for k in ("fake", "deepfake", "manipulated", "synth")):
        return 1

    # DFD / FaceForensics++ naming patterns:
    # Pattern 1: Double actor ID separated by underscore (e.g. 01_02__..., 03_05__, 000_003) -> FAKE
    if re.search(r"\b\d{2,3}_\d{2,3}__", stem) or re.match(r"^\d{2,3}_\d{2,3}\b", stem):
        return 1
    # Pattern 2: Single actor ID followed by double underscore (e.g. 01__..., 02__...) -> REAL
    if re.search(r"\b\d{2,3}__", stem):
        return 0

    return None


def calculate_video_auc(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list],
    pos_label: int = 1,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Area Under the ROC Curve (AUROC) at the video level.

    Args:
        y_true: Array-like of ground-truth binary labels (0 = real, 1 = fake).
        y_score: Array-like of predicted probabilities or scores for the positive class (fake).
                 If 2D with shape (N, 2), column 1 is used.
        pos_label: Label of the positive class (default: 1).

    Returns:
        tuple: (video_auc, fpr, tpr, thresholds)
            - video_auc (float): Video AUROC score in [0.0, 1.0]. Returns NaN if both classes are not present.
            - fpr (np.ndarray): False positive rates.
            - tpr (np.ndarray): True positive rates.
            - thresholds (np.ndarray): Thresholds corresponding to fpr and tpr.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    if y_score.ndim == 2:
        y_score = y_score[:, 1]
    elif y_score.ndim > 1:
        y_score = y_score.squeeze()

    # Filter out NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[valid_mask].astype(int)
    y_score = y_score[valid_mask]

    unique_labels = np.unique(y_true)
    if len(unique_labels) < 2:
        return float("nan"), np.array([]), np.array([]), np.array([])

    fpr, tpr, thresholds = M.roc_curve(y_true, y_score, pos_label=pos_label)
    video_auc = float(M.auc(fpr, tpr))
    return video_auc, fpr, tpr, thresholds


def compute_video_auc_from_frames(
    video_ids: Union[list, np.ndarray],
    frame_probs: Union[list, np.ndarray],
    frame_labels: Union[list, np.ndarray],
    reduce: Literal["mean", "median", "max"] = "mean",
) -> Dict[str, Union[float, int, str, list, np.ndarray]]:
    """
    Aggregate frame-level probabilities for each video and compute Video-level AUC and metrics.

    Args:
        video_ids: List or array of video identifiers (e.g. paths or names) corresponding to each frame.
        frame_probs: Frame-level predicted probabilities (1D fake probs, or 2D [real, fake]).
        frame_labels: Frame-level ground-truth labels (0 = real, 1 = fake).
        reduce: Aggregation function across frames: 'mean', 'median', or 'max'.

    Returns:
        dict: A dictionary containing:
            - 'video_auc': float, video-level AUROC score (or NaN)
            - 'video_eer': float, video-level Equal Error Rate (or NaN)
            - 'num_videos': int, total number of unique videos
            - 'num_real': int, count of real videos
            - 'num_fake': int, count of fake videos
            - 'reduce': str, aggregation method used
            - 'video_ids': list of video IDs
            - 'video_labels': np.ndarray of binary labels per video
            - 'video_scores': np.ndarray of aggregated fake scores per video
            - 'fpr': np.ndarray, False positive rates
            - 'tpr': np.ndarray, True positive rates
    """
    video_ids = list(video_ids)
    frame_probs = np.asarray(frame_probs, dtype=float)
    frame_labels = np.asarray(frame_labels, dtype=int)

    if frame_probs.ndim == 2:
        frame_fake_probs = frame_probs[:, 1]
    else:
        frame_fake_probs = frame_probs.squeeze()

    # Group frames by video
    v2scores: Dict[str, list] = {}
    v2labels: Dict[str, int] = {}
    v2frames: Dict[str, int] = {}

    for vid, score, label in zip(video_ids, frame_fake_probs, frame_labels):
        vid_str = str(vid)
        if vid_str not in v2scores:
            v2scores[vid_str] = []
            v2labels[vid_str] = int(label)
            v2frames[vid_str] = 0
        v2scores[vid_str].append(float(score))
        v2frames[vid_str] += 1

    unique_vids = list(v2scores.keys())
    agg_scores = []
    agg_labels = []

    for vid in unique_vids:
        scores = np.array(v2scores[vid])
        if reduce == "mean":
            agg_scores.append(float(np.mean(scores)))
        elif reduce == "median":
            agg_scores.append(float(np.median(scores)))
        elif reduce == "max":
            agg_scores.append(float(np.max(scores)))
        else:
            raise ValueError(f"Unsupported reduce method: {reduce}. Choose 'mean', 'median', or 'max'.")
        agg_labels.append(v2labels[vid])

    arr_scores = np.array(agg_scores, dtype=float)
    arr_labels = np.array(agg_labels, dtype=int)

    video_auc, fpr, tpr, ths = calculate_video_auc(arr_labels, arr_scores)

    # Calculate video EER
    if not np.isnan(video_auc) and len(np.unique(arr_labels)) == 2:
        y_score_2d = np.column_stack([1.0 - arr_scores, arr_scores])
        video_eer = calculate_eer(arr_labels, y_score_2d)
    else:
        video_eer = float("nan")

    return {
        "video_auc": video_auc,
        "video_eer": video_eer,
        "num_videos": len(unique_vids),
        "num_real": int((arr_labels == 0).sum()),
        "num_fake": int((arr_labels == 1).sum()),
        "reduce": reduce,
        "video_ids": unique_vids,
        "video_labels": arr_labels,
        "video_scores": arr_scores,
        "fpr": fpr,
        "tpr": tpr,
    }
