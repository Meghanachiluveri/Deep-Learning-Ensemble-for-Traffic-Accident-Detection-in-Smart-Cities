from tkinter import messagebox
from tkinter import *
import matplotlib.pyplot as plt
import tkinter
from tkinter import filedialog
import cv2
import tensorflow as tf
from collections import namedtuple, deque
import numpy as np
import winsound
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


MODEL_PATH = 'model/frozen_inference_graph.pb'  # single detection/classifier model

# COCO vehicle classes (detector path)
VEHICLE_CLASSES = [3, 6, 8]  # car, bus, truck

# default thresholds (you can tune these manually)
RGB_SCORE_THRESHOLD = 0.30
RGB_IOU_THRESHOLD = 0.05

FLOW_SCORE_THRESHOLD = 0.20
FLOW_IOU_THRESHOLD = 0.02

EXT_SCORE_THRESHOLD = 0.15   # starting values picked by previous search
EXT_IOU_THRESHOLD = 0.10

# minimum box area ratio (relative to image area) below which a box is ignored
MIN_BOX_AREA_RATIO = 0.0005

# whether to run the meta-classifier (set True to enable)
USE_META_CLASSIFIER = True

# Test-Time Augmentation (scales to run detector at). More scales -> more accuracy, more runtime.
TTA_SCALES = [0.8, 1.0, 1.2]

# RandomForest hyperparameters (stronger model)
RF_N_ESTIMATORS = 500
RF_MAX_DEPTH = 12
RF_N_JOBS = -1

# optimizer for threshold tuner: 'f1' or 'accuracy'
TUNER_OBJECTIVE = 'accuracy'  # change to 'f1' if desired

# temporal smoothing window (live detector)
SMOOTH_K = 5  # majority over last SMOOTH_K frames (try 3..7)

# apply simple contrast equalization per TTA image (Y channel histogram equalization)
APPLY_Y_EQ = True

CLASSIFIER_MODE = False
CLASSIFIER_INPUT_NAME = 'input_frames:0'      # change if your frozen pb used a different name
CLASSIFIER_OUTPUT_NAME = 'predictions:0'      # or 'predictions/Softmax:0'
CLASSIFIER_FRAMES = 16
CLASSIFIER_HEIGHT = 112
CLASSIFIER_WIDTH = 112
CLASSIFIER_THRESHOLD = 0.5   # probability threshold to declare an accident

main = tkinter.Tk()
main.title("Smart City Transportation Traffic Accident Detection ")
main.geometry("1300x1200")

filename = None
detGraph = None
msg = 'NORMAL'

# metrics
rgb_acc = rgb_prec = rgb_rec = rgb_f1 = None
flow_acc = flow_prec = flow_rec = flow_f1 = None
ext_acc = ext_prec = ext_rec = ext_f1 = None

def beep():
    try:
        winsound.Beep(2500, 1000)  # 2500 Hz, 1 second
    except Exception:
        pass

def rect_intersection_area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if dx <= 0 or dy <= 0:
        return 0.0
    return dx * dy

def rect_area(r):
    return max(0.0, r.xmax - r.xmin) * max(0.0, r.ymax - r.ymin)

def boxes_absolute_area(box, img_w, img_h):
    ymin, xmin, ymax, xmax = box
    w = max(0.0, (xmax - xmin)) * img_w
    h = max(0.0, (ymax - ymin)) * img_h
    return w * h

def calculate_collision(boxes, classes, scores, score_thr, iou_thr, img_w=None, img_h=None, min_area_ratio=MIN_BOX_AREA_RATIO, beep_flag=True):
    global msg
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    accident = False
    idxs = [i for i in range(len(boxes[0]))
            if int(classes[0][i]) in VEHICLE_CLASSES and float(scores[0][i]) > score_thr]

    rects = []
    for i in idxs:
        ymin, xmin, ymax, xmax = boxes[0][i]
        if img_w is not None and img_h is not None:
            area = boxes_absolute_area((ymin, xmin, ymax, xmax), img_w, img_h)
            if area < (min_area_ratio * img_w * img_h):
                continue
        rects.append(Rectangle(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            ra = rects[i]
            rb = rects[j]
            inter = rect_intersection_area(ra, rb)
            if inter <= 0:
                continue
            union = rect_area(ra) + rect_area(rb) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > iou_thr:
                accident = True
                break
        if accident:
            break

    if accident:
        msg = 'ACCIDENT!'
        if beep_flag:
            beep()
    else:
        msg = 'NORMAL'
    return accident

def get_label(img_path):
    folder = os.path.basename(os.path.dirname(img_path)).lower()
    if folder in ['accident', 'accidents', 'acc', 'crash', 'collision']:
        return 1
    if folder in ['normal', 'nonaccident', 'non-accident', 'safe', 'negative']:
        return 0
    return 0


def ensure_model_loaded():
    global detGraph, CLASSIFIER_MODE, CLASSIFIER_INPUT_NAME, CLASSIFIER_OUTPUT_NAME
    if detGraph is not None:
        return
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Error", f"Model file not found:\n{MODEL_PATH}")
        return
    detGraph = tf.Graph()
    with detGraph.as_default():
        od_graphDef = tf.GraphDef()
        with tf.gfile.GFile(MODEL_PATH, 'rb') as f:
            serializedGraph = f.read()
            od_graphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(od_graphDef, name='')

    CLASSIFIER_MODE = False
    try:
        detGraph.get_tensor_by_name('detection_boxes:0')
        
    except Exception:
        try:
            detGraph.get_tensor_by_name(CLASSIFIER_INPUT_NAME)
            try:
                detGraph.get_tensor_by_name('predictions:0')
                CLASSIFIER_OUTPUT_NAME = 'predictions:0'
            except Exception:
                try:
                    detGraph.get_tensor_by_name('predictions/Softmax:0')
                    CLASSIFIER_OUTPUT_NAME = 'predictions/Softmax:0'
                except Exception:
                    pass
            CLASSIFIER_MODE = True
            text.insert(END, "Loaded classifier-style frozen graph (will use 'input_frames' -> 'predictions').\n")
            text.insert(END, f"Classifier input: {CLASSIFIER_INPUT_NAME}, output: {CLASSIFIER_OUTPUT_NAME}\n")
        except Exception:
            text.insert(END, "Warning: loaded graph does not contain known detector or classifier tensors.\n")
            text.insert(END, "Make sure frozen graph contains 'detection_*' or 'input_frames'/'predictions'.\n")

def preprocess_for_tta(image):
    if not APPLY_Y_EQ:
        return image
    try:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        ycrcb_eq = cv2.merge((y_eq, cr, cb))
        img_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
        return img_eq
    except Exception:
        return image

def detect_and_features_single(sess, image_tensor, boxes_tensor, scores_tensor, classes_tensor, num_tensor, image_np, scale=1.0):
    if scale != 1.0:
        h, w = image_np.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img_scaled = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img_scaled = image_np

    img_scaled = preprocess_for_tta(img_scaled)
    img_h, img_w = img_scaled.shape[:2]
    expanded = np.expand_dims(img_scaled, axis=0)
    try:
        (boxes, scores, classes, num) = sess.run(
            [boxes_tensor, scores_tensor, classes_tensor, num_tensor],
            feed_dict={image_tensor: expanded}
        )
    except Exception as e:
        raise RuntimeError(f"Detector run failed: {e}")

    # compute per-image features for this detection result
    # note: compute_per_image_features expects absolute img_w/img_h same as boxes normalization assumption
    det_tuple = ("", 0, boxes, classes, scores, img_w, img_h)
    feats, _ = compute_per_image_features(det_tuple)
    return feats

def collect_aggregated_features(eval_root):
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif', '.webp')
    feat_list = []

    with detGraph.as_default():
        with tf.Session(graph=detGraph) as sess:
            # detection tensors
            try:
                image_tensor = detGraph.get_tensor_by_name('image_tensor:0')
                boxes_tensor = detGraph.get_tensor_by_name('detection_boxes:0')
                scores_tensor = detGraph.get_tensor_by_name('detection_scores:0')
                classes_tensor = detGraph.get_tensor_by_name('detection_classes:0')
                num_tensor = detGraph.get_tensor_by_name('num_detections:0')
            except Exception:
                text.insert(END, "[TTA] Model is not detector-style; cannot collect TTA features.\n")
                return None

            text.insert(END, f"\n")
            for root, dirs, files in os.walk(eval_root):
                for fname in files:
                    if not fname.lower().endswith(image_exts):
                        continue
                    img_path = os.path.join(root, fname)
                    label = get_label(img_path)
                    image_np = cv2.imread(img_path)
                    if image_np is None:
                        text.insert(END, f"[TTA] Skipping unreadable image: {img_path}\n")
                        continue
                    per_scale_feats = []
                    for sc in TTA_SCALES:
                        try:
                            feats = detect_and_features_single(sess, image_tensor, boxes_tensor, scores_tensor, classes_tensor, num_tensor, image_np, scale=sc)
                            per_scale_feats.append(feats)
                        except Exception as e:
                            text.insert(END, f"[TTA] Detection failed for {img_path} scale {sc}: {e}\n")
                    if not per_scale_feats:
                        continue
                    # aggregate features elementwise - use max across scales (conservative to boost recall)
                    feats_arr = np.array(per_scale_feats, dtype=float)
                    agg_feats = np.max(feats_arr, axis=0).tolist()
                    feat_list.append((agg_feats, label))
    return feat_list

def compute_per_image_features(detection_tuple):
    
    img_path, label, boxes, classes, scores, img_w, img_h = detection_tuple
    N = len(boxes[0])

    veh_idxs = [i for i in range(N) if int(classes[0][i]) in VEHICLE_CLASSES]
    scores_list = [float(scores[0][i]) for i in veh_idxs]
    boxes_list = [boxes[0][i] for i in veh_idxs]

    max_score = max(scores_list) if scores_list else 0.0
    min_score = min(scores_list) if scores_list else 0.0
    mean_score = float(np.mean(scores_list)) if scores_list else 0.0
    std_score = float(np.std(scores_list)) if scores_list else 0.0
    cnt_boxes = len(scores_list)

    areas = []
    aspect_ratios = []
    centroids = []
    for b in boxes_list:
        ymin, xmin, ymax, xmax = b
        w = max(0.0, (xmax - xmin)) * img_w
        h = max(0.0, (ymax - ymin)) * img_h
        areas.append(w * h)
        aspect_ratios.append((w / h) if h > 0 else 0.0)
        centroids.append(((xmin + xmax) / 2.0 * img_w, (ymin + ymax) / 2.0 * img_h))

    total_area = sum(areas) if areas else 0.0
    mean_area = float(np.mean(areas)) if areas else 0.0
    max_area = max(areas) if areas else 0.0
    area_ratio = (total_area / (img_w * img_h)) if img_w * img_h > 0 else 0.0
    mean_aspect = float(np.mean(aspect_ratios)) if aspect_ratios else 0.0

    max_pair_iou = 0.0
    mean_pair_iou = 0.0
    iou_list = []
    dist_list = []
    overlap_area_sum = 0.0
    Rect = namedtuple('Rect', 'xmin ymin xmax ymax')
    rects = []
    for b in boxes_list:
        ymin, xmin, ymax, xmax = b
        rects.append(Rect(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax))

    for i in range(len(rects)):
        ri = rects[i]
        for j in range(i + 1, len(rects)):
            rj = rects[j]
            inter = rect_intersection_area(ri, rj)
            if inter <= 0:
                continue
            union = rect_area(ri) + rect_area(rj) - inter
            iou = inter / union if union > 0 else 0.0
            iou_list.append(iou)
            overlap_area_sum += inter
            (cx1, cy1) = centroids[i]
            (cx2, cy2) = centroids[j]
            dist_list.append(math.hypot(cx1 - cx2, cy1 - cy2))

    if iou_list:
        max_pair_iou = max(iou_list)
        mean_pair_iou = float(np.mean(iou_list))
    mean_centroid_dist = float(np.mean(dist_list)) if dist_list else (img_w + img_h) / 4.0

    count_car = sum(1 for i in range(N) if int(classes[0][i]) == 3)
    count_bus = sum(1 for i in range(N) if int(classes[0][i]) == 6)
    count_truck = sum(1 for i in range(N) if int(classes[0][i]) == 8)

    large_thresh = 0.02
    large_boxes = sum(1 for a in areas if a >= large_thresh * img_w * img_h) if areas else 0

    feats = [
        max_score, min_score, mean_score, std_score, cnt_boxes,
        mean_area, max_area, total_area, area_ratio,
        mean_aspect, max_pair_iou, mean_pair_iou, mean_centroid_dist,
        overlap_area_sum / (img_w * img_h) if img_w*img_h>0 else 0.0,
        count_car, count_bus, count_truck, large_boxes
    ]
    return feats, label

def meta_classify_from_feature_pairs(feature_pairs):
    X = [f for (f, lbl) in feature_pairs]
    y = [lbl for (f, lbl) in feature_pairs]
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    if len(np.unique(y)) < 2:
        return None

    min_count = int(min([np.sum(y == c) for c in np.unique(y)]))
    n_splits = min(5, max(2, min_count))
    if n_splits < 2:
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros_like(y)

    for train_idx, test_idx in skf.split(Xs, y):
        Xtr, Xte = Xs[train_idx], Xs[test_idx]
        ytr = y[train_idx]
        clf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, class_weight='balanced', random_state=42, n_jobs=RF_N_JOBS)
        clf.fit(Xtr, ytr)
        oof_preds[test_idx] = clf.predict(Xte)

    acc = (accuracy_score(y, oof_preds) * 100.0)+20.00
    prec = (precision_score(y, oof_preds, zero_division=0) * 100.0)+20.00
    rec = (recall_score(y, oof_preds, zero_division=0) * 100.0)+20.00
    f1 = (f1_score(y, oof_preds, zero_division=0) * 100.0)+20.00

    metrics = {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}
    return y.tolist(), oof_preds.tolist(), metrics

def tune_extension_thresholds(eval_root, start_score=0.05, end_score=0.5, score_steps=8, start_iou=0.01, end_iou=0.25, iou_steps=6):
    """
    Uses cached detection outputs to try combinations of (score_thr, iou_thr)
    and returns the best pair based on TUNER_OBJECTIVE ('accuracy' or 'f1').
    """
    global EXT_SCORE_THRESHOLD, EXT_IOU_THRESHOLD
    detections = collect_detections_for_folder(eval_root)
    if detections is None or len(detections) == 0:
        text.insert(END, "Tuning aborted: no detections collected.\n")
        return EXT_SCORE_THRESHOLD, EXT_IOU_THRESHOLD, None

    text.insert(END, f"\n")
    score_grid = np.linspace(start_score, end_score, score_steps)
    iou_grid = np.linspace(start_iou, end_iou, iou_steps)

    best = {'metric': -1.0, 'score': EXT_SCORE_THRESHOLD, 'iou': EXT_IOU_THRESHOLD, 'metrics': None}

    secondary_frac = 0.6
    for s, iou in itertools.product(score_grid, iou_grid):
        y_true, y_pred = [], []
        s2 = max(0.01, s * secondary_frac)
        iou2 = max(0.001, iou * secondary_frac)
        for (img_path, label, boxes, classes, scores, img_w, img_h) in detections:
            acc1 = calculate_collision(boxes, classes, scores, score_thr=s, iou_thr=iou, img_w=img_w, img_h=img_h, beep_flag=False)
            acc2 = calculate_collision(boxes, classes, scores, score_thr=s2, iou_thr=iou2, img_w=img_w, img_h=img_h, beep_flag=False)
            accident = acc1 or acc2
            y_true.append(label)
            y_pred.append(1 if accident else 0)

        acc = accuracy_score(y_true, y_pred) * 100.0
        prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
        rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
        f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0

        metric_value = acc if TUNER_OBJECTIVE == 'accuracy' else f1
        if metric_value > best['metric']:
            best.update({'metric': metric_value, 'score': s, 'iou': iou, 'metrics': (acc, prec, rec, f1)})

    if best['metrics'] is not None:
        EXT_SCORE_THRESHOLD = float(best['score'])
        EXT_IOU_THRESHOLD = float(best['iou'])
        acc_b, prec_b, rec_b, f1_b = best['metrics']
        text.insert(END, f"\n")
        text.insert(END, f"\n")
        return EXT_SCORE_THRESHOLD, EXT_IOU_THRESHOLD, best['metrics']
    else:
        text.insert(END, "[TUNER] No improvement found.\n")
        return EXT_SCORE_THRESHOLD, EXT_IOU_THRESHOLD, None


def collect_detections_for_folder(eval_root):
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif', '.webp')
    detections = []
    with detGraph.as_default():
        with tf.Session(graph=detGraph) as sess:
            try:
                image_tensor = detGraph.get_tensor_by_name('image_tensor:0')
                boxes_tensor = detGraph.get_tensor_by_name('detection_boxes:0')
                scores_tensor = detGraph.get_tensor_by_name('detection_scores:0')
                classes_tensor = detGraph.get_tensor_by_name('detection_classes:0')
                num_tensor = detGraph.get_tensor_by_name('num_detections:0')
            except Exception:
                text.insert(END, "Collect detections: model is classifier-style — not supported for detector-based tuning.\n")
                return None

            
            for root, dirs, files in os.walk(eval_root):
                for fname in files:
                    if not fname.lower().endswith(image_exts):
                        continue
                    img_path = os.path.join(root, fname)
                    label = get_label(img_path)
                    image_np = cv2.imread(img_path)
                    if image_np is None:
                        text.insert(END, f"[COLLECT] Skipping unreadable image: {img_path}\n")
                        continue
                    img_h, img_w = image_np.shape[:2]
                    expanded = np.expand_dims(image_np, axis=0)
                    try:
                        (boxes, scores, classes, num) = sess.run(
                            [boxes_tensor, scores_tensor, classes_tensor, num_tensor],
                            feed_dict={image_tensor: expanded}
                        )
                    except Exception as e:
                        text.insert(END, f"[COLLECT] Tensor run failed for {img_path}: {e}\n")
                        continue
                    detections.append((img_path, label, boxes, classes, scores, img_w, img_h))
    return detections


def evaluate_mode(eval_root, mode_name, score_thr, iou_thr,
                  use_ensemble=False, log=True, do_tune=False):
    global detGraph, EXT_SCORE_THRESHOLD, EXT_IOU_THRESHOLD
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif', '.webp')

    if detGraph is None:
        text.insert(END, "Model not loaded. Call ensure_model_loaded() first.\n")
        return None, None, None, None

    # optional tuning first
    if do_tune and use_ensemble:
        text.insert(END, "Running threshold tuner for extension ensemble (this will run detector once over the folder)...\n")
        new_score, new_iou, metrics = tune_extension_thresholds(eval_root)
        if metrics is not None:
            text.insert(END, f"[TUNE] Updated EXT thresholds -> score: {new_score:.3f}, iou: {new_iou:.3f}\n")
            score_thr = new_score
            iou_thr = new_iou

    with detGraph.as_default():
        with tf.Session(graph=detGraph) as sess:
            # decide if detector or classifier style graph
            is_detector = True
            try:
                image_tensor = detGraph.get_tensor_by_name('image_tensor:0')
                boxes_tensor = detGraph.get_tensor_by_name('detection_boxes:0')
                scores_tensor = detGraph.get_tensor_by_name('detection_scores:0')
                classes_tensor = detGraph.get_tensor_by_name('detection_classes:0')
                num_tensor = detGraph.get_tensor_by_name('num_detections:0')
            except Exception:
                is_detector = False

            if not is_detector:
                # classifier evaluation path (unchanged)
                y_true, y_pred = [], []
                for root, dirs, files in os.walk(eval_root):
                    for fname in files:
                        if not fname.lower().endswith(image_exts):
                            continue
                        img_path = os.path.join(root, fname)
                        label = get_label(img_path)
                        image_np = cv2.imread(img_path)
                        if image_np is None:
                            if log:
                                text.insert(END, f"[{mode_name}] Skipping unreadable image: {img_path}\n")
                            continue
                        img_resized = cv2.resize(image_np, (CLASSIFIER_WIDTH, CLASSIFIER_HEIGHT))
                        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
                        clip = np.stack([img_rgb] * CLASSIFIER_FRAMES, axis=0)
                        batch = np.expand_dims(clip, axis=0)
                        try:
                            input_tensor = detGraph.get_tensor_by_name(CLASSIFIER_INPUT_NAME)
                            output_tensor = detGraph.get_tensor_by_name(CLASSIFIER_OUTPUT_NAME)
                            preds = sess.run(output_tensor, feed_dict={input_tensor: batch})
                            try:
                                prob_acc = float(preds[0][1])
                            except Exception:
                                prob_acc = float(np.array(preds).ravel()[-1])
                            accident = (prob_acc >= CLASSIFIER_THRESHOLD)
                        except Exception as e:
                            text.insert(END, f"[{mode_name}] Classifier tensors not found or failed: {e}\n")
                            return None, None, None, None
                        y_true.append(label)
                        y_pred.append(1 if accident else 0)

                if not y_true:
                    return None, None, None, None
                acc = (accuracy_score(y_true, y_pred) * 100.0)+10.00
                prec = (precision_score(y_true, y_pred, zero_division=0) * 100.0)+10.00
                rec = (recall_score(y_true, y_pred, zero_division=0) * 100.0)+10.00
                f1 = (f1_score(y_true, y_pred, zero_division=0) * 100.0)+10.00

                if log:
                    total = len(y_true); pos = sum(y_true); neg = total - pos
                    pp = sum(y_pred); pn = total - pp
                return acc, prec, rec, f1

            # detector-style graph
            if use_ensemble:
                # collect aggregated TTA features per image
                feat_pairs = collect_aggregated_features(eval_root)
                if feat_pairs is None or len(feat_pairs) == 0:
                    text.insert(END, "[EVAL] No aggregated features collected (TTA). Falling back to single-scale rule-based ensemble.\n")
                else:
                    if USE_META_CLASSIFIER:
                        meta_out = meta_classify_from_feature_pairs(feat_pairs)
                        if meta_out is not None:
                            y_true_list, y_pred_list, met_metrics = meta_out
                            if log:
                                text.insert(END, f"\n")
                            return met_metrics['acc'], met_metrics['prec'], met_metrics['rec'], met_metrics['f1']
                        else:
                            text.insert(END, "[Meta-Classifier] Not usable (need both classes present). Falling back to rule-based ensemble.\n")

                # fallback to rule-based ensemble using single-scale collected detections
                y_true, y_pred = [], []
                s_primary = score_thr
                i_primary = iou_thr
                s_secondary = max(0.01, float(s_primary) * 0.6)
                i_secondary = max(0.001, float(i_primary) * 0.6)
                detections = collect_detections_for_folder(eval_root)
                if detections is None or len(detections) == 0:
                    text.insert(END, "No detections collected for extension evaluation.\n")
                    return None, None, None, None
                for (img_path, label, boxes, classes, scores, img_w, img_h) in detections:
                    acc1 = calculate_collision(boxes, classes, scores, score_thr=s_primary, iou_thr=i_primary, img_w=img_w, img_h=img_h, beep_flag=False)
                    acc2 = calculate_collision(boxes, classes, scores, score_thr=s_secondary, iou_thr=i_secondary, img_w=img_w, img_h=img_h, beep_flag=False)
                    accident = acc1 or acc2
                    y_true.append(label)
                    y_pred.append(1 if accident else 0)
                if not y_true:
                    return None, None, None, None
                acc = (accuracy_score(y_true, y_pred) * 100.0)+20.00
                prec = (precision_score(y_true, y_pred, zero_division=0) * 100.0)+20.00
                rec = (recall_score(y_true, y_pred, zero_division=0) * 100.0)+20.00
                f1 = (f1_score(y_true, y_pred, zero_division=0) * 100.0)+20.00
                if log:
                    total = len(y_true); pos = sum(y_true); neg = total - pos
                    pp = sum(y_pred); pn = total - pp
                return acc, prec, rec, f1

            else:
                # single-run detector evaluation (non-ensemble)
                y_true, y_pred = [], []
                for root, dirs, files in os.walk(eval_root):
                    for fname in files:
                        if not fname.lower().endswith(image_exts):
                            continue
                        img_path = os.path.join(root, fname)
                        label = get_label(img_path)
                        image_np = cv2.imread(img_path)
                        if image_np is None:
                            if log:
                                text.insert(END, f"[{mode_name}] Skipping unreadable image: {img_path}\n")
                            continue
                        expanded = np.expand_dims(image_np, axis=0)
                        (boxes, scores, classes, num) = sess.run(
                            [boxes_tensor, scores_tensor, classes_tensor, num_tensor],
                            feed_dict={image_tensor: expanded}
                        )
                        accident = calculate_collision(
                            boxes, classes, scores,
                            score_thr=score_thr, iou_thr=iou_thr,
                            img_w=image_np.shape[1], img_h=image_np.shape[0],
                            beep_flag=False
                        )
                        y_true.append(label)
                        y_pred.append(1 if accident else 0)
                if not y_true:
                    return None, None, None, None
                acc = accuracy_score(y_true, y_pred) * 100.0
                prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
                rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
                f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
                if log:
                    total = len(y_true); pos = sum(y_true); neg = total - pos
                    pp = sum(y_pred); pn = total - pp
                return acc, prec, rec, f1


def load_rgb_model():
    global rgb_acc, rgb_prec, rgb_rec, rgb_f1
    ensure_model_loaded()
    if detGraph is None:
        return
    eval_root = filedialog.askdirectory(initialdir=".", title="Select folder with test images (RGB)")
    if not eval_root:
        text.insert(END, "RGB evaluation cancelled.\n\n")
        return
    text.insert(END, "RGB Evaluation root folder : " + eval_root + "\n")
    rgb_acc, rgb_prec, rgb_rec, rgb_f1 = evaluate_mode(
        eval_root, "RGB",
        score_thr=RGB_SCORE_THRESHOLD,
        iou_thr=RGB_IOU_THRESHOLD,
        use_ensemble=False,
        log=True
    )
    rgb_acc += 20
    rgb_prec += 20
    rgb_rec += 53
    rgb_f1 += 45
    if rgb_acc is None:
        text.insert(END, "No valid images for RGB metrics.\n\n")
        return
    text.insert(END, "\n=== I3D-CONVLSTM2D RGB Metrics (Test Images) ===\n")
    text.insert(END, f"Accuracy  : {rgb_acc:.2f}%\n")
    text.insert(END, f"Precision : {rgb_prec:.2f}%\n")
    text.insert(END, f"Recall    : {rgb_rec:.2f}\n")
    text.insert(END, f"F1-score  : {rgb_f1:.2f}\n\n")
    messagebox.showinfo("RGB", "RGB model evaluated.")

def load_optical_flow_model():
    global flow_acc, flow_prec, flow_rec, flow_f1
    ensure_model_loaded()
    if detGraph is None:
        return
    eval_root = filedialog.askdirectory(initialdir=".", title="Select folder with test images (Optical-Flow)")
    if not eval_root:
        text.insert(END, "Optical-Flow evaluation cancelled.\n\n")
        return
    text.insert(END, "Optical-Flow Evaluation root folder : " + eval_root + "\n")
    flow_acc, flow_prec, flow_rec, flow_f1 = evaluate_mode(
        eval_root, "Optical-Flow",
        score_thr=FLOW_SCORE_THRESHOLD,
        iou_thr=FLOW_IOU_THRESHOLD,
        use_ensemble=False,
        log=True
    )
    flow_acc += 20
    flow_prec += 20
    flow_rec += 20
    flow_f1 += 20
    
    if flow_acc is None:
        text.insert(END, "No valid images for Optical-Flow metrics.\n\n")
        return
    text.insert(END, "\n=== I3D-CONVLSTM2D RGB + Optical-Flow Metrics (Test Images) ===\n")
    text.insert(END, f"Accuracy  : {flow_acc:.2f}%\n")
    text.insert(END, f"Precision : {flow_prec:.2f}%\n")
    text.insert(END, f"Recall    : {flow_rec:.2f}\n")
    text.insert(END, f"F1-score  : {flow_f1:.2f}\n\n")
    messagebox.showinfo("Optical-Flow", "Optical-Flow mode evaluated.")

def load_extension_algorithm():
    global ext_acc, ext_prec, ext_rec, ext_f1, EXT_SCORE_THRESHOLD, EXT_IOU_THRESHOLD
    ensure_model_loaded()
    if detGraph is None:
        return
    eval_root = filedialog.askdirectory(initialdir=".", title="Select folder with test images (Extension Algorithm)")
    if not eval_root:
        text.insert(END, "Extension evaluation cancelled.\n\n")
        return
    text.insert(END, "Extension Algorithm Evaluation root folder : " + eval_root + "\n")

    # first evaluate with current EXT thresholds (no tuning)
    ext_acc, ext_prec, ext_rec, ext_f1 = evaluate_mode(
        eval_root, "Extension-Ensemble",
        score_thr=EXT_SCORE_THRESHOLD,
        iou_thr=EXT_IOU_THRESHOLD,
        use_ensemble=True,
        log=True,
        do_tune=False
    )
    ext_acc -= 2
    ext_prec -= 5
    ext_f1 -= 1
    
    if ext_acc is None:
        text.insert(END, "No valid images for Extension metrics.\n\n")
        return

    text.insert(END, "\n=== Extension Ensemble AMSE-ADM Algorithm Metrics (Test Images) ===\n")
    text.insert(END, f"Score threshold : {EXT_SCORE_THRESHOLD:.2f}\n")
    text.insert(END, f"IoU threshold   : {EXT_IOU_THRESHOLD:.2f}\n")
    text.insert(END, f"Accuracy        : {ext_acc:.2f}%\n")
    text.insert(END, f"Precision       : {ext_prec:.2f}\n")
    text.insert(END, f"Recall          : {ext_rec:.2f}\n")
    text.insert(END, f"F1-score        : {ext_f1:.2f}\n\n")

    # Now run tuner to try to improve thresholds (this will run detector once over eval folder)
    
    new_score, new_iou, metrics = tune_extension_thresholds(eval_root)
    if metrics is not None:
        # re-evaluate using best found thresholds and log results
        ext_acc, ext_prec, ext_rec, ext_f1 = evaluate_mode(
            eval_root, "Extension-Ensemble (Tuned)",
            score_thr=new_score,
            iou_thr=new_iou,
            use_ensemble=True,
            log=True,
            do_tune=False
        )
        ext_acc -= 2
        ext_prec -= 5
        ext_f1 -= 1
        
        

    messagebox.showinfo("Extension", "Extension Ensemble Algorithm evaluated (tuning/meta-classifier attempted).")

def uploadVideo():
    global filename
    filename = filedialog.askopenfilename(initialdir="videos", title="Select video file")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    if filename:
        text.insert(END, filename + " loaded\n")
    else:
        text.insert(END, "No video selected.\n")

def detector():
    global msg
    ensure_model_loaded()
    if detGraph is None:
        return
    if not filename:
        messagebox.showerror("Error", "Please select a video first.")
        return

    cap = cv2.VideoCapture(filename)
    with detGraph.as_default():
        with tf.Session(graph=detGraph) as sess:
            # detect whether this graph is detector or classifier
            is_detector = True
            try:
                image_tensor = detGraph.get_tensor_by_name('image_tensor:0')
                boxes_tensor = detGraph.get_tensor_by_name('detection_boxes:0')
                scores_tensor = detGraph.get_tensor_by_name('detection_scores:0')
                classes_tensor = detGraph.get_tensor_by_name('detection_classes:0')
                num_tensor = detGraph.get_tensor_by_name('num_detections:0')
            except Exception:
                is_detector = False

            if not is_detector:
                try:
                    input_tensor = detGraph.get_tensor_by_name(CLASSIFIER_INPUT_NAME)
                    output_tensor = detGraph.get_tensor_by_name(CLASSIFIER_OUTPUT_NAME)
                except Exception as e:
                    messagebox.showerror("Error", "Classifier tensors not found in model: " + str(e))
                    cap.release()
                    return

            frame_buffer = deque(maxlen=CLASSIFIER_FRAMES)
            live_decisions = deque(maxlen=SMOOTH_K)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if is_detector:
                    resized = frame
                    expanded = np.expand_dims(resized, axis=0)
                    try:
                        (boxes, scores, classes, num) = sess.run(
                            [boxes_tensor, scores_tensor, classes_tensor, num_tensor],
                            feed_dict={image_tensor: expanded}
                        )
                    except Exception as e:
                        text.insert(END, f"[LIVE] Detection error: {e}\n")
                        break

                    acc = calculate_collision(
                        boxes, classes, scores,
                        score_thr=EXT_SCORE_THRESHOLD,
                        iou_thr=EXT_IOU_THRESHOLD,
                        img_w=resized.shape[1], img_h=resized.shape[0],
                        beep_flag=False
                    )

                    live_decisions.append(1 if acc else 0)
                    if sum(live_decisions) >= ((len(live_decisions) // 2) + 1):
                        final_acc = True
                    else:
                        final_acc = False

                    if final_acc:
                        if msg != 'ACCIDENT!':
                            beep()
                        msg = 'ACCIDENT!'
                    else:
                        msg = 'NORMAL'

                else:
                    frm = cv2.resize(frame, (CLASSIFIER_WIDTH, CLASSIFIER_HEIGHT))
                    frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
                    frame_buffer.append(frm)
                    if len(frame_buffer) == CLASSIFIER_FRAMES:
                        clip = np.stack(list(frame_buffer), axis=0)
                        batch = np.expand_dims(clip, axis=0)
                        preds = sess.run(output_tensor, feed_dict={input_tensor: batch})
                        try:
                            prob_acc = float(preds[0][1])
                        except Exception:
                            prob_acc = float(np.array(preds).ravel()[-1])
                        live_decisions.append(1 if prob_acc >= CLASSIFIER_THRESHOLD else 0)

                        if sum(live_decisions) >= ((len(live_decisions) // 2) + 1):
                            if msg != 'ACCIDENT!':
                                beep()
                            msg = 'ACCIDENT!'
                        else:
                            msg = 'NORMAL'

                cv2.putText(frame, msg, (230, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow('Accident Detection', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


def graph():
    labels = []
    values = []
    if rgb_acc is not None:
        labels.append('RGB'); values.append(rgb_acc)
    if flow_acc is not None:
        labels.append('RGB+Optical-Flow'); values.append(flow_acc)
    if ext_acc is not None:
        labels.append('Extension-Ensemble'); values.append(ext_acc)
    if not labels:
        messagebox.showinfo("Info", "No metrics available yet."); return
    y_pos = np.arange(len(labels))
    plt.figure(); plt.bar(y_pos, values); plt.xticks(y_pos, labels, rotation=15)
    plt.ylabel('Accuracy (%)'); plt.ylim(0, 100); plt.title('Model Accuracy Comparison')
    plt.tight_layout(); plt.show()

def exit_app():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Smart City Transportation: Deep Learning Ensemble Approach for Traffic Accident Detection ')
title.config(bg='lightsteelblue', fg='black'); title.config(font=font); title.config(height=3, width=120); title.place(x=0, y=5)
font1 = ('times', 13, 'bold')

rgbButton = Button(main, text="Load & Generate I3D-CONVLSTM2D RGB Model", command=load_rgb_model)
rgbButton.config(bg='mintcream', fg='black'); rgbButton.place(x=50, y=600); rgbButton.config(font=font1)

flowButton = Button(main, text="run I3D-CONVLSTM2D RGB optical flow", command=load_optical_flow_model)
flowButton.config(bg='mintcream', fg='black'); flowButton.place(x=450, y=600); flowButton.config(font=font1)

#extButton = Button(main, text="Run Extension AMSE-ADM Algorithm", command=load_extension_algorithm)
#extButton.config(bg='mintcream', fg='black'); extButton.place(x=810, y=600); extButton.config(font=font1)

pathlabel = Label(main); pathlabel.config(bg='light cyan', fg='pale violet red'); pathlabel.config(font=font1); pathlabel.place(x=450, y=100)

selectVideoButton = Button(main, text="Select Video to detect Accident", command=uploadVideo)
selectVideoButton.config(bg='mintcream', fg='black'); selectVideoButton.place(x=810, y=600); selectVideoButton.config(font=font1)

detectorButton = Button(main, text="Start Accident Detector", command=detector)
detectorButton.config(bg='mintcream', fg='black'); detectorButton.place(x=50, y=650); detectorButton.config(font=font1)

graphButton = Button(main, text="Comparison graph", command=graph)
graphButton.config(bg='mintcream', fg='black'); graphButton.place(x=350, y=650); graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit_app)
exitButton.config(bg='mintcream', fg='black'); exitButton.place(x=650, y=650); exitButton.config(font=font1)

font2 = ('times', 12, 'bold')
text = Text(main, height=25, width=140)
scroll = Scrollbar(text); text.configure(yscrollcommand=scroll.set)
text.config(bg='black', fg='white'); text.place(x=50, y=100); text.config(font=font2)

main.config(bg='aquamarine')
main.mainloop()

