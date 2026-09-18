"""
body.py — Full Body Detection & Pose Visualizer (Tasks API Edition)
===================================================================
Detects the full human body using MediaPipe Tasks PoseLandmarker (33 landmarks).
Draws a beautiful neon skeleton overlay with color-coded limb segments,
confidence scores, joint labels, and a pro HUD panel.

Controls:
  q / Esc  → Quit
  s        → Save snapshot
  h        → Toggle joint-label HUD
"""

import os
import sys
import time
import collections

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
POSE_MODEL_PATH = "pose_landmarker.task"
CAMERA_INDEX    = 0
CAMERA_API      = cv2.CAP_DSHOW
WINDOW_NAME     = "⬛ Full Body Detection  |  q=Quit  s=Snapshot  h=Labels"
SNAPSHOT_DIR    = "snapshots"
FPS_HISTORY     = 60          # frames to average for FPS graph

# ── Colour palette (BGR) ──────────────────────
C_BG_HUD     = (15, 15, 20)
C_ACCENT     = (0, 230, 180)       # cyan-green
C_SPINE      = (100, 200, 255)     # warm sky
C_LEFT_ARM   = (255, 140, 50)      # orange
C_RIGHT_ARM  = (50, 140, 255)      # blue
C_LEFT_LEG   = (220, 80, 255)      # magenta
C_RIGHT_LEG  = (80, 220, 80)       # lime
C_FACE_RING  = (255, 230, 80)      # gold
C_JOINT      = (255, 255, 255)     # white dots
C_TEXT       = (230, 230, 230)
C_SHADOW     = (20, 20, 20)

PL = vision.PoseLandmark

# ── Landmark groups (for colour-coded drawing) ─
SPINE_CONN = [
    (PL.LEFT_SHOULDER,  PL.RIGHT_SHOULDER),
    (PL.LEFT_SHOULDER,  PL.LEFT_HIP),
    (PL.RIGHT_SHOULDER, PL.RIGHT_HIP),
    (PL.LEFT_HIP,       PL.RIGHT_HIP),
    (PL.LEFT_SHOULDER,  PL.NOSE),
    (PL.RIGHT_SHOULDER, PL.NOSE),
]
LEFT_ARM_CONN = [
    (PL.LEFT_SHOULDER, PL.LEFT_ELBOW),
    (PL.LEFT_ELBOW,    PL.LEFT_WRIST),
    (PL.LEFT_WRIST,    PL.LEFT_PINKY),
    (PL.LEFT_WRIST,    PL.LEFT_INDEX),
    (PL.LEFT_WRIST,    PL.LEFT_THUMB),
]
RIGHT_ARM_CONN = [
    (PL.RIGHT_SHOULDER, PL.RIGHT_ELBOW),
    (PL.RIGHT_ELBOW,    PL.RIGHT_WRIST),
    (PL.RIGHT_WRIST,    PL.RIGHT_PINKY),
    (PL.RIGHT_WRIST,    PL.RIGHT_INDEX),
    (PL.RIGHT_WRIST,    PL.RIGHT_THUMB),
]
LEFT_LEG_CONN = [
    (PL.LEFT_HIP,   PL.LEFT_KNEE),
    (PL.LEFT_KNEE,  PL.LEFT_ANKLE),
    (PL.LEFT_ANKLE, PL.LEFT_HEEL),
    (PL.LEFT_ANKLE, PL.LEFT_FOOT_INDEX),
]
RIGHT_LEG_CONN = [
    (PL.RIGHT_HIP,   PL.RIGHT_KNEE),
    (PL.RIGHT_KNEE,  PL.RIGHT_ANKLE),
    (PL.RIGHT_ANKLE, PL.RIGHT_HEEL),
    (PL.RIGHT_ANKLE, PL.RIGHT_FOOT_INDEX),
]

LANDMARK_LABELS = {
    PL.NOSE:              "Nose",
    PL.LEFT_SHOULDER:     "L.Shoulder",
    PL.RIGHT_SHOULDER:    "R.Shoulder",
    PL.LEFT_ELBOW:        "L.Elbow",
    PL.RIGHT_ELBOW:       "R.Elbow",
    PL.LEFT_WRIST:        "L.Wrist",
    PL.RIGHT_WRIST:       "R.Wrist",
    PL.LEFT_HIP:          "L.Hip",
    PL.RIGHT_HIP:         "R.Hip",
    PL.LEFT_KNEE:         "L.Knee",
    PL.RIGHT_KNEE:        "R.Knee",
    PL.LEFT_ANKLE:        "L.Ankle",
    PL.RIGHT_ANKLE:       "R.Ankle",
}

ALL_GROUPS = [
    (SPINE_CONN,    C_SPINE),
    (LEFT_ARM_CONN, C_LEFT_ARM),
    (RIGHT_ARM_CONN,C_RIGHT_ARM),
    (LEFT_LEG_CONN, C_LEFT_LEG),
    (RIGHT_LEG_CONN,C_RIGHT_LEG),
]


# ─────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────
def shadow_text(frame, text, pos, scale=0.6, color=C_TEXT, thickness=1):
    x, y = pos
    cv2.putText(frame, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, C_SHADOW, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_filled_rect(frame, x1, y1, x2, y2, color, alpha=0.55):
    """Semi-transparent filled rectangle."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def lm_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def draw_skeleton(frame, lms, w, h, visibility_thresh=0.5):
    """Draw neon skeleton with colour-coded limb groups."""
    # ── bones ───────────────────────────────
    for group_conns, color in ALL_GROUPS:
        for a, b in group_conns:
            la, lb = lms[a], lms[b]
            if la.visibility < visibility_thresh or lb.visibility < visibility_thresh:
                continue
            pa = lm_px(la, w, h)
            pb = lm_px(lb, w, h)
            # glow effect: thick dim + thin bright
            cv2.line(frame, pa, pb, tuple(c // 3 for c in color), 6, cv2.LINE_AA)
            cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)

    # ── joints ──────────────────────────────
    for idx, lm in enumerate(lms):
        if lm.visibility < visibility_thresh:
            continue
        px = lm_px(lm, w, h)
        cv2.circle(frame, px, 6,  C_JOINT, -1,  cv2.LINE_AA)
        cv2.circle(frame, px, 9,  C_ACCENT, 1,  cv2.LINE_AA)

    # ── face ring (nose + eyes + ears) ──────
    face_ids = [PL.NOSE, PL.LEFT_EYE, PL.RIGHT_EYE,
                PL.LEFT_EAR, PL.RIGHT_EAR]
    face_pts = []
    for fid in face_ids:
        lm = lms[fid]
        if lm.visibility >= visibility_thresh:
            face_pts.append(lm_px(lm, w, h))
    if len(face_pts) >= 2:
        xs = [p[0] for p in face_pts]
        ys = [p[1] for p in face_pts]
        cx, cy = sum(xs) // len(xs), sum(ys) // len(ys)
        r = max(max(xs) - min(xs), max(ys) - min(ys)) // 2 + 20
        cv2.circle(frame, (cx, cy), r, C_FACE_RING, 2, cv2.LINE_AA)


def draw_joint_labels(frame, lms, w, h, visibility_thresh=0.55):
    for idx, label in LANDMARK_LABELS.items():
        lm = lms[idx]
        if lm.visibility < visibility_thresh:
            continue
        px = lm_px(lm, w, h)
        shadow_text(frame, label, (px[0] + 8, px[1] - 6),
                    scale=0.38, color=C_ACCENT, thickness=1)


def draw_hud(frame, fps, fps_history, body_detected, w, h, show_labels):
    """Pro HUD panel — top-left stats card."""
    panel_w, panel_h = 260, 110
    draw_filled_rect(frame, 0, 0, panel_w, panel_h, C_BG_HUD, alpha=0.75)

    # border accent line
    cv2.line(frame, (0, panel_h), (panel_w, panel_h), C_ACCENT, 1)
    cv2.line(frame, (panel_w, 0), (panel_w, panel_h), C_ACCENT, 1)

    # title
    shadow_text(frame, "FULL BODY DETECTION", (10, 22),
                scale=0.55, color=C_ACCENT, thickness=1)

    # FPS
    color_fps = (0, 220, 80) if fps >= 24 else (0, 140, 255)
    shadow_text(frame, f"FPS  {fps:5.1f}", (10, 48),
                scale=0.55, color=color_fps)

    # status
    status     = "BODY  DETECTED" if body_detected else "NO BODY"
    color_stat = (0, 220, 80) if body_detected else (0, 80, 220)
    shadow_text(frame, status, (10, 72), scale=0.55, color=color_stat)

    # labels toggle hint
    hint_col = C_ACCENT if show_labels else (120, 120, 120)
    shadow_text(frame, f"Labels {'ON' if show_labels else 'OFF'}  [H]",
                (10, 96), scale=0.40, color=hint_col)

    # ── mini FPS sparkline ──────────────────
    if len(fps_history) > 1:
        bar_x0, bar_y0 = panel_w + 10, 8
        bar_w, bar_h   = 120, panel_h - 16
        max_fps = max(60.0, max(fps_history))
        pts = []
        for i, f in enumerate(fps_history):
            px = bar_x0 + int(i / (FPS_HISTORY - 1) * bar_w)
            py = bar_y0 + bar_h - int(f / max_fps * bar_h)
            pts.append((px, py))
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], C_ACCENT, 1, cv2.LINE_AA)

    # ── key hints (bottom-right) ────────────
    hints = ["q/Esc  Quit", "s  Snapshot", "h  Toggle labels"]
    bx, by = w - 170, h - 10
    for i, hint in enumerate(reversed(hints)):
        shadow_text(frame, hint, (bx, by - i * 20),
                    scale=0.4, color=(160, 160, 160))


# ─────────────────────────────────────────────
#  Model Builder
# ─────────────────────────────────────────────
def build_pose_landmarker():
    if not os.path.exists(POSE_MODEL_PATH):
        sys.exit(f"Model file not found: {POSE_MODEL_PATH}\n"
                 "Download pose_landmarker.task from MediaPipe and place it "
                 "next to body.py.")

    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    return vision.PoseLandmarker.create_from_options(opts)


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_API)
    if not cap.isOpened():
        sys.exit("Could not open webcam. Close other apps and try again.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Full Body Detection (MediaPipe Tasks API) running.")
    print("  q / Esc  → Quit")
    print("  s        → Save snapshot")
    print("  h        → Toggle joint labels")

    fps_history  = collections.deque(maxlen=FPS_HISTORY)
    prev_ts      = 0
    fps          = 0.0
    show_labels  = True
    last_infer_ms = 0

    landmarker = build_pose_landmarker()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            # FPS
            now_ms = time.monotonic_ns() // 1_000_000
            if prev_ts:
                inst = 1000.0 / max(now_ms - prev_ts, 1)
                fps  = inst if fps == 0 else 0.85 * fps + 0.15 * inst
            prev_ts = now_ms
            fps_history.append(fps)

            h, w = frame.shape[:2]

            # ── Process ─────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            infer_ms = max(now_ms, last_infer_ms + 1)
            last_infer_ms = infer_ms
            results = landmarker.detect_for_video(mp_img, infer_ms)

            # ── Draw ─────────────────────────────
            body_detected = bool(results.pose_landmarks)
            if body_detected:
                lms = results.pose_landmarks[0]
                draw_skeleton(frame, lms, w, h)
                if show_labels:
                    draw_joint_labels(frame, lms, w, h)

            draw_hud(frame, fps, fps_history, body_detected, w, h, show_labels)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("s"):
                fname = os.path.join(
                    SNAPSHOT_DIR,
                    time.strftime("body_%Y%m%d_%H%M%S.png"))
                cv2.imwrite(fname, frame)
                print(f"Snapshot saved → {fname}")
            elif key == ord("h"):
                show_labels = not show_labels

    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
