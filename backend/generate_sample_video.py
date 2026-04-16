import cv2
import numpy as np
import random
import os

OUT_PATH = "../sample_data/sample_dashcam.mp4"
os.makedirs("../sample_data", exist_ok=True)

W, H = 640, 360
FPS = 30
DURATION = 12  # seconds
TOTAL = FPS * DURATION

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (W, H))

# Road + sky colors
SKY = (180, 210, 230)
ROAD = (80, 80, 85)
LINE = (220, 220, 180)

potholes = []  # (start_frame, cx_pct, depth_px)
for i in range(6):
    potholes.append(
        {
            "start": random.randint(10, TOTAL - 60),
            "cx": random.uniform(0.25, 0.75),
            "r": random.randint(18, 35),
        }
    )

for f in range(TOTAL):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    # Sky
    frame[: H // 2] = SKY
    # Road
    frame[H // 2 :] = ROAD
    # Dashed center line
    horizon_y = H // 2
    for seg in range(10):
        y1 = horizon_y + seg * 25 + (f * 2) % 25
        y2 = y1 + 12
        if y1 < H and y2 < H:
            cv2.line(frame, (W // 2, y1), (W // 2, y2), LINE, 2)
    # Road edges
    cv2.line(frame, (W // 4, H // 2), (0, H), (200, 200, 180), 2)
    cv2.line(frame, (3 * W // 4, H // 2), (W, H), (200, 200, 180), 2)
    # Potholes
    for ph in potholes:
        s = ph["start"]
        if s <= f <= s + 45:
            progress = (f - s) / 45.0
            cy = int(H // 2 + 20 + progress * (H - H // 2 - 40))
            cx = int(ph["cx"] * W)
            r = ph["r"]
            cv2.ellipse(frame, (cx, cy), (r, r // 2), 0, 0, 360, (45, 45, 50), -1)
            cv2.ellipse(frame, (cx, cy), (r, r // 2), 0, 0, 360, (30, 30, 35), 2)
    # Speed overlay
    speed = random.randint(28, 45)
    cv2.putText(
        frame,
        f"{speed} km/h",
        (10, H - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    out.write(frame)

out.release()
print(f"✅ Sample video saved → {OUT_PATH}")
