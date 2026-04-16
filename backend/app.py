import asyncio
import base64
import json
import os
import smtplib
import threading
import time
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional
import functools

import cv2
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("onnxruntime not found — DEMO mode")

# ── Config ─────────────────────────────────────────────────────────────────────
# YOLOv5 raw output: confidence = objectness * class_score
# 0.25 is a good starting point; raise to 0.35-0.5 to reduce false positives
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.20"))
NMS_THRESHOLD = float(os.getenv("NMS_THRESHOLD", "0.3"))
INPUT_SIZE = (640, 640)
STREAM_FPS = int(os.getenv("STREAM_FPS", "30"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
THUMB_QUALITY = int(os.getenv("THUMB_QUALITY", "80"))
NOTIF_DEBOUNCE_SEC = 30

app = FastAPI(title="RoadVisionAI API v11")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── Class config ───────────────────────────────────────────────────────────────
# Your YOLOv5 export: output shape (1, 25200, 12) → 4 box + 1 obj + 7 classes
# Class order from your export log (coco128.yaml was placeholder, real order below)
CLASS_NAMES = ["D00", "D20", "D50", "D40", "D43", "D10", "D44"]
CLASS_COLORS_BGR = {
    "D00": (48, 75, 255),  # longitudinal crack  — blue/red
    "D20": (255, 100, 255),  # alligator crack     — magenta
    "D50": (0, 200, 255),  # pothole             — cyan
    "D40": (255, 185, 115),  # rutting             — orange
    "D43": (85, 238, 48),  # lane mark degrad.   — green
    "D10": (0, 184, 255),  # transverse crack    — light blue
    "D44": (130, 121, 253),  # crosswalk blur      — purple
}
DEFAULT_COLOR = (0, 60, 240)

D44_CLASS_IDX = CLASS_NAMES.index("D44") if "D44" in CLASS_NAMES else -1


# ── ONNX Detector ──────────────────────────────────────────────────────────────
class PotholeDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.session = None
        self.input_name = None
        self._lock = threading.Lock()
        self._yolov5 = False  # will be set True when we detect v5 output shape

        if model_path and ONNX_AVAILABLE and os.path.exists(model_path):
            providers = self._best_providers()
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            out_shape = self.session.get_outputs()[0].shape
            # YOLOv5 raw: (1, 25200, num_classes+5)
            # If dim[1] >> dim[2] it's the raw anchor grid format
            if len(out_shape) == 3 and out_shape[1] > out_shape[2]:
                self._yolov5 = True
                print(f"✅ YOLOv5 raw output detected: {out_shape}")
            else:
                print(f"✅ Standard output detected: {out_shape}")
            print(f"   Providers: {self.session.get_providers()}")

    def _best_providers(self):
        avail = ort.get_available_providers() if ONNX_AVAILABLE else []
        p = []
        if "CoreMLExecutionProvider" in avail:
            p.append("CoreMLExecutionProvider")
        if "CUDAExecutionProvider" in avail:
            p.append(("CUDAExecutionProvider", {"device_id": 0}))
        p.append("CPUExecutionProvider")
        return p

    def preprocess(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        img = cv2.resize(frame, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]
        return img, h, w

    # ── YOLOv5 raw postprocess ─────────────────────────────────────────────
    def _postprocess_yolov5(self, output: np.ndarray, orig_h: int, orig_w: int):
        """
        YOLOv5 ONNX raw output: shape (1, 25200, 5 + num_classes)
          col 0-3 : cx, cy, w, h  (normalised to INPUT_SIZE)
          col 4   : objectness score (already sigmoid'd by the model)
          col 5+  : per-class scores (already sigmoid'd)

        Final confidence = objectness × max_class_score
        This is what YOLOv5 does internally before its own NMS.
        """
        pred = output[0]  # (25200, 12)
        sx = orig_w / INPUT_SIZE[0]
        sy = orig_h / INPUT_SIZE[1]

        boxes, scores, class_ids = [], [], []

        for row in pred:
            obj_conf = float(row[4])  # objectness
            cls_scores = row[5:]  # per-class probabilities

            # Skip rows where objectness is already hopeless
            if obj_conf < 0.01:
                continue

            cls_id = int(cls_scores.argmax())
            cls_conf = float(cls_scores[cls_id])
            conf = obj_conf * cls_conf  # ← THE KEY FIX

            if conf < CONFIDENCE_THRESHOLD:
                continue
            if cls_id == D44_CLASS_IDX:
                continue

            # cx, cy, w, h → x1, y1, x2, y2  (scale to original image)
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            x1 = (cx - bw / 2) * sx
            y1 = (cy - bh / 2) * sy
            x2 = (cx + bw / 2) * sx
            y2 = (cy + bh / 2) * sy

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)
            class_ids.append(cls_id)

        if not boxes:
            return []

        # cv2.dnn.NMSBoxes expects float32 lists
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, w_, h_ = boxes[i]
                results.append([x1, y1, x1 + w_, y1 + h_, scores[i], class_ids[i]])
        return results

    # ── Standard / YOLOv8 postprocess (kept for compatibility) ────────────
    def _postprocess_standard(self, output: np.ndarray, orig_h: int, orig_w: int):
        """
        For models that already applied NMS or use the transposed YOLOv8 format:
        shape (1, num_classes+4, num_boxes)  → transpose to (num_boxes, ...)
        """
        pred = output[0]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T  # (num_boxes, 4 + num_classes)

        sx = orig_w / INPUT_SIZE[0]
        sy = orig_h / INPUT_SIZE[1]
        boxes, scores, class_ids = [], [], []

        for row in pred:
            cls_scores = row[4:]
            conf = float(cls_scores.max())
            cls_id = int(cls_scores.argmax())

            if conf < CONFIDENCE_THRESHOLD:
                continue
            if cls_id == D44_CLASS_IDX:
                continue

            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            x1 = (cx - bw / 2) * sx
            y1 = (cy - bh / 2) * sy
            x2 = (cx + bw / 2) * sx
            y2 = (cy + bh / 2) * sy

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)
            class_ids.append(cls_id)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x1, y1, w_, h_ = boxes[i]
                results.append([x1, y1, x1 + w_, y1 + h_, scores[i], class_ids[i]])
        return results

    def postprocess(self, output: np.ndarray, orig_h: int, orig_w: int):
        if self._yolov5:
            return self._postprocess_yolov5(output, orig_h, orig_w)
        return self._postprocess_standard(output, orig_h, orig_w)

    def detect(self, frame: np.ndarray):
        if self.session is None:
            return self._demo_detections(frame)
        img, h, w = self.preprocess(frame)
        with self._lock:
            outputs = self.session.run(None, {self.input_name: img})
        return self.postprocess(outputs[0], h, w)

    def _demo_detections(self, frame: np.ndarray):
        import random

        h, w = frame.shape[:2]
        if random.random() > 0.22:
            return []
        out = []
        for _ in range(random.randint(1, 2)):
            x1 = random.randint(w // 4, w // 2)
            y1 = random.randint(h // 2, 2 * h // 3)
            bw = random.randint(60, 140)
            bh = random.randint(40, 90)
            cls_id = random.randint(0, len(CLASS_NAMES) - 2)
            out.append([x1, y1, x1 + bw, y1 + bh, random.uniform(0.45, 0.85), cls_id])
        return out


def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    out = frame.copy()
    h, w = frame.shape[:2]
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
        color = CLASS_COLORS_BGR.get(name, DEFAULT_COLOR)
        # Outer shadow rect
        # cv2.rectangle(
        #     out,
        #     (x1 - 1, y1 - 1),
        #     (x2 + 1, y2 + 1),
        #     tuple(int(c // 3) for c in color),
        #     1,
        # )
        # Main rect
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        label = f"{name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        # cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
        # cv2.putText(
        #     out,
        #     label,
        #     (x1 + 4, y1 - 5),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.48,
        #     (0, 0, 0),
        #     1,
        #     cv2.LINE_AA,
        # )
    return out


# ── GPS Interpolation ──────────────────────────────────────────────────────────
def interpolate_gps(waypoints: list, total_frames: int, frame_idx: int) -> dict:
    if not waypoints:
        return {"lat": 18.5204, "lng": 73.8567}
    if len(waypoints) == 1 or total_frames <= 1:
        return waypoints[0]
    t = max(0.0, min(frame_idx / (total_frames - 1), 1.0))
    n = len(waypoints) - 1
    si = min(int(t * n), n - 1)
    st = t * n - si
    a, b = waypoints[si], waypoints[si + 1]
    return {
        "lat": a["lat"] + (b["lat"] - a["lat"]) * st,
        "lng": a["lng"] + (b["lng"] - a["lng"]) * st,
    }


# ── Notifications ──────────────────────────────────────────────────────────────
class NotificationManager:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_pass = os.getenv("SMTP_PASS", "")
        self.alert_to = os.getenv("ALERT_TO", "")
        self.ntfy_topic = os.getenv("NTFY_TOPIC", "")
        self.enabled = bool(self.smtp_user and self.alert_to)
        self._last_sent = 0.0

    def configure(self, cfg: dict):
        for k in (
            "smtp_host",
            "smtp_port",
            "smtp_user",
            "smtp_pass",
            "alert_to",
            "ntfy_topic",
        ):
            if k in cfg:
                setattr(self, k, cfg[k])
        self.smtp_port = int(self.smtp_port)
        self.enabled = bool(self.smtp_user and self.alert_to)

    def should_notify(self) -> bool:
        return (time.time() - self._last_sent) > NOTIF_DEBOUNCE_SEC

    def send(self, event: dict, thumb_b64: Optional[str] = None):
        if not self.should_notify():
            return
        self._last_sent = time.time()
        if self.enabled and self.smtp_user:
            try:
                self._send_email(event, thumb_b64)
            except Exception as e:
                print(f"[SMTP] Error: {e}")
        if self.ntfy_topic:
            try:
                self._send_ntfy(event)
            except Exception as e:
                print(f"[ntfy] Error: {e}")

    def _send_email(self, ev: dict, thumb_b64: Optional[str]):
        conf = ev.get("conf", 0)
        lat = ev.get("lat", 0)
        lng = ev.get("lng", 0)
        frame = ev.get("frame", 0)
        cls = ev.get("class_name", "road damage")
        maps_url = f"https://maps.google.com/?q={lat},{lng}"
        html = f"""<html><body style="font-family:sans-serif">
        <h2 style="color:#ff4b4b">⚠ {cls.title()} detected</h2>
        <p>Confidence: {conf * 100:.1f}% | Frame: #{frame}</p>
        <p>GPS: <a href="{maps_url}">{lat:.6f}, {lng:.6f}</a></p>
        {"<img src='cid:thumb' style='max-width:480px;border-radius:8px'>" if thumb_b64 else ""}
        </body></html>"""
        msg = MIMEMultipart("related")
        msg["Subject"] = f"🚨 RoadVisionAI: {cls} detected ({conf * 100:.0f}%)"
        msg["From"] = self.smtp_user
        msg["To"] = self.alert_to
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText("Road damage detected.", "plain"))
        alt.attach(MIMEText(html, "html"))
        msg.attach(alt)
        if thumb_b64:
            img_part = MIMEImage(base64.b64decode(thumb_b64), _subtype="jpeg")
            img_part.add_header("Content-ID", "<thumb>")
            img_part.add_header("Content-Disposition", "inline")
            msg.attach(img_part)
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as s:
            s.ehlo()
            s.starttls()
            s.login(self.smtp_user, self.smtp_pass)
            for rcpt in self.alert_to.split(","):
                s.sendmail(self.smtp_user, rcpt.strip(), msg.as_string())

    def _send_ntfy(self, ev: dict):
        conf = ev.get("conf", 0)
        lat = ev.get("lat", 0)
        lng = ev.get("lng", 0)
        cls = ev.get("class_name", "road damage")
        httpx.post(
            f"https://ntfy.sh/{self.ntfy_topic}",
            content=f"{cls.title()} @ {lat:.5f},{lng:.5f} — {conf * 100:.0f}% conf",
            headers={
                "Title": f"RoadVisionAI: {cls.title()} detected",
                "Priority": "high" if conf > 0.7 else "default",
                "Tags": "warning,road",
                "Click": f"https://maps.google.com/?q={lat},{lng}",
            },
            timeout=5,
        )


# ── Global singletons ──────────────────────────────────────────────────────────
detector = PotholeDetector()
notifier = NotificationManager()

video_path: Optional[str] = None
waypoints: list = [
    {"lat": 18.5204, "lng": 73.8567},
    {"lat": 18.5290, "lng": 73.8660},
]
pothole_events: list = []


# ── HTTP Routes ────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "status": "RoadVisionAI API v3",
        "mode": "onnx" if detector.session else "demo",
    }


@app.get("/health")
async def health():
    return {
        "ok": True,
        "onnx": ONNX_AVAILABLE,
        "model_loaded": detector.session is not None,
        "yolov5_mode": detector._yolov5,
        "providers": detector.session.get_providers() if detector.session else [],
        "conf_thresh": CONFIDENCE_THRESHOLD,
        "nms_thresh": NMS_THRESHOLD,
        "notifier_on": notifier.enabled,
        "ntfy_topic": notifier.ntfy_topic,
        "stream_fps": STREAM_FPS,
    }


@app.post("/upload/model")
async def upload_model(file: UploadFile = File(...)):
    global detector
    dest = Path("models") / file.filename
    dest.parent.mkdir(exist_ok=True)
    dest.write_bytes(await file.read())
    detector = PotholeDetector(str(dest))
    return {
        "status": "ok",
        "model": file.filename,
        "yolov5": detector._yolov5,
        "providers": detector.session.get_providers() if detector.session else [],
    }


@app.post("/upload/video")
async def upload_video(file: UploadFile = File(...)):
    global video_path
    dest = Path("uploads") / file.filename
    dest.parent.mkdir(exist_ok=True)
    dest.write_bytes(await file.read())
    video_path = str(dest)
    return {"status": "ok", "video": file.filename}


@app.post("/upload/waypoints")
async def set_waypoints(points: list = Body(...)):
    global waypoints
    waypoints = points
    return {"status": "ok", "count": len(points)}


@app.post("/config")
async def set_config(cfg: dict = Body(...)):
    notif_keys = {
        "smtp_host",
        "smtp_port",
        "smtp_user",
        "smtp_pass",
        "alert_to",
        "ntfy_topic",
    }
    if notif_keys & set(cfg.keys()):
        notifier.configure({k: cfg[k] for k in notif_keys if k in cfg})
    return {"status": "ok", "notifier_enabled": notifier.enabled}


@app.get("/events")
async def get_events():
    return JSONResponse(pothole_events)


@app.post("/test-notification")
async def test_notification():
    fake_ev = {
        "id": 0,
        "lat": 18.5204,
        "lng": 73.8567,
        "conf": 0.82,
        "frame": 0,
        "class_name": "D20 (test)",
        "ts": time.time(),
    }
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, functools.partial(notifier.send, fake_ev, None))
    return {
        "status": "sent",
        "enabled": notifier.enabled,
        "ntfy": bool(notifier.ntfy_topic),
    }


# ── WebSocket Stream ───────────────────────────────────────────────────────────
@app.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    global pothole_events
    pothole_events = []

    query = websocket.query_params.get("src", "file")
    is_webcam = query == "webcam"
    loop = asyncio.get_event_loop()
    encode_p = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    thumb_p = [cv2.IMWRITE_JPEG_QUALITY, THUMB_QUALITY]
    frame_idx = 0

    async def send_frame(frame: np.ndarray, fi: int, total: int):
        detections = await loop.run_in_executor(None, detector.detect, frame)
        annotated = draw_detections(frame, detections)

        ok, buf = await loop.run_in_executor(
            None, functools.partial(cv2.imencode, ".jpg", annotated, encode_p)
        )
        b64 = base64.b64encode(buf.tobytes()).decode()
        gps = interpolate_gps(waypoints, total, fi)

        payload = {
            "frame": b64,
            "gps": gps,
            "detections": len(detections),
            "frame_idx": fi,
            "total_frames": total,
        }

        if detections:
            ok2, tbuf = await loop.run_in_executor(
                None, functools.partial(cv2.imencode, ".jpg", annotated, thumb_p)
            )
            thumb_b64 = base64.b64encode(tbuf.tobytes()).decode()
            top_det = max(detections, key=lambda d: d[4])
            cls_id = int(top_det[5]) if len(top_det) > 5 else 0
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
            ev = {
                "id": len(pothole_events),
                "lat": gps["lat"],
                "lng": gps["lng"],
                "conf": round(float(top_det[4]), 3),
                "class_name": class_name,
                "frame": fi,
                "thumb": thumb_b64,
                "ts": time.time(),
            }
            pothole_events.append(ev)
            payload["event"] = ev
            if notifier.should_notify():
                await loop.run_in_executor(
                    None, functools.partial(notifier.send, ev, thumb_b64)
                )

        await websocket.send_json(payload)

    # ── Webcam mode ────────────────────────────────────────────────────────
    if is_webcam:
        try:
            while True:
                data = await websocket.receive_bytes()
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                await send_frame(frame, frame_idx, 99999)
                frame_idx += 1
        except WebSocketDisconnect:
            print("Webcam client disconnected")
        return

    # ── File mode ──────────────────────────────────────────────────────────
    source = video_path if video_path else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        await websocket.send_json({"error": f"Cannot open source: {source}"})
        await websocket.close()
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 99999
    frame_dt = 1.0 / STREAM_FPS
    last_send = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await websocket.send_json(
                    {"done": True, "frame_idx": frame_idx, "total_frames": total_frames}
                )
                break

            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            now = time.monotonic()
            if now - last_send < frame_dt:
                continue

            await send_frame(frame, frame_idx, total_frames)
            last_send = now

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Stream error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        cap.release()


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("DEV", "0") == "1",
        workers=1,
        loop="uvloop",
        http="httptools",
    )
