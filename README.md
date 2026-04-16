# RoadVisionAI

Real-time pothole detection with optical flow speed estimation.
YOLOv8 ONNX inference · FastAPI WebSocket streaming · Leaflet map.

```
project/
├── backend/
│   ├── app.py                  ← FastAPI + WebSocket stream + OF speed
│   ├── requirements.txt        ← Docker / Linux deps (CPU ONNX)
│   ├── requirements-native.txt ← macOS deps (CoreML ONNX)
│   └── Dockerfile
├── index.html              ← Self-contained SPA, open directly in browser
├── docker-compose.yml
├── run-native.sh               ← Start natively (CoreML/MPS on Apple Silicon)
└── run-docker.sh               ← Build + start Docker backend
```

---

## Quick start

### Native (recommended on Apple Silicon — uses CoreML / MPS)
```bash
chmod +x run-native.sh
./run-native.sh
# Then open frontend/index.html in your browser
```

### Docker
```bash
chmod +x run-docker.sh
./run-docker.sh
# Then open frontend/index.html in your browser
```

---

## Why ONNX Runtime, not WASM or Triton

| | ONNX Runtime (native) | WASM (browser) | Triton |
|---|---|---|---|
| Speed | ✅ Fast, uses CoreML/GPU | ❌ ~4× slower, CPU only | ✅ Fast |
| MPS support | ✅ CoreML EP on macOS | ❌ No | ❌ CUDA only |
| Setup complexity | Low | Medium | High |
| Works in Docker | ✅ CPU | ✅ N/A | ❌ Needs NVIDIA |
| Recommended | **Yes** | No | No (wrong platform) |

**WASM** runs inference in the browser but has no access to MPS and is
significantly slower than native ONNX. It would work for a toy demo but
at 640×640 YOLOv8 input you'd get ~2–5 fps in the browser vs ~20+ fps native.

**Triton Inference Server** requires NVIDIA CUDA GPUs. It will not use Apple
Silicon's GPU or ANE. Skip entirely for MPS machines.

**CoreML Execution Provider** (included in standard `onnxruntime` on macOS 12+)
maps the model to Metal/ANE automatically. Verify with:
```bash
python3 -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should include: CoreMLExecutionProvider
```

---

## Speed estimation — how it works

Speed is estimated using **sparse Lucas-Kanade optical flow** on road-surface
feature points only. This is significantly more accurate than GPS interpolation.

### Key design choices:

1. **Road-plane masking** — features are only tracked in the bottom 55% of
   the frame (the road surface). This excludes sky, buildings, and vehicles ahead.

2. **Moving object rejection** — after computing flow vectors, outliers beyond
   2.5× the median magnitude are discarded. A car driving at a different speed
   in the road region will have anomalous flow vectors and is automatically filtered.

3. **Ground-plane scale** — pixel displacement is converted to metres using a
   pinhole camera model approximation:
   - Assumes ~60° horizontal FoV (typical dashcam)
   - Camera mounted at ~1.2m height
   - Bottom of frame ≈ 3m ahead of vehicle

4. **Temporal smoothing** — last 8 readings are averaged to reduce jitter.

5. **Feature refresh** — Shi-Tomasi corners are re-detected every time fewer
   than 20 good tracks remain, ensuring continuous coverage.

This approach does **not** require GPS, a known reference object, or calibration.
It self-scales based on the camera model geometry.

---

## FPS optimisation

Three changes achieve ~25 fps output:

1. **Async inference queue** (`InferencePipeline`) — ONNX runs in a background
   thread. The WebSocket loop never waits for inference; it uses the last
   available result. This decouples encoding speed from model speed.

2. **Inference skip** (`INFERENCE_SKIP=2`) — inference runs every 2nd frame.
   At 25 fps stream → ~12 inferences/s. Potholes don't disappear in 80ms.

3. **JPEG encoding in executor** — `cv2.imencode` is moved to a thread pool
   via `loop.run_in_executor` so the asyncio loop is never blocked.

Tune via environment variables:
```bash
STREAM_FPS=30 INFERENCE_SKIP=3 JPEG_QUALITY=65 ./run-native.sh
```

---

## ONNX model

Export your YOLOv8 pothole model:
```bash
pip install ultralytics
yolo export model=pothole-best.pt format=onnx imgsz=640 opset=17
```

Drop the `.onnx` file in `models/` and upload via the UI, or set the path
via the `/upload/model` endpoint.

Without a model the system runs in **DEMO mode** with random synthetic
detections — useful for testing the UI and WebSocket pipeline.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `STREAM_FPS` | 25 | Target WebSocket send rate |
| `INFERENCE_SKIP` | 2 | Run inference every N frames |
| `JPEG_QUALITY` | 72 | Stream JPEG quality (40–95) |
| `THUMB_QUALITY` | 50 | Event thumbnail JPEG quality |
| `CONFIDENCE_THRESHOLD` | 0.35 | Min detection confidence |
| `PORT` | 8000 | Server port |
