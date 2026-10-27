import subprocess
import json
import os
import ffmpeg_utils
import math
import sys
import types
from PIL import Image # type: ignore
import numpy as np # type: ignore

# mediapipe internally imports cv2 only for its drawing utilities (drawing_styles.py ->
# drawing_utils.py). Our code never calls any drawing functions. We pre-stub cv2 in
# sys.modules so the native OpenCV binary (and its libGL dependency) is never loaded.
if 'cv2' not in sys.modules:
    _cv2_stub = types.ModuleType('cv2')
    sys.modules['cv2'] = _cv2_stub

import mediapipe as mp # type: ignore
from mediapipe.tasks import python # type: ignore
from mediapipe.tasks.python import vision # type: ignore
import io
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')

def _ensure_model_exists():
  if not os.path.exists(model_path):
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    urllib.request.urlretrieve(url, model_path)

_ensure_model_exists()
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.18)
face_detector = vision.FaceDetector.create_from_options(options)

async def process_clip(
  video_path: str,
  suggestion: dict[str, Any],
  output_dir: str,
  project_id: str,
  forced_layout_mode: Optional[str] = None
) -> dict[str, Any]:
  """
  Full pipeline for one clip:
  1. Cut raw clip at suggestion timestamps
  2. Sample frames using FFmpeg
  3. Detect faces in sampled frames (with deduplication)
  4. Generate keyframes & apply dynamic face-reactive zoom using sendcmd
  5. Return clip metadata
  """

  raw_clip = os.path.join(
    output_dir,
    f"raw_{project_id}_{suggestion['start']}.mp4"
  )
  await _cut_raw_clip(
    video_path, suggestion['start'],
    suggestion['end'], raw_clip
  )

  width, height = _get_video_dimensions(raw_clip)
  duration = suggestion['end'] - suggestion['start']
  
  frames = _extract_sample_frames(raw_clip)
  face_data = _detect_faces_all_frames(frames, width, height)

  output_clip = os.path.join(
    output_dir,
    f"clip_{project_id}_{suggestion['start']}.mp4"
  )

  success, face_count = await _reframe_dynamic_zoom(raw_clip, output_clip, width, height, face_data, duration, frames=frames)
  if not success:
    await _reframe_static_fallback(raw_clip, output_clip, width, height, face_data)

  if os.path.exists(raw_clip):
    os.remove(raw_clip)

  return {
    "file_path": output_clip,
    "start_time": suggestion['start'],
    "end_time": suggestion['end'],
    "duration": duration,
    "title": suggestion.get('title', ''),
    "reason": suggestion.get('reason', ''),
    "viral_score": suggestion.get('viral_score', 0),
    "face_count": face_count,
    "layout_mode": "dynamic_zoom",
    "needs_user_confirm": False,
    "reframed": True
  }


def _extract_sample_frames(video_path: str) -> list[Image.Image]:
  """
  Extract 1 frame per second as raw bytes using FFmpeg.
  No temp files — pipe directly to memory.
  """
  ffmpeg_path = ffmpeg_utils.get_ffmpeg_exe()
  result = subprocess.run([
    ffmpeg_path, "-i", video_path,
    "-vf", "fps=1",
    "-f", "image2pipe",
    "-vcodec", "mjpeg",
    "pipe:1"
  ], capture_output=True)

  raw = result.stdout
  frames = []
  start = 0
  while True:
    start = raw.find(b'\xff\xd8', start)
    if start == -1:
      break
    end = raw.find(b'\xff\xd9', start)
    if end == -1:
      break
    jpeg_bytes = bytes(raw[start:end + 2])
    try:
      img = Image.open(io.BytesIO(jpeg_bytes))
      frames.append(img)
    except Exception:
      pass
    start = end + 2

  return frames


def _detect_faces_all_frames(
  frames: list[Image.Image],
  vid_width: int,
  vid_height: int
) -> list[list[dict]]:
  """
  Run MediaPipe face detection on each PIL frame.
  Applies deduplication tracking rules.
  """
  import numpy as np # type: ignore

  all_faces = []
  for frame in frames:
    rgb = np.array(frame.convert('RGB'))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_detector.detect(mp_image)
    faces = []
    if results.detections:
      for det in results.detections:
        bb = det.bounding_box
        cx = bb.origin_x + bb.width / 2
        cy = bb.origin_y + bb.height / 2
        fw = bb.width
        fh = bb.height
        size = bb.height / vid_height
        
        # Deduplication Rule: Remove size < 0.05
        if size < 0.05:
            continue
            
        faces.append({
          "cx": cx, "cy": cy,
          "w": fw, "h": fh,
          "size": size
        })
    
    # Deduplication Rule: Remove detections within 50px of another
    valid_faces = []
    for f in faces:
        is_dup = False
        for vf in valid_faces:
            if math.hypot(f['cx'] - vf['cx'], f['cy'] - vf['cy']) < 50:
                is_dup = True
                break
        if not is_dup:
            valid_faces.append(f)
            
    all_faces.append(valid_faces)
  return all_faces



def _extract_mouth_roi(frame: Image.Image, face: dict) -> Optional[np.ndarray]:
    """Extract and normalize mouth region of interest for speech motion detection."""
    cx, cy = face['cx'], face['cy']
    fw, fh = face['w'], face['h']
    x1 = int(max(0, cx - fw * 0.28))
    x2 = int(min(frame.width, cx + fw * 0.28))
    y1 = int(max(0, cy + fh * 0.12))
    y2 = int(min(frame.height, cy + fh * 0.44))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame.crop((x1, y1, x2, y2)).convert('L').resize((32, 32))
    return np.array(crop, dtype=np.float32)

def _compute_mouth_diff(roi1: Optional[np.ndarray], roi2: Optional[np.ndarray]) -> float:
    """Compute frame-to-frame mouth movement difference."""
    if roi1 is None or roi2 is None:
        return 0.0
    return float(np.mean(np.abs(roi1 - roi2)))

def _ease_in_out(t: float) -> float:
    return 2 * t * t if t < 0.5 else 1 - math.pow(-2 * t + 2, 2) / 2


async def _reframe_dynamic_zoom(
  input_path: str,
  output_path: str,
  vid_width: int,
  vid_height: int,
  face_data: list[list[dict]],
  duration: float,
  frames: Optional[list[Image.Image]] = None
) -> tuple[bool, int]:
    fps = 30
    total_frames = int(duration * fps)
    
    # Majority vote for face count
    face_counts = [len(f) for f in face_data]
    overall_face_count = max(set(face_counts), key=face_counts.count) if face_counts else 0

    # 1. Calculate an adaptive, stable base zoom from median face size across the clip.
    # This prevents the camera from "breathing" (zooming in/out erratically on tiny face fluctuations).
    all_sizes = [f['size'] for sec in face_data for f in sec if f.get('size')]
    if all_sizes:
        all_sizes.sort()
        median_size = all_sizes[len(all_sizes) // 2]
        # In vertical 9:16, face occupying ~22% of height gives a clean, professional Medium Close-Up.
        base_zoom = max(1.20, min(0.24 / max(median_size, 0.08), 1.75))
    else:
        median_size = 0.25
        base_zoom = 1.35

    logger.info(f"[Reframer] Professional Auto-Reframe: median face size={median_size:.3f}, base_zoom={base_zoom:.2f}")

    # 2. Build per-second targets with Dynamic Scene-Aware Active Speaker Tracking
    # Seamlessly handles: single speaker, dual speaker interviews, scene changes, and momentary face turns.
    sec_targets = []
    total_secs = math.ceil(duration)
    last_known_cx = vid_width / 2.0
    last_known_cy = vid_height / 2.0
    active_speaker_hold_streak = 0
    SWITCH_STREAK_REQUIRED = 2

    crop_w_nominal = (float(vid_height) / base_zoom) * 9.0 / 16.0

    for s in range(len(face_data)):
        sec_faces = face_data[s]

        if not sec_faces:
            # Case 3: Momentary face turn / profile -> HOLD LAST KNOWN SPEAKER!
            # Never drift into the empty middle space!
            sec_targets.append({
                "has_face": False,
                "cx": last_known_cx,
                "cy": last_known_cy,
                "zoom": base_zoom
            })
            continue

        if len(sec_faces) == 1:
            # Case 1: Exactly 1 person on screen (e.g. monologue, host cut) -> Lock on person
            f = sec_faces[0]
            last_known_cx = f['cx']
            last_known_cy = f['cy']
            sec_targets.append({
                "has_face": True,
                "cx": f['cx'],
                "cy": f['cy'],
                "zoom": base_zoom
            })
        else:
            # Case 2: Multiple people on screen (e.g. podcast table, interview)
            sorted_faces = sorted(sec_faces, key=lambda x: x['cx'])
            f1, f2 = sorted_faces[0], sorted_faces[-1]
            face_dist = abs(f1['cx'] - f2['cx'])

            # If both people are close enough to fit in a 9:16 vertical frame:
            if face_dist < (crop_w_nominal * 0.72):
                c_x = (f1['cx'] + f2['cx']) / 2.0
                c_y = (f1['cy'] + f2['cy']) / 2.0
                last_known_cx = c_x
                last_known_cy = c_y
                sec_targets.append({
                    "has_face": True,
                    "cx": c_x,
                    "cy": c_y,
                    "zoom": max(1.15, min(base_zoom, 1.35))
                })
            else:
                # Two separated speakers (e.g. Left vs Right):
                # Measure mouth activity if frames exist
                m1_diff, m2_diff = 0.0, 0.0
                if frames and s > 0 and s < len(frames):
                    prev_f = frames[s - 1]
                    curr_f = frames[s]
                    r1_prev = _extract_mouth_roi(prev_f, f1)
                    r1_curr = _extract_mouth_roi(curr_f, f1)
                    r2_prev = _extract_mouth_roi(prev_f, f2)
                    r2_curr = _extract_mouth_roi(curr_f, f2)
                    m1_diff = _compute_mouth_diff(r1_prev, r1_curr)
                    m2_diff = _compute_mouth_diff(r2_prev, r2_curr)

                # Determine who is actively speaking
                if m2_diff > m1_diff * 1.35 and m2_diff > 6.0:
                    dominant_face = f2
                elif m1_diff > m2_diff * 1.35 and m1_diff > 6.0:
                    dominant_face = f1
                else:
                    # Maintain focus on whichever face is closer to the current camera position
                    dominant_face = min(sorted_faces, key=lambda f: abs(f['cx'] - last_known_cx))

                last_known_cx = dominant_face['cx']
                last_known_cy = dominant_face['cy']
                sec_targets.append({
                    "has_face": True,
                    "cx": dominant_face['cx'],
                    "cy": dominant_face['cy'],
                    "zoom": base_zoom
                })

    while len(sec_targets) < total_secs:
        sec_targets.append(sec_targets[-1] if sec_targets else {
            "has_face": False, "cx": last_known_cx, "cy": last_known_cy, "zoom": base_zoom
        })

    # 3. Continuous frame-by-frame smoothing with Smart Deadband.
    # A seated speaker talking/nodding shifts by 20-40px. A professional camera operator
    # locks the camera unless the speaker actually moves outside the tolerance zone.
    DEADBAND_X = 65.0      # Tolerance zone (px) - no camera wobble on natural head gestures
    DEADBAND_Y = 45.0
    ALPHA_POS = 0.045      # Cinematic gimbal glide (~0.8s smooth settling)
    ALPHA_ZOOM = 0.020     # Ultra-gentle zoom transition (~1.5s settling)
    MAX_PAN_SPEED = 4.5    # Max 4.5 px/frame (135 px/sec) to prevent dizzying whip-pans
    HOLD_NO_FACE = int(3.5 * fps)  # Hold framing 3.5s on B-roll/slides before gentle wide center

    curr_cx = sec_targets[0]["cx"] if sec_targets[0]["has_face"] else vid_width / 2.0
    curr_cy = sec_targets[0]["cy"] if sec_targets[0]["has_face"] else vid_height / 2.0
    curr_zoom = base_zoom if sec_targets[0]["has_face"] else 1.0

    target_cx = curr_cx
    target_cy = curr_cy
    target_zoom = curr_zoom

    frames_since_face = 0
    frame_crops = []

    for f_idx in range(total_frames):
        sec = min(int(f_idx / fps), total_secs - 1)
        st = sec_targets[sec]

        if st["has_face"]:
            frames_since_face = 0
            # Apply Deadband: Only re-anchor target if movement exceeds tolerance zone
            dx = abs(st["cx"] - target_cx)
            dy = abs(st["cy"] - target_cy)
            if dx > DEADBAND_X:
                target_cx = st["cx"]
            if dy > DEADBAND_Y:
                target_cy = st["cy"]
            target_zoom = base_zoom
        else:
            frames_since_face += 1
            # Hold the last speaker position - never steer away to empty center
            target_cx = last_known_cx
            target_cy = last_known_cy
            target_zoom = base_zoom

        # Exponential moving average with velocity limiting
        step_x = (target_cx - curr_cx) * ALPHA_POS
        step_x = max(-MAX_PAN_SPEED, min(step_x, MAX_PAN_SPEED))
        curr_cx += step_x

        step_y = (target_cy - curr_cy) * ALPHA_POS
        step_y = max(-MAX_PAN_SPEED, min(step_y, MAX_PAN_SPEED))
        curr_cy += step_y

        curr_zoom += (target_zoom - curr_zoom) * ALPHA_ZOOM

        # Calculate 9:16 vertical crop
        crop_h = float(vid_height) / curr_zoom
        crop_w = crop_h * 9.0 / 16.0

        if crop_w > vid_width:
            crop_w = float(vid_width)
            crop_h = crop_w * 16.0 / 9.0
            if crop_h > vid_height:
                crop_h = float(vid_height)
                crop_w = crop_h * 9.0 / 16.0

        # Horizontal: Centered on smoothed speaker position
        crop_x = curr_cx - crop_w / 2.0
        crop_x = max(0.0, min(crop_x, float(vid_width) - crop_w))

        # Vertical: Cinematic Eye-Level Rule (38% from top)
        # Places face at golden upper-third, leaving headroom & clear lower chest for subtitles
        crop_y = curr_cy - crop_h * 0.38
        crop_y = max(0.0, min(crop_y, float(vid_height) - crop_h))

        frame_crops.append({
            "ts": f_idx / float(fps),
            "w": int(crop_w), "h": int(crop_h),
            "x": int(crop_x), "y": int(crop_y)
        })

    # Step 6: Write sendcmd commands
    cmd_file = input_path + ".sendcmd"
    lines = []
    for c in frame_crops:
        ts_str = f"{c['ts']:.3f}"
        lines.append(f"{ts_str} crop w {c['w']}, crop h {c['h']}, crop x {c['x']}, crop y {c['y']};")

    with open(cmd_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    filter_cmd_file = cmd_file.replace('\\', '/').replace(':', '\\:')

    # Step 7: Apply sendcmd with FFmpeg scaling to 1080x1920
    filter_cx = (
      f"sendcmd=f='{filter_cmd_file}',"
      f"crop={int(vid_height*9/16)}:{vid_height}:0:0,"
      f"scale=1080:1920:force_original_aspect_ratio=decrease,"
      f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
    )

    ffmpeg_path = ffmpeg_utils.get_ffmpeg_exe()
    cmd = [
      ffmpeg_path,
      "-i", input_path,
      "-vf", filter_cx,
      "-c:v", "libx264",
      "-preset", "veryfast",
      "-crf", "22",
      "-c:a", "copy",
      "-movflags", "+faststart",
      output_path, "-y"
    ]

    result = subprocess.run(cmd, capture_output=True)

    if os.path.exists(cmd_file):
        os.remove(cmd_file)

    if result.returncode != 0:
        logger.error(f"FFmpeg dynamic sendcmd reframe failed: {result.stderr.decode()}")
        return False, overall_face_count

    return True, overall_face_count


async def _reframe_static_fallback(
  input_path: str,
  output_path: str,
  vid_width: int,
  vid_height: int,
  face_data: list[list[dict]]
) -> None:
  """
  Fallback static crop with cinematic eye-level composition.
  """
  all_faces = [f for sec_data in face_data for f in sec_data]

  if not all_faces:
      avg_cx = vid_width / 2.0
      avg_cy = vid_height / 2.0
  else:
      avg_cx = sum(f['cx'] for f in all_faces) / len(all_faces)
      avg_cy = sum(f['cy'] for f in all_faces) / len(all_faces)

  crop_h = float(vid_height)
  crop_w = crop_h * 9.0 / 16.0

  if crop_w > vid_width:
      crop_w = float(vid_width)
      crop_h = crop_w * 16.0 / 9.0

  crop_x = avg_cx - crop_w / 2.0
  crop_x = max(0.0, min(crop_x, float(vid_width) - crop_w))

  # Cinematic Eye-Level vertical framing
  crop_y = avg_cy - crop_h * 0.38
  crop_y = max(0.0, min(crop_y, float(vid_height) - crop_h))

  filter_cx = (
      f"crop={int(crop_w)}:{int(crop_h)}:{int(crop_x)}:{int(crop_y)},"
      f"scale=1080:1920:force_original_aspect_ratio=decrease,"
      f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2"
  )

  ffmpeg_path = ffmpeg_utils.get_ffmpeg_exe()
  cmd = [
    ffmpeg_path, "-i", input_path,
    "-vf", filter_cx,
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "22",
    "-c:a", "copy",
    "-movflags", "+faststart",
    output_path, "-y"
  ]
  result = subprocess.run(cmd, capture_output=True)
  if result.returncode != 0:
    raise RuntimeError(
      f"FFmpeg static fallback reframe failed: {result.stderr.decode()}"
    )


def _get_video_dimensions(video_path: str) -> tuple[int, int]:
  """
  Use ffprobe to get width and height.
  """
  ffprobe_path = ffmpeg_utils.get_ffprobe_exe()
  cmd = [
      ffprobe_path, 
      "-v", "error", "-select_streams", "v:0", 
      "-show_entries", "stream=width,height", "-of", "json", video_path
  ]
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
      raise RuntimeError(f"FFprobe failed: {result.stderr}")
  data = json.loads(result.stdout)
  stream = data["streams"][0]
  return stream["width"], stream["height"]

async def _cut_raw_clip(
  video_path: str,
  start: float,
  end: float,
  output_path: str
) -> None:
  """
  ffmpeg -ss {start} -i {video_path} ...
  """
  duration = end - start
  ffmpeg_path = ffmpeg_utils.get_ffmpeg_exe()
  cmd = [
      ffmpeg_path, "-y", "-ss", str(start), "-i", video_path,
      "-t", str(duration), "-c", "copy", output_path
  ]
  result = subprocess.run(cmd, capture_output=True, text=True)
  if result.returncode != 0:
      raise RuntimeError(f"FFmpeg failed cutting raw clip: {result.stderr}")
