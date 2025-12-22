import streamlit as st
import cv2
import numpy as np
from groq import Groq
from PIL import Image
import json
import subprocess
import os
import shutil
import requests
import base64
from io import BytesIO

# ------------------ CONFIG & GLOBALS ------------------

st.set_page_config(page_title="AI Video Generator (Groq)", page_icon="🎬", layout="wide")

# stop flag
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False


def request_stop():
    st.session_state.stop_requested = True


# ------------- LOAD SECRETS -------------

try:
    DEEPGRAM_KEY = st.secrets["DEEPGRAM_API_KEY"]
    GROQ_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error(
        "🚨 Secrets missing! Please configure DEEPGRAM_API_KEY and GROQ_API_KEY "
        "in your Streamlit secrets."
    )
    st.stop()

# Groq client
groq_client = Groq(api_key=GROQ_KEY)

# ------------- TEMP DIR -------------

TEMP_DIR = "temp_processing"
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR, exist_ok=True)

# ------------------ UTILS ------------------


def get_audio_duration(audio_path: str):
    """
    ffprobe se audio ki duration (seconds) nikal lo.
    Agar ffprobe na chale to None return karega.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                audio_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


# ------------------ DEEPGRAM (AUDIO → TEXT) ------------------


def get_transcript(audio_path):
    """
    Deepgram REST API se audio -> text.
    """
    url = "https://api.deepgram.com/v1/listen"

    headers = {
        "Authorization": f"Token {DEEPGRAM_KEY}",
        "Content-Type": "audio/mpeg",  # MP3
    }

    params = {
        "model": "nova-2",
        "smart_format": "true",
    }

    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        resp = requests.post(url, headers=headers, params=params, data=audio_data)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"❌ Deepgram request failed: {e}")
        return []

    try:
        alt = data["results"]["channels"][0]["alternatives"][0]
    except Exception as e:
        st.error(f"❌ Unexpected Deepgram response format: {e}")
        st.write(data)
        return []

    text = alt.get("transcript", "").strip()
    if not text:
        return []

    words = alt.get("words", [])
    start = words[0]["start"] if words else 0.0
    end = words[-1]["end"] if words else 0.0

    return [{"text": text, "start": start, "end": end}]


# ------------------ GROQ VISION HELPERS ------------------


def encode_upload_to_data_url(uploaded_file):
    """
    UploadedFile -> resized JPEG -> base64 data URL (<=4MB target).
    """
    uploaded_file.seek(0)
    img = Image.open(uploaded_file).convert("RGB")
    max_side = 1280
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        img = img.resize((int(w * scale), int(h * scale)))

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def analyze_images_with_groq(image_files):
    """
    Groq vision (Llama 4 Scout) se image descriptions.
    Max 5 images per request, batching.
    Returns: dict {filename: description}
    """
    if not image_files:
        return {}

    model_id = "meta-llama/llama-4-scout-17b-16e-instruct"

    descriptions = {}
    batch_size = 5
    total = len(image_files)
    progress = st.progress(0)
    status = st.empty()

    for i in range(0, total, batch_size):
        if st.session_state.stop_requested:
            st.warning("⛔ Processing stopped by user during image analysis.")
            break

        batch = image_files[i : i + batch_size]
        status.text(f"👁️ Analyzing images {i+1} to {min(i+batch_size, total)} with Groq...")

        content = [
            {
                "type": "text",
                "text": (
                    "You will receive several images. For each image, you will see a "
                    "filename text like 'Filename: xyz.jpg' followed by the image. "
                    "Return ONLY JSON in this format:\n"
                    '{ "images": [ { "filename": "file1.jpg", "description": "..." }, '
                    '{ "filename": "file2.png", "description": "..." } ] }'
                ),
            }
        ]

        for uf in batch:
            data_url = encode_upload_to_data_url(uf)
            content.append({"type": "text", "text": f"Filename: {uf.name}"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )

        try:
            resp = groq_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                max_completion_tokens=1024,
            )
            text = resp.choices[0].message.content or "{}"
            data = json.loads(text)
            for item in data.get("images", []):
                fn = item.get("filename")
                desc = item.get("description")
                if fn and desc:
                    descriptions[fn] = desc
        except Exception as e:
            st.warning(f"Groq vision batch failed, will fallback to slideshow. Error: {e}")
            break

        progress.progress(min((i + batch_size) / total, 1.0))

    return descriptions


# ------------------ GROQ TEXT (TIMELINE PLAN) ------------------


def create_sync_plan_with_groq(script_text, image_desc_map):
    """
    Groq text model se JSON timeline (image + duration).
    Raw shape ko normalize_clean_plan() me further clean karein.
    """
    if not image_desc_map:
        return []

    model_id = "llama-3.1-8b-instant"

    desc_json = json.dumps(image_desc_map)

    prompt = f"""
SCRIPT:
\"\"\"{script_text}\"\"\"

IMAGE_DESCRIPTIONS (JSON mapping filename -> description):
{desc_json}

TASK:
Create a JSON timeline that maps images to the script based on context.

Desired JSON format:
[
  {{ "image": "filename1.jpg", "duration_seconds": 5.0 }},
  {{ "image": "filename2.png", "duration_seconds": 4.0 }}
]

Rules:
- Only return valid JSON.
- JSON may be either a list directly, or an object with a top-level key
  like "timeline" or "scenes" that contains the list.
- "image" must be one of the provided filenames.
- "duration_seconds" must be a positive number.
- Rough total duration should match the length of the script.
"""

    try:
        resp = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_completion_tokens=1024,
            temperature=0.2,
        )
        text = resp.choices[0].message.content or "[]"
        data = json.loads(text)
    except Exception as e:
        st.warning(f"⚠️ Groq planning failed, using fallback slideshow. Error: {e}")
        return []

    return data


def build_slideshow_timeline(valid_image_names, audio_duration=None, script_text=None):
    """
    Simple fallback: har image ko equal duration.
    """
    if not valid_image_names:
        return []

    if audio_duration and audio_duration > 1:
        dur = audio_duration / len(valid_image_names)
    else:
        # crude estimation based on script length
        words = len(script_text.split()) if script_text else 120
        total_guess = max(words / 2.4, 30)  # ~2.4 wps, min 30s
        dur = total_guess / len(valid_image_names)

    dur = max(dur, 1.0)  # at least 1 second per image
    return [{"image": name, "duration_seconds": float(dur)} for name in valid_image_names]


def normalize_and_scale_timeline(
    raw_plan, valid_image_names, audio_duration=None, script_text=None
):
    """
    - Raw Groq plan ko clean karo.
    - Sirf valid filenames rakho.
    - Durations ko float + positive banao.
    - Phir total ko audio_duration se match karne ke liye scale karo.
    """
    valid_set = set(valid_image_names)
    if not valid_set:
        return []

    # 1) Raw shape -> list of dicts
    if isinstance(raw_plan, list):
        raw_list = raw_plan
    elif isinstance(raw_plan, dict):
        if "timeline" in raw_plan and isinstance(raw_plan["timeline"], list):
            raw_list = raw_plan["timeline"]
        elif "scenes" in raw_plan and isinstance(raw_plan["scenes"], list):
            raw_list = raw_plan["scenes"]
        else:
            raw_list = raw_plan.get("items", [])
            if not isinstance(raw_list, list):
                raw_list = []
    else:
        raw_list = []

    cleaned = []
    for entry in raw_list:
        if not isinstance(entry, dict):
            continue
        img = entry.get("image") or entry.get("filename")
        if not img or img not in valid_set:
            continue
        dur = (
            entry.get("duration_seconds")
            or entry.get("duration")
            or entry.get("seconds")
            or 5
        )
        try:
            dur = float(dur)
        except Exception:
            dur = 5.0
        if dur <= 0:
            dur = 5.0
        cleaned.append({"image": img, "duration_seconds": dur})

    # Agar sab filter ho gaye / kuch bhi nahi bacha:
    if not cleaned:
        return build_slideshow_timeline(valid_image_names, audio_duration, script_text)

    # 2) Audio length se match karne ke liye scale
    if audio_duration and audio_duration > 5:
        total_planned = sum(max(c["duration_seconds"], 0.1) for c in cleaned)
        if total_planned > 0:
            scale = audio_duration / total_planned
            for c in cleaned:
                c["duration_seconds"] = max(c["duration_seconds"] * scale, 0.5)

    return cleaned


# ------------------ VIDEO RENDERING ------------------


def render_video(timeline, audio_path, image_map, output_name="final_video.mp4"):
    """
    OpenCV + FFmpeg: images -> video frames -> merge with audio.
    Shows percentage progress, respects stop flag.
    """
    if not timeline:
        image_names = list(image_map.keys())
        if not image_names:
            st.error("No images available for rendering.")
            return None
        timeline = [{"image": name, "duration_seconds": 5.0} for name in image_names]

    width, height = 854, 480
    fps = 24

    temp_video = os.path.join(TEMP_DIR, "temp_silent.mp4")
    final_output = os.path.join(TEMP_DIR, output_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    st.write("🎥 Rendering video frames...")
    progress_bar = st.progress(0)
    percent_text = st.empty()

    total_sec = 0.0
    for t in timeline:
        if isinstance(t, dict):
            try:
                total_sec += float(t.get("duration_seconds", 0))
            except Exception:
                continue
    total_frames_all = max(int(total_sec * fps), 1)
    current_frame = 0

    for item in timeline:
        if st.session_state.stop_requested:
            st.warning("⛔ Processing stopped by user during rendering.")
            out.release()
            return None

        if not isinstance(item, dict):
            continue

        img_name = item.get("image")
        try:
            duration = float(item.get("duration_seconds", 0))
        except Exception:
            duration = 5.0

        if not img_name or img_name not in image_map or duration <= 0:
            continue

        image_map[img_name].seek(0)
        file_bytes = np.asarray(bytearray(image_map[img_name].read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is None:
            continue

        img = cv2.resize(img, (width, height))
        frames_in_clip = max(int(duration * fps), 1)

        for i in range(frames_in_clip):
            if st.session_state.stop_requested:
                st.warning("⛔ Processing stopped by user during rendering.")
                out.release()
                return None

            scale = 1.0 + (0.05 * i / frames_in_clip)
            M = cv2.getRotationMatrix2D((width // 2, height // 2), 0, scale)
            zoomed = cv2.warpAffine(img, M, (width, height))

            out.write(zoomed)
            current_frame += 1

            if current_frame % 20 == 0:
                frac = min(current_frame / float(total_frames_all), 1.0)
                progress_bar.progress(frac)
                percent_text.text(f"Rendering: {int(frac * 100)}%")

    out.release()

    if st.session_state.stop_requested:
        st.warning("⛔ Processing stopped before audio merge.")
        return None

    st.write("🎵 Merging audio with video...")
    command = (
        f'ffmpeg -y -i "{temp_video}" -i "{audio_path}" '
        f'-c:v copy -c:a aac -shortest "{final_output}"'
    )
    subprocess.run(command, shell=True)

    return final_output


# ------------------ UI ------------------

st.title("🤖 Free AI Video Generator (Groq)")
st.markdown(
    "Stack: **Deepgram** (audio) + **Groq Llama 4 Scout** (vision) + "
    "**Llama 3.1 8B** (logic) + **OpenCV** (render)"
)

col1, col2 = st.columns(2)
with col1:
    audio_file = st.file_uploader("Upload voiceover (MP3)", type=["mp3"])
with col2:
    uploaded_images = st.file_uploader(
        "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

st.button("🛑 Stop processing", on_click=request_stop)

if st.button("🚀 Generate video", type="primary"):
    st.session_state.stop_requested = False  # reset flag

    if not audio_file or not uploaded_images:
        st.error("Please upload both audio and images first.")
    else:
        status = st.status("Starting process...", expanded=True)

        # 1. Save audio
        status.write("💾 Saving audio file...")
        local_audio = os.path.join(TEMP_DIR, "input.mp3")
        with open(local_audio, "wb") as f:
            f.write(audio_file.getbuffer())

        audio_duration = get_audio_duration(local_audio)

        if st.session_state.stop_requested:
            st.warning("⛔ Processing stopped by user.")
            st.stop()

        # 2. Transcribe
        status.write("👂 Transcribing audio with Deepgram...")
        transcript = get_transcript(local_audio)
        if not transcript:
            st.error("❌ Could not get transcript from audio.")
            st.stop()

        full_text = " ".join([t["text"] for t in transcript])

        if st.session_state.stop_requested:
            st.warning("⛔ Processing stopped by user.")
            st.stop()

        # 3. Analyze images (Groq vision)
        status.write("👁️ Analyzing images with Groq Vision...")
        image_desc_map = analyze_images_with_groq(uploaded_images)

        if st.session_state.stop_requested:
            st.warning("⛔ Processing stopped by user.")
            st.stop()

        # 4. Plan scenes (Groq text)
        status.write("🧠 Planning scenes with Groq Text...")
        raw_plan = create_sync_plan_with_groq(full_text, image_desc_map)

        valid_image_names = [img.name for img in uploaded_images]
        plan = normalize_and_scale_timeline(
            raw_plan,
            valid_image_names,
            audio_duration=audio_duration,
            script_text=full_text,
        )

        if not plan:
            status.write("⚠️ AI plan empty, using simple slideshow.")
            plan = build_slideshow_timeline(
                valid_image_names, audio_duration, script_text=full_text
            )

        if st.session_state.stop_requested:
            st.warning("⛔ Processing stopped by user.")
            st.stop()

        # 5. Render
        status.write("🎬 Rendering video with OpenCV...")
        image_map = {img.name: img for img in uploaded_images}
        final_vid_path = render_video(plan, local_audio, image_map)

        if final_vid_path is None or st.session_state.stop_requested:
            st.warning("⛔ Processing stopped before completion.")
            st.stop()

        status.update(label="✅ Video rendered!", state="complete", expanded=False)
        if audio_duration:
            st.success(f"Video rendered! (audio ≈ {int(audio_duration)} seconds)")
        else:
            st.success("Video rendered! You can watch or download it below.")

        # Play in app
        st.video(final_vid_path)

        # Download button
        with open(final_vid_path, "rb") as f:
            video_bytes = f.read()

        st.download_button(
            "⬇️ Download video",
            data=video_bytes,
            file_name="AI_Gen_Video.mp4",
            mime="video/mp4",
        )
