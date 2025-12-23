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
import tempfile

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Video Generator (Deepgram + Groq Full Debug)",
    page_icon="🎬",
    layout="wide",
)

# =========================
# STOP FLAG
# =========================

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False


def request_stop():
    st.session_state.stop_requested = True


# =========================
# SECRETS
# =========================

try:
    DEEPGRAM_KEY = st.secrets["DEEPGRAM_API_KEY"]
except Exception:
    st.error("🚨 Please add DEEPGRAM_API_KEY in Streamlit secrets.")
    st.stop()

GROQ_KEY = st.secrets.get("GROQ_API_KEY", None)
if not GROQ_KEY:
    st.error("🚨 Please add GROQ_API_KEY in Streamlit secrets.")
    st.stop()

groq_client = Groq(api_key=GROQ_KEY)

# =========================
# TEMP DIR (/tmp – safe)
# =========================

BASE_TEMP_DIR = os.path.join(tempfile.gettempdir(), "ai_video_app")
os.makedirs(BASE_TEMP_DIR, exist_ok=True)
TEMP_DIR = BASE_TEMP_DIR


def clean_temp_dir():
    """Har run se pehle temp dir ke andar ki purani files clean kar do."""
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)
        return
    for name in os.listdir(TEMP_DIR):
        path = os.path.join(TEMP_DIR, name)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception:
            pass


# =========================
# UTILS
# =========================

def get_audio_duration(audio_path: str):
    """ffprobe se audio ki duration (seconds) nikal lo."""
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


# =========================
# DEEPGRAM: TRANSCRIPT + TIMESTAMPS
# =========================

def get_transcript_chunks(audio_path, audio_duration=None, max_gap=2.0):
    """
    Deepgram:
    - full transcript
    - word timestamps
    - un words ko chunks (sentences / segments) me group karein.

    Return:
        full_text (str),
        chunks = [
          {"index": 1, "text": "...", "start": 0.5, "end": 7.8},
          ...
        ],
        deepgram_raw (dict)
    """
    url = "https://api.deepgram.com/v1/listen"

    headers = {
        "Authorization": f"Token {DEEPGRAM_KEY}",
        "Content-Type": "audio/mpeg",
    }

    params = {
        "model": "nova-2",
        "smart_format": "true",
        "punctuate": "true",
        "utterances": "false",
        "diarize": "false",
    }

    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()

        resp = requests.post(url, headers=headers, params=params, data=audio_data)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"❌ Deepgram request failed: {e}")
        return "", [], None

    try:
        alt = data["results"]["channels"][0]["alternatives"][0]
    except Exception as e:
        st.error(f"❌ Unexpected Deepgram response format: {e}")
        st.write(data)
        return "", [], data

    full_text = alt.get("transcript", "").strip()
    words = alt.get("words", [])

    if not words:
        if not audio_duration:
            return full_text, [], data
        return full_text, [
            {
                "index": 1,
                "text": full_text,
                "start": 0.0,
                "end": float(audio_duration),
            }
        ], data

    chunks = []
    current_words = []
    current_start = words[0]["start"]
    prev_end = words[0]["end"]

    for w in words:
        word = w.get("word", "")
        start = w.get("start", prev_end)
        end = w.get("end", start)

        gap = start - prev_end

        # agar beech ka gap zyada ho to naya chunk shuru
        if gap > max_gap and current_words:
            chunk_text = " ".join(current_words).strip()
            chunks.append(
                {
                    "index": len(chunks) + 1,
                    "text": chunk_text,
                    "start": current_start,
                    "end": prev_end,
                }
            )
            current_words = [word]
            current_start = start
        else:
            current_words.append(word)

        prev_end = end

    if current_words:
        chunk_text = " ".join(current_words).strip()
        chunks.append(
            {
                "index": len(chunks) + 1,
                "text": chunk_text,
                "start": current_start,
                "end": prev_end,
            }
        )

    # last chunk ko audio_duration se align kar do
    if audio_duration and chunks:
        if chunks[-1]["end"] > audio_duration + 0.3:
            chunks[-1]["end"] = float(audio_duration)
        elif chunks[-1]["end"] < audio_duration - 0.5:
            chunks[-1]["end"] = float(audio_duration)

    return full_text, chunks, data


# =========================
# GROQ VISION: DESCRIBE IMAGES (BATCHES OF 5)
# =========================

def encode_upload_to_data_url(uploaded_file):
    """Uploaded image -> resized JPEG -> base64 data URL."""
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


def describe_all_images_with_groq(image_files, max_images=50):
    """
    Groq Vision se sab images ki short descriptions nikalta hai.
    - Batches of 5 images (model limit)
    - Max `max_images` images use karega

    Returns:
        desc_map: {filename: description}
        raw_responses: raw JSON objects list (for debug)
    """
    if not image_files:
        return {}, []

    # Hard cap
    image_files = image_files[:max_images]
    model_id = "meta-llama/llama-4-scout-17b-16e-instruct"

    desc_map = {}
    raw_responses = []

    batch_size = 5
    total = len(image_files)
    progress = st.progress(0)
    status = st.empty()

    for i in range(0, total, batch_size):
        if st.session_state.stop_requested:
            st.warning("⛔ Stopped by user during image description.")
            break

        batch = image_files[i : i + batch_size]
        status.text(f"👁️ Describing images {i+1} to {min(i+batch_size, total)}...")

        content = [
            {
                "type": "text",
                "text": (
                    "You will see several images. After that, return ONLY JSON:\n"
                    "{\n"
                    '  \"images\": [\n'
                    '    { \"index\": 1, \"filename\": \"...\", \"description\": \"...\" },\n'
                    '    { \"index\": 2, \"filename\": \"...\", \"description\": \"...\" }\n'
                    "  ]\n"
                    "}\n"
                    "Use very short English descriptions (one sentence)."
                ),
            }
        ]

        indexed_files = []
        for idx, uf in enumerate(batch, start=1):
            data_url = encode_upload_to_data_url(uf)
            indexed_files.append((idx, uf.name))
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": f"Image {idx} (filename: {uf.name})",
                }
            )

        try:
            resp = groq_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                max_completion_tokens=512,
                temperature=0.4,
            )
            text = resp.choices[0].message.content or "{}"
            data = json.loads(text)
            raw_responses.append(data)

            items = data.get("images", [])
            # map by filename
            for item in items:
                fn = item.get("filename")
                desc = item.get("description")
                if fn and desc:
                    desc_map[fn] = desc.strip()
        except Exception as e:
            st.warning(f"⚠️ Groq Vision batch failed: {e}")

        progress.progress(min((i + batch_size) / total, 1.0))

    return desc_map, raw_responses


# =========================
# GROQ TEXT: MAP CHUNKS -> IMAGES USING DESCRIPTIONS
# =========================

def map_chunks_to_images_using_descriptions(chunks, image_desc_map):
    """
    Text-only mapping:
    - chunks: [{index, text, start, end}]
    - image_desc_map: {filename: description}

    Groq ko:
      - chunks (brief)
      - images list (filename + description)
    bhejte hain, output JSON:
      { "mappings": [ { "chunk_index": 1, "image": "file.png" }, ... ] }

    Returns:
      cleaned_mappings, raw_data
    """
    if not chunks or not image_desc_map:
        return [], None

    model_id = "llama-3.1-8b-instant"

    # shorten chunks for prompt
    chunks_brief = []
    for c in chunks:
        text = c["text"]
        if len(text) > 300:
            text = text[:300] + "..."
        chunks_brief.append({"index": c["index"], "text": text})

    images_list = [
        {"filename": fn, "description": desc}
        for fn, desc in image_desc_map.items()
    ]

    prompt = f"""
You are aligning narration chunks with images.

CHUNKS:
{json.dumps(chunks_brief, ensure_ascii=False, indent=2)}

IMAGES:
{json.dumps(images_list, ensure_ascii=False, indent=2)}

TASK:
For each chunk, choose ONE image filename that best matches its meaning.
Distribute chunks sensibly across images (do NOT assign all chunks to the same image).

Return ONLY JSON in this exact format:
{{
  "mappings": [
    {{ "chunk_index": 1, "image": "filename1.png" }},
    {{ "chunk_index": 2, "image": "filename2.png" }}
  ]
}}

Rules:
- "chunk_index" must match one of the CHUNKS indexes.
- "image" must be exactly one of the filenames from IMAGES.
"""

    try:
        resp = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_completion_tokens=1024,
            temperature=0.3,
        )
        text = resp.choices[0].message.content or "{}"
        data = json.loads(text)
    except Exception as e:
        st.warning(f"⚠️ Groq text mapping failed: {e}")
        return [], None

    mappings = data.get("mappings")
    if not isinstance(mappings, list):
        return [], data

    valid_set = set(image_desc_map.keys())
    cleaned = []
    for m in mappings:
        if not isinstance(m, dict):
            continue
        idx = m.get("chunk_index")
        img = m.get("image")
        if isinstance(idx, int) and img in valid_set:
            cleaned.append({"chunk_index": idx, "image": img})

    return cleaned, data


# =========================
# TIMELINE BUILD
# =========================

def build_timeline_sequential(chunks, image_names, audio_duration=None, min_dur=0.5):
    """
    Simple deterministic fallback:
    - Duration = Deepgram timestamps
    - image = index-based rotation
    """
    if not chunks or not image_names:
        return []

    timeline = []
    n_images = len(image_names)

    for ch in chunks:
        dur = max(ch["end"] - ch["start"], min_dur)
        idx = ch["index"]
        img = image_names[(idx - 1) % n_images]
        timeline.append({"image": img, "duration_seconds": float(dur)})

    if audio_duration and audio_duration > 5:
        total_planned = sum(max(t["duration_seconds"], 0.1) for t in timeline)
        if total_planned > 0:
            scale = audio_duration / total_planned
            for t in timeline:
                t["duration_seconds"] = max(t["duration_seconds"] * scale, min_dur)

    return timeline


def build_timeline_with_mapping(chunks, mappings, image_names, audio_duration=None, min_dur=0.5):
    """
    Mapping-based:
    - If mapping missing / useless (all same image) → sequential fallback
    """
    if not chunks or not image_names:
        return []

    if not mappings:
        return build_timeline_sequential(chunks, image_names, audio_duration, min_dur)

    idx_to_img = {}
    for m in mappings:
        idx = m.get("chunk_index")
        img = m.get("image")
        if isinstance(idx, int) and img in image_names:
            idx_to_img[idx] = img

    used_images = set(idx_to_img.values())
    if len(used_images) <= 1 and len(image_names) > 1:
        # sab ko ek hi image de diya? bakwaas → fallback
        return build_timeline_sequential(chunks, image_names, audio_duration, min_dur)

    timeline = []
    n_images = len(image_names)

    for ch in chunks:
        dur = max(ch["end"] - ch["start"], min_dur)
        idx = ch["index"]
        img = idx_to_img.get(idx)
        if not img:
            img = image_names[(idx - 1) % n_images]
        timeline.append({"image": img, "duration_seconds": float(dur)})

    if audio_duration and audio_duration > 5:
        total_planned = sum(max(t["duration_seconds"], 0.1) for t in timeline)
        if total_planned > 0:
            scale = audio_duration / total_planned
            for t in timeline:
                t["duration_seconds"] = max(t["duration_seconds"] * scale, min_dur)

    return timeline


# =========================
# VIDEO RENDERING
# =========================

def render_video(timeline, audio_path, image_map, output_name="final_video.mp4"):
    """
    OpenCV + FFmpeg:
    - frames from images
    - merge with audio
    """
    if not timeline:
        st.error("No timeline provided.")
        return None

    width, height = 854, 480
    fps = 24

    temp_video = os.path.join(TEMP_DIR, "temp_silent.mp4")
    final_output = os.path.join(TEMP_DIR, output_name)

    temp_video_abs = os.path.abspath(temp_video)
    audio_abs = os.path.abspath(audio_path)
    final_output_abs = os.path.abspath(final_output)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_abs, fourcc, fps, (width, height))

    st.write("🎥 Rendering video frames...")
    progress_bar = st.progress(0)
    percent_text = st.empty()

    total_sec = sum(t.get("duration_seconds", 0) for t in timeline)
    total_frames_all = max(int(total_sec * fps), 1)
    current_frame = 0

    for item in timeline:
        if st.session_state.stop_requested:
            st.warning("⛔ Stopped during rendering.")
            out.release()
            return None

        img_name = item.get("image")
        duration = float(item.get("duration_seconds", 0) or 0)

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
                st.warning("⛔ Stopped during rendering.")
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
        st.warning("⛔ Stopped before audio merge.")
        return None

    if not os.path.exists(audio_abs):
        st.error(f"❌ Audio file not found for FFmpeg: {audio_abs}")
        return None
    if not os.path.exists(temp_video_abs):
        st.error(f"❌ Temp video not found: {temp_video_abs}")
        return None

    st.write("🎵 Merging audio with video (FFmpeg)...")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        temp_video_abs,
        "-i",
        audio_abs,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        final_output_abs,
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    if result.returncode != 0:
        st.error("❌ FFmpeg failed while merging audio and video.")
        st.code(result.stdout[-2000:])
        return None

    if not os.path.exists(final_output_abs):
        st.error("❌ Final video file was not created.")
        st.code(result.stdout[-2000:])
        return None

    if os.path.getsize(final_output_abs) < 1024:
        st.error("❌ Final video too small / corrupted.")
        st.code(result.stdout[-2000:])
        return None

    return final_output_abs


# =========================
# UI
# =========================

st.title("🤖 AI Video Generator (Deepgram + Groq, Full Debug)")
st.markdown(
    """
**Pipeline:**

1. Deepgram → voiceover ko **chunks + timestamps** me todta hai  
2. Groq Vision → images ko batches of 5 me analyze karke **descriptions** banata hai (max ~50 images)  
3. Groq Text → Deepgram chunks + image descriptions se **JSON mapping** banata hai (`chunk_index → image filename`)  
4. Timeline → Deepgram ke time se **100% audio sync**, Groq se **image choice**  
5. OpenCV + FFmpeg → final video render, zoom effect ke sath  

Neeche expanders me tum **har step ki JSON / data** dekh sakte ho:
- Deepgram raw JSON
- Chunks list
- Image descriptions (Groq Vision)
- Mapping JSON (Groq Text)
- Final timeline
"""
)

col1, col2 = st.columns(2)
with col1:
    audio_file = st.file_uploader("Upload voiceover (MP3)", type=["mp3"])
with col2:
    uploaded_images = st.file_uploader(
        "Upload images (max ~50, order ≈ story flow)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

st.button("🛑 Stop processing", on_click=request_stop)

if st.button("🚀 Generate video", type="primary"):
    st.session_state.stop_requested = False

    if not audio_file or not uploaded_images:
        st.error("Please upload both audio and images first.")
    else:
        # optional: cap images to e.g. 50
        if len(uploaded_images) > 50:
            st.warning(
                f"⚠️ {len(uploaded_images)} images uploaded. "
                "Using only the first 50 for analysis + mapping."
            )
            uploaded_images = uploaded_images[:50]

        status = st.status("Starting process...", expanded=True)

        deepgram_raw = None
        image_desc_raw_list = []
        image_desc_map = {}
        mapping_raw = None
        mappings = []
        timeline = []

        # 0) Clean temp
        status.write("🧹 Cleaning temp folder...")
        clean_temp_dir()

        # 1) Save audio
        status.write("💾 Saving audio file...")
        local_audio = os.path.join(TEMP_DIR, "input.mp3")
        local_audio = os.path.abspath(local_audio)
        with open(local_audio, "wb") as f:
            f.write(audio_file.getbuffer())

        if os.path.exists(local_audio):
            size_kb = os.path.getsize(local_audio) / 1024
            st.info(f"Audio saved: {local_audio} ({size_kb:.1f} KB)")
        else:
            st.error("❌ Failed to save audio file.")
            st.stop()

        audio_duration = get_audio_duration(local_audio)
        if audio_duration:
            st.info(f"⏱️ Audio duration (ffprobe): ~{audio_duration:.2f} seconds")

        if st.session_state.stop_requested:
            st.warning("⛔ Stopped by user.")
            st.stop()

        # 2) Deepgram
        status.write("👂 Getting transcript + timestamps (Deepgram)...")
        full_text, chunks, deepgram_raw = get_transcript_chunks(
            local_audio, audio_duration=audio_duration
        )

        image_names = [img.name for img in uploaded_images]

        if not chunks:
            status.write("⚠️ No chunks from Deepgram, falling back to simple slideshow.")
            dummy_chunks = []
            if audio_duration and image_names:
                # ek simple chunk per image
                avg_dur = audio_duration / max(len(image_names), 1)
                t = 0.0
                for i, name in enumerate(image_names, start=1):
                    dummy_chunks.append(
                        {"index": i, "text": "", "start": t, "end": t + avg_dur}
                    )
                    t += avg_dur
            chunks = dummy_chunks

        if st.session_state.stop_requested:
            st.warning("⛔ Stopped by user.")
            st.stop()

        # 3) Groq Vision: all image descriptions (batches of 5)
        status.write("👁️ Describing ALL images with Groq Vision (batches of 5)...")
        image_desc_map, image_desc_raw_list = describe_all_images_with_groq(
            uploaded_images
        )

        if st.session_state.stop_requested:
            st.warning("⛔ Stopped by user.")
            st.stop()

        # 4) Groq Text: mapping using descriptions
        if image_desc_map:
            status.write("🧠 Mapping chunks → images using descriptions (Groq Text)...")
            mappings, mapping_raw = map_chunks_to_images_using_descriptions(
                chunks, image_desc_map
            )
            status.write("🧩 Building final timeline (mapping + timestamps)...")
            timeline = build_timeline_with_mapping(
                chunks, mappings, image_names, audio_duration=audio_duration
            )
        else:
            status.write(
                "⚠️ No image descriptions available, using sequential mapping only."
            )
            timeline = build_timeline_sequential(
                chunks, image_names, audio_duration=audio_duration
            )

        if st.session_state.stop_requested:
            st.warning("⛔ Stopped by user.")
            st.stop()

        # 5) Render video
        status.write("🎬 Rendering video with OpenCV + FFmpeg...")
        image_map = {img.name: img for img in uploaded_images}
        final_vid_path = render_video(timeline, local_audio, image_map)

        if final_vid_path is None or st.session_state.stop_requested:
            st.warning("⛔ Processing stopped before completion or video failed.")
            st.stop()

        status.update(label="✅ Video rendered!", state="complete", expanded=False)
        if audio_duration:
            st.success(f"Video rendered! (≈ {int(audio_duration)} seconds)")
        else:
            st.success("Video rendered! You can watch or download it below.")

        # Show + download
        try:
            with open(final_vid_path, "rb") as f:
                video_bytes = f.read()
        except Exception as e:
            st.error(f"❌ Could not read final video file: {e}")
            st.stop()

        st.video(video_bytes)

        st.download_button(
            "⬇️ Download video",
            data=video_bytes,
            file_name="AI_Gen_Video.mp4",
            mime="video/mp4",
        )

        # =========================
        # DEBUG / INSPECT SECTIONS
        # =========================

        with st.expander("🧩 Deepgram Output (raw JSON + chunks + transcript)"):
            st.subheader("Raw Deepgram JSON (full)")
            st.json(deepgram_raw)
            st.subheader("Transcript (combined text)")
            st.write(full_text)
            st.subheader("Chunks (index, text, start, end)")
            st.json(chunks)

        with st.expander("👁️ Groq Vision – Image Descriptions"):
            st.markdown("**Raw responses per batch (JSON exactly as Groq returned):**")
            for idx, raw in enumerate(image_desc_raw_list, start=1):
                st.markdown(f"**Batch {idx}**")
                st.json(raw)
            st.subheader("Final aggregated map: filename → description")
            st.json(image_desc_map)

        with st.expander("🧠 Groq Text – Mapping + Timeline"):
            st.subheader("Raw mapping JSON from Groq Text (as is)")
            st.json(mapping_raw)
            st.subheader("Cleaned mappings (chunk_index → image filename)")
            st.json(mappings)
            st.subheader("Final timeline used for rendering (image + duration_seconds)")
            st.json(timeline)
