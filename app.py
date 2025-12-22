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

# ------------------ PAGE CONFIG ------------------

st.set_page_config(
    page_title="AI Video Generator (Deepgram + Groq Vision Mapping)",
    page_icon="🎬",
    layout="wide",
)

# ------------------ STOP FLAG ------------------

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False


def request_stop():
    st.session_state.stop_requested = True


# ------------------ SECRETS ------------------

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

# ------------------ TEMP DIR (/tmp – safe for Streamlit Cloud) ------------------

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


# ------------------ UTILS ------------------


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


# ------------------ DEEPGRAM: TRANSCRIPT + TIMESTAMPS ------------------


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


# ------------------ GROQ VISION MAPPING (images actually seen) ------------------


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


def map_chunks_to_images_groq_vision(chunks, uploaded_images):
    """
    Yahan Groq ko mapping ke waqt bhi **images dikhai ja rahi hain**.

    Input:
      - chunks: [{index, text, start, end}, ...]
      - uploaded_images: Streamlit UploadedFile list

    Prompt structure:
      - Pehle instruction + images (Image 1, Image 2, ...)
      - Phir CHUNKS ka JSON (sirf index + text)
      - Output: JSON { "mappings": [ {"chunk_index": 1, "image_index": 2}, ... ] }

    Return:
      - cleaned_mappings = [ {"chunk_index": int, "image": real_filename}, ... ]
      - raw_response_json (for debug)
    """
    if not chunks or not uploaded_images:
        return [], None

    model_id = "meta-llama/llama-4-scout-17b-16e-instruct"

    # Chunks ko thoda chhota rakhte hain prompt ke liye
    chunks_brief = []
    for c in chunks:
        text = c["text"]
        if len(text) > 300:  # 300 characters enough for context
            text = text[:300] + "..."
        chunks_brief.append({"index": c["index"], "text": text})

    # Content build: instruction + images + chunks
    content = [
        {
            "type": "text",
            "text": (
                "You will see several images identified as 'Image 1', 'Image 2', etc.\n"
                "Then you will see a list of narration chunks with indexes.\n\n"
                "TASK:\n"
                "- For each chunk, choose ONE image index that best matches the meaning of the chunk.\n"
                "- Distribute chunks sensibly across images (do NOT assign all chunks to the same image).\n\n"
                "Return ONLY JSON in this exact format:\n"
                "{\n"
                '  "mappings": [\n'
                '    { "chunk_index": 1, "image_index": 2 },\n'
                '    { "chunk_index": 2, "image_index": 1 }\n'
                "  ]\n"
                "}\n"
                "Where:\n"
                "- chunk_index is from the CHUNKS list.\n"
                "- image_index is the number from 'Image 1', 'Image 2', etc.\n"
            ),
        }
    ]

    # Images section
    indexed_files = []
    for idx, uf in enumerate(uploaded_images, start=1):
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
                "text": f"Image {idx}",
            }
        )

    # Chunks section as text
    content.append(
        {
            "type": "text",
            "text": "CHUNKS:\n" + json.dumps(chunks_brief, ensure_ascii=False, indent=2),
        }
    )

    try:
        resp = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_completion_tokens=1024,
            temperature=0.3,
        )
        text = resp.choices[0].message.content or "{}"
        data = json.loads(text)
    except Exception as e:
        st.warning(f"⚠️ Groq vision mapping failed: {e}")
        return [], None

    mappings = data.get("mappings")
    if not isinstance(mappings, list):
        return [], data

    # Map image_index -> actual filename
    index_to_name = {i: name for i, name in indexed_files}
    cleaned = []
    for m in mappings:
        if not isinstance(m, dict):
            continue
        idx = m.get("chunk_index")
        img_idx = m.get("image_index")
        if not isinstance(idx, int) or not isinstance(img_idx, int):
            continue
        filename = index_to_name.get(img_idx)
        if filename:
            cleaned.append({"chunk_index": idx, "image": filename})

    return cleaned, data


# ------------------ TIMELINE BUILD ------------------


def build_timeline_from_chunks_sequential(
    chunks, image_names, audio_duration=None, min_dur=0.5
):
    """
    Simple deterministic mapping:
    - Durations = Deepgram timestamps (end - start)
    - Image = chunk index ke hisaab se sequential:
        chunk 1 -> image 1
        chunk 2 -> image 2
        ...
        chunk N -> image (N % len(images))
    """
    if not chunks or not image_names:
        return []

    timeline = []
    n_images = len(image_names)

    for chunk in chunks:
        dur = max(chunk["end"] - chunk["start"], min_dur)
        idx = chunk["index"]
        img = image_names[(idx - 1) % n_images]
        timeline.append({"image": img, "duration_seconds": float(dur)})

    # Scale to match audio duration
    if audio_duration and audio_duration > 5:
        total_planned = sum(max(t["duration_seconds"], 0.1) for t in timeline)
        if total_planned > 0:
            scale = audio_duration / total_planned
            for t in timeline:
                t["duration_seconds"] = max(t["duration_seconds"] * scale, min_dur)

    return timeline


def build_timeline_from_chunks_with_mapping(
    chunks, mappings, image_names, audio_duration=None, min_dur=0.5
):
    """
    Groq ke mapping result se timeline banata hai.
    Agar mapping useless ho (sab same image / empty) to sequential fallback.
    """
    if not chunks or not image_names:
        return []

    # Koi mapping nahi mila
    if not mappings:
        return build_timeline_from_chunks_sequential(
            chunks, image_names, audio_duration, min_dur
        )

    idx_to_img = {}
    for m in mappings:
        idx = m.get("chunk_index")
        img = m.get("image")
        if isinstance(idx, int) and img in image_names:
            idx_to_img[idx] = img

    used_images = set(idx_to_img.values())
    # agar sirf 1 image hi use hui -> useless mapping => sequential
    if len(used_images) <= 1 and len(image_names) > 1:
        return build_timeline_from_chunks_sequential(
            chunks, image_names, audio_duration, min_dur
        )

    timeline = []
    n_images = len(image_names)

    for chunk in chunks:
        dur = max(chunk["end"] - chunk["start"], min_dur)
        idx = chunk["index"]
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


# ------------------ VIDEO RENDERING ------------------


def render_video(timeline, audio_path, image_map, output_name="final_video.mp4"):
    """
    OpenCV + FFmpeg: images -> video frames -> merge with audio.
    Absolute paths + checks to avoid 'No such file' errors.
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
            st.warning("⛔ Processing stopped by user during rendering.")
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
                st.warning("⛔ Processing stopped by user during rendering.")
                out.release()
                return None

            # light zoom effect
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

    if not os.path.exists(audio_abs):
        st.error(f"❌ Audio file not found for FFmpeg: {audio_abs}")
        return None
    if not os.path.exists(temp_video_abs):
        st.error(f"❌ Temp video not found for FFmpeg: {temp_video_abs}")
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
        st.error("❌ Final video seems too small / corrupted.")
        st.code(result.stdout[-2000:])
        return None

    return final_output_abs


# ------------------ UI ------------------

st.title("🤖 AI Video Generator (Deepgram + Groq Vision Mapping)")
st.markdown(
    """
**Pipeline:**

1. Deepgram → audio ko **chunks + timestamps** me todta hai  
2. Groq Vision (mapping step) → **images ko dekh kar** har chunk ke liye image choose karta hai  
3. Agar mapping bakwaas ho (sab chunks same image) → **auto sequential fallback**  
4. OpenCV + FFmpeg → exact durations ke saath video render  

Neeche debug expanders me tum 1–1 cheez dekh sakte ho:
- Deepgram ka raw JSON + chunks
- Groq ka raw mapping JSON
- Clean mapping (chunk_index → image_index → filename)
- Final timeline (image + duration_seconds)
"""
)

col1, col2 = st.columns(2)
with col1:
    audio_file = st.file_uploader("Upload voiceover (MP3)", type=["mp3"])
with col2:
    uploaded_images = st.file_uploader(
        "Upload images (order ≈ story flow)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

st.button("🛑 Stop processing", on_click=request_stop)

if st.button("🚀 Generate video", type="primary"):
    st.session_state.stop_requested = False

    if not audio_file or not uploaded_images:
        st.error("Please upload both audio and images first.")
    else:
        status = st.status("Starting process...", expanded=True)

        deepgram_raw = None
        groq_mapping_raw = None
        mappings = []

        # 0. clean temp dir
        status.write("🧹 Cleaning temp folder...")
        clean_temp_dir()

        # 1. Save audio
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

        if st.session_state.stop_requested:
            st.warning("⛔ Processing stopped by user.")
            st.stop()

        # 2. Deepgram chunks
        status.write("👂 Getting transcript + timestamps (Deepgram)...")
        full_text, chunks, deepgram_raw = get_transcript_chunks(
            local_audio, audio_duration=audio_duration
        )

        valid_image_names = [img.name for img in uploaded_images]

        # 3. Mapping + timeline
        if not chunks:
            status.write("⚠️ No chunks from Deepgram, using sequential slideshow.")
            plan = build_timeline_from_chunks_sequential(
                [{"index": i + 1, "text": "", "start": 0, "end": audio_duration or 5}
                 for i in range(len(valid_image_names))],
                valid_image_names,
                audio_duration=audio_duration,
            )
        else:
            status.write("🧠 Mapping chunks to images with Groq (using actual images)...")
            mappings, groq_mapping_raw = map_chunks_to_images_groq_vision(
                chunks, uploaded_images
            )

            status.write("🧩 Building final timeline...")
            plan = build_timeline_from_chunks_with_mapping(
                chunks,
                mappings,
                valid_image_names,
                audio_duration=audio_duration,
            )

        if st.session_state.stop_requested:
            st.warning("⛔ Processing stopped by user.")
            st.stop()

        # 4. Render video
        status.write("🎬 Rendering video with OpenCV...")
        image_map = {img.name: img for img in uploaded_images}
        final_vid_path = render_video(plan, local_audio, image_map)

        if final_vid_path is None or st.session_state.stop_requested:
            st.warning(
                "⛔ Processing stopped before completion or video failed to render."
            )
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

        # --------- DEBUG SECTIONS ---------

        with st.expander("🧩 Deepgram Output (raw + chunks + transcript)"):
            if deepgram_raw is not None:
                st.subheader("Raw Deepgram JSON (trimmed)")
                st.json(deepgram_raw)
            st.subheader("Transcript (combined)")
            st.write(full_text)
            st.subheader("Chunks (index, text, start, end)")
            st.json(chunks)

        with st.expander("🧠 Groq Mapping + Final Timeline"):
            st.subheader("Groq raw mapping response (if any)")
            st.json(groq_mapping_raw)
            st.subheader("Cleaned mappings (chunk_index → image filename)")
            st.json(mappings)
            st.subheader("Final timeline (image + duration_seconds)")
            st.json(plan)
