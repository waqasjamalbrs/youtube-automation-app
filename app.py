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

st.set_page_config(
    page_title="AI Video Generator (Deepgram + Groq Scene Sync)",
    page_icon="🎬",
    layout="wide",
)

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


# ------------------ DEEPGRAM: TRANSCRIPT + TIMESTAMPS ------------------


def get_transcript_chunks(audio_path, audio_duration=None, max_gap=2.0):
    """
    Deepgram REST API se:
    - full transcript text
    - word-level timestamps
    - un words ko small chunks (sentences/segments) me group karna

    Return:
        full_text (str),
        chunks = [
          {"index": 1, "text": "...", "start": 0.5, "end": 7.8},
          ...
        ]
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
        return "", []

    try:
        alt = data["results"]["channels"][0]["alternatives"][0]
    except Exception as e:
        st.error(f"❌ Unexpected Deepgram response format: {e}")
        st.write(data)
        return "", []

    full_text = alt.get("transcript", "").strip()
    words = alt.get("words", [])

    if not words:
        # koi words nahi → ek big chunk (0 → audio_duration)
        if not audio_duration:
            return full_text, []
        return full_text, [
            {
                "index": 1,
                "text": full_text,
                "start": 0.0,
                "end": float(audio_duration),
            }
        ]

    chunks = []
    current_words = []
    current_start = words[0]["start"]
    prev_end = words[0]["end"]

    for w in words:
        word = w.get("word", "")
        start = w.get("start", prev_end)
        end = w.get("end", start)

        gap = start - prev_end

        # naya chunk agar gap zyada hai
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

    # last chunk ko slightly adjust kar do agar audio_duration pata ho
    if audio_duration and chunks:
        if chunks[-1]["end"] > audio_duration + 0.3:
            chunks[-1]["end"] = float(audio_duration)
        elif chunks[-1]["end"] < audio_duration - 0.5:
            # thoda extend kar do
            chunks[-1]["end"] = float(audio_duration)

    return full_text, chunks


# ------------------ GROQ VISION: IMAGE DESCRIPTIONS ------------------


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
            st.warning(f"Groq vision batch failed, will still continue. Error: {e}")
            break

        progress.progress(min((i + batch_size) / total, 1.0))

    return descriptions


# ------------------ GROQ TEXT: MAP CHUNKS TO IMAGES ------------------


def map_chunks_to_images_groq(chunks, image_desc_map, image_names):
    """
    Groq text model:
    - chunks: [{index, text, start, end}, ...]
    - image_desc_map: {filename: description}
    - image_names: actual filenames (for safety)

    Returns:
        mappings = [
          {"chunk_index": 1, "image": "some_file.jpg"},
          ...
        ]
    """
    if not chunks or not image_names:
        return []

    model_id = "llama-3.1-8b-instant"

    # Keep text short to avoid huge prompt
    chunks_brief = []
    for c in chunks:
        text = c["text"]
        if len(text) > 300:
            text = text[:300] + "..."
        chunks_brief.append({"index": c["index"], "text": text})

    payload = {
        "chunks": chunks_brief,
        "images": [
            {
                "filename": name,
                "description": image_desc_map.get(name, ""),
            }
            for name in image_names
        ],
    }

    prompt = f"""
You are helping to align audio narration chunks with images.

CHUNKS:
{json.dumps(payload["chunks"], ensure_ascii=False, indent=2)}

IMAGES (with optional descriptions):
{json.dumps(payload["images"], ensure_ascii=False, indent=2)}

TASK:
For each chunk, choose ONE image filename that best matches its content.
If descriptions are empty, rely on filename hints and the order.
Distribute chunks sensibly across images (do not always choose the same image).

Return ONLY JSON in this format:
{{
  "mappings": [
    {{"chunk_index": 1, "image": "filename1.jpg"}},
    {{"chunk_index": 2, "image": "filename2.jpg"}}
  ]
}}

RULES:
- "chunk_index" must match one of the provided chunk indexes.
- "image" must be exactly one of the provided filenames.
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
        st.warning(f"⚠️ Groq mapping failed, will fallback to simple order. Error: {e}")
        return []

    mappings = data.get("mappings") if isinstance(data, dict) else None
    if not isinstance(mappings, list):
        return []

    cleaned = []
    valid_set = set(image_names)
    for m in mappings:
        if not isinstance(m, dict):
            continue
        idx = m.get("chunk_index")
        img = m.get("image")
        if isinstance(idx, int) and img in valid_set:
            cleaned.append({"chunk_index": idx, "image": img})

    return cleaned


# ------------------ TIMELINE BUILD ------------------


def build_timeline_from_chunks(
    chunks, mappings, image_names, audio_duration=None, min_dur=0.5
):
    """
    - chunk durations Deepgram ke timestamps se (end - start)
    - Groq mapping se har chunk ko image assign
    - Agar mapping missing/invalid → fallback sequential mapping
    - Total ko audio_duration ke hisaab se scale karte hain
    """
    if not chunks or not image_names:
        return []

    idx_to_img = {}
    if mappings:
        for m in mappings:
            idx = m.get("chunk_index")
            img = m.get("image")
            if isinstance(idx, int) and img in image_names:
                idx_to_img[idx] = img

    timeline = []
    for i, chunk in enumerate(chunks):
        dur = max(chunk["end"] - chunk["start"], min_dur)
        idx = chunk["index"]
        img = idx_to_img.get(idx)
        if not img:
            # fallback: sequential assignment
            img = image_names[(idx - 1) % len(image_names)]

        timeline.append({"image": img, "duration_seconds": float(dur)})

    # Scale durations to match audio duration (optional but nice)
    if audio_duration and audio_duration > 5:
        total_planned = sum(max(t["duration_seconds"], 0.1) for t in timeline)
        if total_planned > 0:
            scale = audio_duration / total_planned
            for t in timeline:
                t["duration_seconds"] = max(t["duration_seconds"] * scale, min_dur)

    return timeline


def build_equal_slideshow(image_names, audio_duration=None):
    """
    Simple fallback: equal time per image.
    """
    if not image_names:
        return []

    if audio_duration and audio_duration > 1:
        per = audio_duration / len(image_names)
    else:
        per = 5.0

    per = max(per, 1.0)
    return [{"image": name, "duration_seconds": float(per)} for name in image_names]


# ------------------ VIDEO RENDERING ------------------


def render_video(timeline, audio_path, image_map, output_name="final_video.mp4"):
    """
    OpenCV + FFmpeg: images -> video frames -> merge with audio.
    """
    if not timeline:
        st.error("No timeline provided.")
        return None

    width, height = 854, 480
    fps = 24

    temp_video = os.path.join(TEMP_DIR, "temp_silent.mp4")
    final_output = os.path.join(TEMP_DIR, output_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

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

    st.write("🎵 Merging audio with video...")
    command = (
        f'ffmpeg -y -i "{temp_video}" -i "{audio_path}" '
        f'-c:v copy -c:a aac -shortest "{final_output}"'
    )
    subprocess.run(command, shell=True)

    return final_output


# ------------------ UI ------------------

st.title("🤖 AI Video Generator (Scene Sync)")
st.markdown(
    """
**Pipeline:**
1. Deepgram → audio ko chunks + timestamps me todta hai  
2. Groq Vision → images ka meaning samajhta hai  
3. Groq Text → har chunk ko best image assign karta hai  
4. OpenCV/FFmpeg → real-time durations ke saath video render
"""
)

col1, col2 = st.columns(2)
with col1:
    audio_file = st.file_uploader("Upload voiceover (MP3)", type=["mp3"])
with col2:
    uploaded_images = st.file_uploader(
        "Upload images (any order)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
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

        # 2. Deepgram chunks
        status.write("👂 Getting transcript + timestamps (Deepgram)...")
        full_text, chunks = get_transcript_chunks(local_audio, audio_duration=audio_duration)

        if not chunks:
            status.write("⚠️ No chunks from Deepgram, using equal slideshow.")
            valid_image_names = [img.name for img in uploaded_images]
            plan = build_equal_slideshow(valid_image_names, audio_duration)
        else:
            # 3. Image analysis
            status.write("👁️ Analyzing images with Groq Vision...")
            image_desc_map = analyze_images_with_groq(uploaded_images)

            if st.session_state.stop_requested:
                st.warning("⛔ Processing stopped by user.")
                st.stop()

            valid_image_names = [img.name for img in uploaded_images]

            # 4. Chunk → image mapping
            status.write("🧠 Mapping chunks to images with Groq Text...")
            mappings = map_chunks_to_images_groq(chunks, image_desc_map, valid_image_names)

            # 5. Build final timeline
            status.write("🧩 Building final timeline...")
            plan = build_timeline_from_chunks(
                chunks,
                mappings,
                valid_image_names,
                audio_duration=audio_duration,
            )

            if not plan:
                status.write("⚠️ Mapping failed, using equal slideshow.")
                plan = build_equal_slideshow(valid_image_names, audio_duration)

        if st.session_state.stop_requested:
            st.warning("⛔ Processing stopped by user.")
            st.stop()

        # 6. Render
        status.write("🎬 Rendering video with OpenCV...")
        image_map = {img.name: img for img in uploaded_images}
        final_vid_path = render_video(plan, local_audio, image_map)

        if final_vid_path is None or st.session_state.stop_requested:
            st.warning("⛔ Processing stopped before completion.")
            st.stop()

        status.update(label="✅ Video rendered!", state="complete", expanded=False)
        if audio_duration:
            st.success(f"Video rendered! (≈ {int(audio_duration)} seconds)")
        else:
            st.success("Video rendered! You can watch or download it below.")

        st.video(final_vid_path)

        with open(final_vid_path, "rb") as f:
            video_bytes = f.read()

        st.download_button(
            "⬇️ Download video",
            data=video_bytes,
            file_name="AI_Gen_Video.mp4",
            mime="video/mp4",
        )

        with st.expander("🔎 Debug: chunks + mappings (for testing)"):
            st.write("**Transcript (Deepgram):**")
            st.write(full_text)
            st.write("**Chunks:**")
            st.json(chunks)
            st.write("**Plan (timeline):**")
            st.json(plan)
