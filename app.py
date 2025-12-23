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
import re

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Video Generator (Deepgram + Groq)",
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
    st.error("🚨 Please add GROQ_API_KEY in Streamlit secrets as GROQ_API_KEY.")
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
# DEEPGRAM: TRANSCRIPT + TIMESTAMPS + SMART CHUNKS
# =========================

def get_transcript_chunks(
    audio_path,
    audio_duration=None,
    max_gap=1.5,
    max_words_per_chunk=25,
    max_chunk_duration=12.0,
):
    """
    Deepgram:
    - full transcript
    - word timestamps
    - words ko chunks (segments) me todta hai:
        - agar words ke beech bohot gap ho (silence)
        - YA words count zyada ho jaye
        - YA chunk ki duration bohot badi ho jaye

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
        current_duration = prev_end - current_start
        too_long = current_duration >= max_chunk_duration
        too_many_words = len(current_words) >= max_words_per_chunk

        if (gap > max_gap or too_long or too_many_words) and current_words:
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

    if audio_duration and chunks:
        if chunks[-1]["end"] > audio_duration + 0.3:
            chunks[-1]["end"] = float(audio_duration)
        elif chunks[-1]["end"] < audio_duration - 0.5:
            chunks[-1]["end"] = float(audio_duration)

    return full_text, chunks, data


# =========================
# GROQ VISION: PER-IMAGE DESCRIPTION
# =========================

def describe_all_images_with_groq(image_files, max_images=50):
    """
    Groq Vision se images ki detailed descriptions nikalta hai.
    - Har image ke liye ALAG request (zyada accurate)
    - Prompt: "What is in this image and what is happening?" (English)
    Returns:
        desc_map: {filename: description}
        raw_responses: list of raw JSON responses (debug ke liye)
    """
    if not image_files:
        return {}, []

    image_files = image_files[:max_images]
    model_id = "meta-llama/llama-4-scout-17b-16e-instruct"

    desc_map = {}
    raw_responses = []

    total = len(image_files)
    progress = st.progress(0)
    status = st.empty()

    for i, uf in enumerate(image_files, start=1):
        if st.session_state.stop_requested:
            st.warning("⛔ Stopped by user during image description.")
            break

        status.text(f"👁️ Describing image {i} of {total}: {uf.name}")

        # image → resized JPEG → data URL
        uf.seek(0)
        img = Image.open(uf).convert("RGB")
        max_side = 1280
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)))

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{b64}"

        content = [
            {
                "type": "text",
                "text": (
                    "You are describing ONE image.\n\n"
                    "QUESTION:\n"
                    "- What is in this image?\n"
                    "- What is happening in this image?\n"
                    "- Answer in clear, simple English.\n\n"
                    "Return ONLY JSON in this exact format:\n"
                    "{\n"
                    f'  \"filename\": \"{uf.name}\",\n'
                    '  \"description\": "one or two sentences describing what is in the image and what is happening"\n'
                    "}\n"
                    "- Do NOT mention indexes.\n"
                    "- Do NOT talk about other images.\n"
                    "- Focus ONLY on this single image."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            },
        ]

        try:
            resp = groq_client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                max_completion_tokens=256,
                temperature=0.2,
            )
            text = resp.choices[0].message.content or "{}"
            data = json.loads(text)
            raw_responses.append(data)

            desc = data.get("description") or ""
            desc = desc.strip()
            if not desc:
                desc = "No clear description."

            # apne actual filename ko source of truth rakhte hain
            desc_map[uf.name] = desc
        except Exception as e:
            st.warning(f"⚠️ Groq Vision failed for {uf.name}: {e}")
            desc_map[uf.name] = "Description failed."

        progress.progress(i / total)

    return desc_map, raw_responses


# =========================
# GROQ TEXT: SEMANTIC MAPPING (VOICEOVER + DESCRIPTIONS)
# =========================

def map_chunks_to_images_using_descriptions(chunks, image_desc_map):
    """
    Groq TEXT se semantic mapping:
    - CHUNKS (Deepgram se): [{index, text, start, end}]
    - IMAGE_DESC_MAP: {filename: description}

    Groq ko:
      - chunks (brief)
      - saari images (filename + description)
    bhejte hain, aur yeh maangte hain:
      {
        "mappings": [
          { "chunk_index": 1, "image": "some_file.png" },
          ...
        ]
      }

    Rules:
      - Har chunk_index EXACTLY once aana chahiye
      - Image name IMAGES list se hi copy hona chahiye
      - Sequence pe rely nahi karna, sirf meaning pe
    """
    if not chunks or not image_desc_map:
        return [], None

    model_id = "llama-3.1-8b-instant"

    # short chunks for context
    chunks_brief = []
    for c in chunks:
        text = c.get("text", "")
        if len(text) > 350:
            text = text[:350] + "..."
        chunks_brief.append({"index": c["index"], "text": text})

    images_list = [
        {"filename": fn, "description": desc}
        for fn, desc in image_desc_map.items()
    ]

    num_chunks = len(chunks_brief)
    num_images = len(images_list)
    min_distinct = min(4, num_images)

    prompt = f"""
You are aligning narration chunks with images.

CHUNKS (from a voiceover, already time-aligned separately):
{json.dumps(chunks_brief, ensure_ascii=False, indent=2)}

IMAGES (each with filename and description):
{json.dumps(images_list, ensure_ascii=False, indent=2)}

TASK:
For EACH chunk in CHUNKS, choose EXACTLY ONE image filename from IMAGES
that best matches the MEANING of that chunk's text.
Use semantic similarity between the chunk text and the image description.

VERY IMPORTANT RULES:
- You MUST create exactly {num_chunks} mapping objects.
- Each "chunk_index" must appear EXACTLY ONCE in the mappings.
- "image" must be EXACTLY one of the filenames from IMAGES (copy-paste).
- Do NOT assign based on list order or index (no 1->1, 2->2 just by position).
- REUSE of images is allowed, but ONLY if the meaning clearly matches.
- Try to use AT LEAST {min_distinct} different images overall (if logically possible).
- Do NOT assign all chunks to the same image.

RETURN FORMAT (JSON ONLY):
{{
  "mappings": [
    {{ "chunk_index": 1, "image": "FILENAME_FROM_IMAGES_LIST" }},
    {{ "chunk_index": 2, "image": "FILENAME_FROM_IMAGES_LIST" }}
  ]
}}
"""

    try:
        resp = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_completion_tokens=2048,
            temperature=0.2,
        )
        text = resp.choices[0].message.content or "{}"
        data = json.loads(text)
    except Exception as e:
        st.warning(f"⚠️ Groq text mapping failed: {e}")
        return [], None

    mappings = data.get("mappings")
    if not isinstance(mappings, list):
        return [], data

    valid_files = set(image_desc_map.keys())
    valid_chunk_indexes = {c["index"] for c in chunks}

    cleaned = []
    seen_chunks = set()

    for m in mappings:
        if not isinstance(m, dict):
            continue
        idx = m.get("chunk_index")
        img = m.get("image")
        if (
            isinstance(idx, int)
            and idx in valid_chunk_indexes
            and img in valid_files
            and idx not in seen_chunks
        ):
            cleaned.append({"chunk_index": idx, "image": img})
            seen_chunks.add(idx)

    if len(cleaned) < len(chunks):
        st.warning(
            f"⚠️ Groq mapping only returned {len(cleaned)} of {len(chunks)} chunks. "
            "Missing chunks will use fallback later."
        )

    return cleaned, data


# =========================
# TIMELINE BUILD
# =========================

def build_timeline_sequential(chunks, image_names, audio_duration=None, min_dur=0.5):
    """Simple deterministic fallback: chunk 1 → img 1, chunk 2 → img 2, etc."""
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


def build_timeline_with_mapping(chunks, mappings, image_names,
                                audio_duration=None, min_dur=0.5):
    """
    Mapping-based:
    - Agar mapping bekaar ho (sirf 1 image / 1 chunk / bohot kam coverage)
      → sequential fallback
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
    used_chunks = set(idx_to_img.keys())
    coverage_ratio = len(used_chunks) / max(len(chunks), 1)

    if (
        len(used_images) <= 1
        or len(used_chunks) <= 1
        or coverage_ratio < 0.5
    ):
        st.warning("⚠️ Groq mapping looked degenerate, using sequential mapping instead.")
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

st.title("🤖 AI Video Generator (Deepgram + Groq Vision/Text)")
st.markdown(
    """
**Pipeline Overview**

1. **Deepgram** – Voiceover se transcript + timestamps + multiple chunks  
2. **Groq Vision** – Har image pe: "What is in this image? What is happening?" (English description)  
3. **Groq Text** – Voiceover chunks + image descriptions → semantic mapping (chunk → best image)  
4. **Timeline** – Deepgram time se sync, Groq mapping + strong fallback  
5. **OpenCV + FFmpeg** – Frames render + audio merge, progress + stop button  

Neeche expanders me har step ka JSON dekh sakte ho.
"""
)

col1, col2 = st.columns(2)
with col1:
    audio_file = st.file_uploader("Upload voiceover (MP3)", type=["mp3"])
with col2:
    uploaded_images = st.file_uploader(
        "Upload images (max ~50, story-related)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

st.button("🛑 Stop processing", on_click=request_stop)

if st.button("🚀 Generate video", type="primary"):
    st.session_state.stop_requested = False

    if not audio_file or not uploaded_images:
        st.error("Please upload both audio and images first.")
    else:
        # ============ Init debug holders ============
        deepgram_raw = None
        image_desc_raw_list = []
        image_desc_map = {}
        mappings = []
        mapping_raw = None
        timeline = []
        full_text = ""
        chunks = []

        if len(uploaded_images) > 50:
            st.warning(
                f"⚠️ {len(uploaded_images)} images uploaded. "
                "Using only the first 50 for analysis + mapping."
            )
            uploaded_images = uploaded_images[:50]

        status = st.status("Starting process...", expanded=True)

        status.write("🧹 Cleaning temp folder...")
        clean_temp_dir()

        # Save audio
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

        # Deepgram
        status.write("👂 Getting transcript + timestamps (Deepgram)...")
        full_text, chunks, deepgram_raw = get_transcript_chunks(
            local_audio, audio_duration=audio_duration
        )

        image_names = [img.name for img in uploaded_images]

        if not chunks:
            status.write("⚠️ No chunks from Deepgram, falling back to simple slideshow.")
            dummy_chunks = []
            if audio_duration and image_names:
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

        # Groq Vision: descriptions
        status.write("👁️ Describing ALL images with Groq Vision...")
        image_desc_map, image_desc_raw_list = describe_all_images_with_groq(
            uploaded_images
        )

        if st.session_state.stop_requested:
            st.warning("⛔ Stopped by user.")
            st.stop()

        # Groq Text: mapping
        if image_desc_map:
            status.write("🧠 Mapping chunks → images (Groq Text)...")
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

        # Render video
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

        with open(final_vid_path, "rb") as f:
            video_bytes = f.read()

        st.video(video_bytes)

        st.download_button(
            "⬇️ Download video",
            data=video_bytes,
            file_name="AI_Gen_Video.mp4",
            mime="video/mp4",
        )

        # DEBUG PANELS

        with st.expander("🧩 Deepgram Output (raw JSON + chunks + transcript)"):
            st.subheader("Raw Deepgram JSON (full)")
            st.json(deepgram_raw)
            st.subheader("Transcript (combined text)")
            st.write(full_text)
            st.subheader("Chunks (index, text, start, end)")
            st.json(chunks)

        with st.expander("👁️ Groq Vision – Image Descriptions"):
            st.markdown("**Raw responses per image (JSON exactly as Groq returned):**")
            for idx, raw in enumerate(image_desc_raw_list, start=1):
                st.markdown(f"**Image {idx}**")
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
