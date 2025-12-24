import os
import re
import json
import time
import shutil
import base64
import tempfile
import subprocess
from io import BytesIO
from typing import List, Tuple
from urllib.parse import urljoin

import requests
import streamlit as st
import cv2
import numpy as np
from groq import Groq
from PIL import Image

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Video Generator",
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
# SECRETS / KEYS
# =========================

# Transcription API key
try:
    DEEPGRAM_KEY = st.secrets["DEEPGRAM_API_KEY"]
except Exception:
    st.error("🚨 Please add DEEPGRAM_API_KEY in Streamlit secrets.")
    st.stop()

# Text model API key (Groq)
GROQ_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_KEY:
    st.error("🚨 Please add GROQ_API_KEY in Streamlit secrets.")
    st.stop()

groq_client = Groq(api_key=GROQ_KEY)

# Image API key
IMAGE_API_KEY = st.secrets.get("YOUSMIND_API_KEY")
if not IMAGE_API_KEY:
    st.error("🚨 Please add image API key in Streamlit secrets as YOUSMIND_API_KEY.")
    st.stop()

# Internal image API URL (hidden from UI)
IMAGE_API_URL = "https://yousmind.com/api/image-generator/generate"

# =========================
# TEMP DIR
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

FILENAME_SAFE = re.compile(r"[^a-zA-Z0-9_\-]+")


def safe_name(s: str, max_len: int = 60) -> str:
    """Convert a prompt into a safe file name."""
    s = s.strip().replace(" ", "_")
    s = FILENAME_SAFE.sub("", s)
    return (s[:max_len] or "prompt").strip("_")


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


def download_image(url: str, timeout: int = 60) -> Tuple[str, bytes]:
    """
    Download an image from a URL.
    """
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()

    ext = "png"
    lower = url.lower()
    if ".jpg" in lower or ".jpeg" in lower:
        ext = "jpg"
    elif ".gif" in lower:
        ext = "gif"
    elif ".webp" in lower:
        ext = "webp"
    else:
        ct = (r.headers.get("content-type") or "").lower()
        if "jpeg" in ct or "jpg" in ct:
            ext = "jpg"
        elif "gif" in ct:
            ext = "gif"
        elif "webp" in ct:
            ext = "webp"

    return ext, r.content


# =========================
# TRANSCRIPTION: TRANSCRIPT + CHUNKS
# =========================

def get_transcript_chunks(
    audio_path,
    audio_duration=None,
    max_gap=1.5,
    max_words_per_chunk=25,
    max_chunk_duration=30.0,
):
    """
    Transcription API:
    - full transcript
    - word timestamps
    - words ko chunks (segments) me todta hai:
        - agar words ke beech bohot gap ho (silence)
        - YA words count zyada ho jaye
        - YA chunk ki duration bohot badi ho jaye
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
        st.error(f"❌ Transcription request failed: {e}")
        return "", [], None

    try:
        alt = data["results"]["channels"][0]["alternatives"][0]
    except Exception as e:
        st.error(f"❌ Unexpected transcription response format: {e}")
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
        # last chunk ko audio ke end ke sath align rakhna
        if chunks[-1]["end"] > audio_duration + 0.3:
            chunks[-1]["end"] = float(audio_duration)
        elif chunks[-1]["end"] < audio_duration - 0.5:
            chunks[-1]["end"] = float(audio_duration)

    return full_text, chunks, data


# =========================
# TEXT MODEL: CHUNKS → IMAGE PROMPTS
# =========================

def generate_image_prompts_for_chunks(chunks):
    """
    Har chunk ke liye ek image-generation prompt.
    Response format (text only):
      Scene 1: ...
      Scene 2: ...
    """
    if not chunks:
        return {}, None

    model_id = "meta-llama/llama-4-maverick-17b-128e-instruct"

    chunks_brief = []
    for c in chunks:
        text = c.get("text", "")
        if len(text) > 400:
            text = text[:400] + "..."
        chunks_brief.append(
            {
                "index": c["index"],
                "start": round(float(c["start"]), 2),
                "end": round(float(c["end"]), 2),
                "text": text,
            }
        )

    prompt = f"""
You are an expert cinematic storyboard artist.

You receive a list of narration CHUNKS from a voiceover, each with:
- index
- start time (seconds)
- end time (seconds)
- text (the spoken line in that part of the story)

Your job is to create ONE image-generation prompt PER chunk.

Goals:
- Each prompt must closely match the meaning and mood of that chunk.
- Imagine this is for a cinematic video.
- Use clear, specific English.
- You can mention camera / composition / lighting when helpful.
- Do NOT mention the word "chunk", "voiceover", "caption", or any subtitles.
- Do NOT add anything about on-screen text or UI.
- Keep each prompt as a single sentence or short paragraph focused on visuals only.

CHUNKS:
{json.dumps(chunks_brief, ensure_ascii=False, indent=2)}

RESPONSE FORMAT (TEXT ONLY, NO JSON):
- For EACH chunk, write exactly ONE line in this format:
  Scene <index>: <image prompt>
"""

    try:
        resp = groq_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=4096,
            temperature=0.4,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        st.error(f"❌ Failed to generate image prompts: {e}")
        return {}, None

    raw_text = text

    prompts_map = {}
    for line in raw_text.splitlines():
        m = re.match(r"\s*Scene\s+(\d+)\s*:\s*(.+)", line, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            pmt = m.group(2).strip()
            if pmt:
                prompts_map[idx] = pmt

    expected_indexes = {c["index"] for c in chunks}
    missing = sorted(list(expected_indexes - set(prompts_map.keys())))
    if missing:
        st.error(
            f"❌ Some scene prompts are missing from the AI response. "
            f"Missing scene indexes: {missing}\n\n"
            "Open the 'Image prompts' debug expander, check the model output, "
            "and try again."
        )
        return {}, raw_text

    return prompts_map, raw_text


# =========================
# IMAGE API: SINGLE IMAGE PER PROMPT
# =========================

def generate_image_from_prompt(prompt, aspect_ratio, mode_label, api_key, timeout=60):
    """
    Image API wrapper:
    - Prompt is ALWAYS plain text string.
    - Returns: (filename, image_bytes, raw_response_dict)
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }
    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "provider": mode_label,
        "n": 1,
    }

    r = requests.post(IMAGE_API_URL, headers=headers, json=payload, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"raw_text": r.text}

    if r.status_code != 200:
        raise RuntimeError(
            f"HTTP {r.status_code} | Response: {str(data)[:400]}"
        )

    urls = data.get("image_urls", [])
    if not urls:
        raise RuntimeError(
            f"No image_urls in response: {str(data)[:400]}"
        )

    url = urls[0]
    if not url.lower().startswith("http"):
        url = urljoin(IMAGE_API_URL, url)

    ext, raw = download_image(url, timeout=timeout)
    filename = f"{safe_name(prompt)}.{ext}"
    return filename, raw, data


# =========================
# VIDEO RENDERING FROM SCENES
# =========================

def render_video_from_scenes(
    scenes,
    audio_path,
    color_mode="Color",
    zoom_strength=0.05,
    output_name="final_video.mp4",
):
    """
    Scenes: list of dicts:
        {
          "index": int,
          "start": float,
          "end": float,
          "text": str,
          "prompt": str,
          "image_name": str,
          "image_data": bytes
        }
    """
    if not scenes:
        st.error("No scenes to render.")
        return None

    for sc in scenes:
        if not sc.get("image_data"):
            st.error(f"Scene {sc['index']} has no image. Generate images for all scenes first.")
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

    # Precompute frame ranges per scene
    scene_frame_ranges = []
    for sc in scenes:
        start_s = max(float(sc["start"]), 0.0)
        end_s = max(float(sc["end"]), start_s + 0.04)

        start_f = int(round(start_s * fps))
        end_f = int(round(end_s * fps))
        if end_f <= start_f:
            end_f = start_f + 1

        scene_frame_ranges.append((start_f, end_f))

    total_frames_all = sum(end_f - start_f for (start_f, end_f) in scene_frame_ranges)
    if total_frames_all <= 0:
        st.error("❌ Computed total frames <= 0. Check timestamps.")
        return None

    st.write("🎥 Rendering video frames (frame-accurate sync)...")
    progress_bar = st.progress(0)
    percent_text = st.empty()

    current_frame = 0

    for sc, (start_f, end_f) in zip(scenes, scene_frame_ranges):
        if st.session_state.stop_requested:
            st.warning("⛔ Stopped during rendering.")
            out.release()
            return None

        frames_in_clip = max(end_f - start_f, 1)
        img_bytes = sc["image_data"]
        if not img_bytes:
            continue

        file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is None:
            continue

        img = cv2.resize(img, (width, height))

        for i in range(frames_in_clip):
            if st.session_state.stop_requested:
                st.warning("⛔ Stopped during rendering.")
                out.release()
                return None

            scale = 1.0 + (zoom_strength * i / frames_in_clip)
            M = cv2.getRotationMatrix2D((width // 2, height // 2), 0, scale)
            frame = cv2.warpAffine(img, M, (width, height))

            if color_mode == "Black & white":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            out.write(frame)
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
        st.error(f"❌ Audio file not found: {audio_abs}")
        return None
    if not os.path.exists(temp_video_abs):
        st.error(f"❌ Temp video not found: {temp_video_abs}")
        return None

    st.write("🎵 Merging audio with video...")

    command = [
        "ffmpeg",
        "-y",
        "-i", temp_video_abs,
        "-i", audio_abs,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
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
        st.error("❌ Video/audio merge failed.")
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
# SESSION STATE STRUCTURE
# =========================

if "scenes" not in st.session_state:
    st.session_state.scenes = []

if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "audio_duration" not in st.session_state:
    st.session_state.audio_duration = None

if "transcription_raw" not in st.session_state:
    st.session_state.transcription_raw = None

if "prompt_raw" not in st.session_state:
    st.session_state.prompt_raw = None

# ✅ store rendered video persistently
if "final_video_bytes" not in st.session_state:
    st.session_state.final_video_bytes = None

if "final_video_name" not in st.session_state:
    st.session_state.final_video_name = "AI_Video.mp4"


# =========================
# SIDEBAR SETTINGS
# =========================

with st.sidebar:
    st.header("Image Settings")

    mode_label = st.selectbox(
        "Image mode",
        ["1.5-Fast", "1.0-Slow"],
        index=0,
        help="Choose which style/engine mode to use.",
    )

    aspect_ratio = st.selectbox(
        "Aspect ratio",
        ["16:9", "9:16", "1:1"],
        index=0,
        help="Select the output aspect ratio.",
    )

    timeout = st.slider("Image request timeout (seconds)", 10, 180, 60)

    color_mode = st.radio(
        "Video color mode",
        ["Color", "Black & white"],
        index=0,
    )

    zoom_strength = st.slider(
        "Zoom strength",
        0.0,
        0.15,
        0.05,
        step=0.01,
        help="0 = no zoom, 0.15 = strong zoom",
    )

    st.button("🛑 Stop processing", on_click=request_stop)


# =========================
# MAIN UI
# =========================

st.title("🎬 AI Video Generator")

st.markdown("")

audio_file = st.file_uploader("Upload voiceover (MP3)", type=["mp3"])

st.divider()

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    analyze = st.button("1️⃣ Analyze Voiceover & Create Scenes", type="primary")
with col_btn2:
    reset = st.button("♻️ Reset / Clear")

if reset:
    st.session_state.scenes = []
    st.session_state.audio_path = None
    st.session_state.audio_duration = None
    st.session_state.transcription_raw = None
    st.session_state.prompt_raw = None
    st.session_state.final_video_bytes = None
    st.session_state.final_video_name = "AI_Video.mp4"
    st.session_state.stop_requested = False
    st.success("State cleared.")
    st.stop()

# =========================
# STEP 1: ANALYZE (TRANSCRIPT + PROMPTS + AUTO IMAGES)
# =========================

if analyze:
    st.session_state.stop_requested = False

    if not audio_file:
        st.error("Please upload an MP3 voiceover first.")
        st.stop()

    status = st.status("Starting analysis...", expanded=True)

    # Clean temp
    status.write("🧹 Cleaning temp folder...")
    clean_temp_dir()

    # Save audio
    status.write("💾 Saving audio file...")
    local_audio = os.path.join(TEMP_DIR, "input.mp3")
    local_audio = os.path.abspath(local_audio)
    with open(local_audio, "wb") as f:
        f.write(audio_file.getbuffer())

    if not os.path.exists(local_audio):
        st.error("❌ Failed to save audio file.")
        st.stop()

    audio_duration = get_audio_duration(local_audio)
    st.session_state.audio_path = local_audio
    st.session_state.audio_duration = audio_duration

    if audio_duration:
        status.write(f"⏱️ Audio duration: ~{audio_duration:.2f} seconds")

    if st.session_state.stop_requested:
        st.warning("⛔ Stopped by user.")
        st.stop()

    # Transcription
    status.write("👂 Getting transcript + timestamps...")
    full_text, chunks, raw = get_transcript_chunks(
        local_audio, audio_duration=audio_duration
    )
    st.session_state.transcription_raw = raw

    if not chunks:
        st.error("❌ No scenes found from voiceover.")
        st.stop()

    if st.session_state.stop_requested:
        st.warning("⛔ Stopped by user.")
        st.stop()

    # Text model: prompts
    status.write("🧠 Generating image prompts for each scene (Groq Llama-4 Maverick)...")
    prompts_map, prompt_raw = generate_image_prompts_for_chunks(chunks)
    st.session_state.prompt_raw = prompt_raw

    if not prompts_map:
        status.update(label="❌ Prompt generation failed.", state="error", expanded=True)
        st.stop()

    # Build scenes
    scenes = []
    for c in chunks:
        idx = c["index"]
        scenes.append(
            {
                "index": idx,
                "start": float(c["start"]),
                "end": float(c["end"]),
                "text": c["text"],
                "prompt": prompts_map.get(idx, ""),
                "image_name": None,
                "image_data": None,
            }
        )

    # ✅ last scene audio se 2s zyada
    if st.session_state.audio_duration and scenes:
        ad = float(st.session_state.audio_duration)
        target_end = ad + 2.0
        if scenes[-1]["end"] < target_end:
            scenes[-1]["end"] = target_end

    # Auto-generate images
    status.write("🖼️ Generating images for all scenes...")
    img_progress = st.progress(0.0)
    total_scenes = len(scenes)
    generated_count = 0

    for i, sc in enumerate(scenes, start=1):
        if st.session_state.stop_requested:
            st.warning("⛔ Stopped during image generation.")
            break
        try:
            with st.spinner(f"Generating image for scene {sc['index']}..."):
                fname, data, _ = generate_image_from_prompt(
                    sc["prompt"],
                    aspect_ratio,
                    mode_label,
                    IMAGE_API_KEY,
                    timeout=timeout,
                )
                sc["image_name"] = fname
                sc["image_data"] = data
                generated_count += 1
        except Exception as e:
            st.error(f"Scene {sc['index']} image error: {e}")
        img_progress.progress(i / total_scenes)

    st.session_state.scenes = scenes

    if generated_count == total_scenes:
        status.update(label="✅ Scenes + prompts + images ready!", state="complete", expanded=False)
        st.success("All scenes created and images generated. Scroll down to review or edit.")
    else:
        status.update(
            label=f"⚠️ Scenes created, but some images failed ({generated_count}/{total_scenes}).",
            state="running",
            expanded=True,
        )


# =========================
# SCENE EDITOR (PROMPT + IMAGE)
# =========================

scenes = st.session_state.scenes
audio_path = st.session_state.audio_path

if scenes:
    st.subheader("Scene Prompts & Images (Editing)")

    st.caption(
        "Har scene voiceover se linked hai. Prompt edit karo, phir image generate / regenerate karo."
    )

    # Bulk generate missing images
    if st.button("🖼️ Generate images for ALL scenes without image"):
        for sc in scenes:
            if st.session_state.stop_requested:
                st.warning("⛔ Stopped by user.")
                break
            if sc["image_data"]:
                continue
            try:
                with st.spinner(f"Generating image for scene {sc['index']}..."):
                    fname, data, _ = generate_image_from_prompt(
                        sc["prompt"],
                        aspect_ratio,
                        mode_label,
                        IMAGE_API_KEY,
                        timeout=timeout,
                    )
                    sc["image_name"] = fname
                    sc["image_data"] = data
            except Exception as e:
                st.error(f"Scene {sc['index']} image error: {e}")
        st.session_state.scenes = scenes
        st.rerun()

    st.divider()

    # Per-scene UI
    for sc in scenes:
        idx = sc["index"]
        start = sc["start"]
        end = sc["end"]
        text = sc["text"]

        with st.container():
            st.markdown(f"### 🎬 Scene {idx}")
            st.caption(f"Time: {start:.2f}s → {end:.2f}s  |  Duration: {end - start:.2f}s")

            st.text_area(
                "Voiceover text (read-only)",
                value=text,
                key=f"scene_text_{idx}",
                height=80,
                disabled=True,
            )

            prompt_key = f"scene_prompt_{idx}"
            if prompt_key not in st.session_state:
                st.session_state[prompt_key] = sc["prompt"]

            new_prompt = st.text_area(
                "Image prompt",
                key=prompt_key,
                height=80,
            )
            sc["prompt"] = new_prompt

            col_a, col_b = st.columns([1, 2])

            with col_a:
                gen_label = "Generate image" if not sc["image_data"] else "Regenerate image"
                if st.button(gen_label, key=f"gen_btn_{idx}"):
                    try:
                        with st.spinner(f"Generating image for scene {idx}..."):
                            fname, data, _ = generate_image_from_prompt(
                                sc["prompt"],
                                aspect_ratio,
                                mode_label,
                                IMAGE_API_KEY,
                                timeout=timeout,
                            )
                            sc["image_name"] = fname
                            sc["image_data"] = data
                            st.session_state.scenes = scenes
                            st.rerun()
                    except Exception as e:
                        st.error(f"Image error (scene {idx}): {e}")

            with col_b:
                if sc["image_data"]:
                    st.image(
                        sc["image_data"],
                        caption=f"{sc['image_name'] or 'Scene image'}",
                        width="stretch",
                    )
                else:
                    st.info("No image generated yet for this scene.")

            st.markdown("---")

    st.divider()

    # =========================
    # FINAL REVIEW
    # =========================

    st.subheader("Final Review (Text + Prompts + Images)")

    st.caption(
        "Yahan se last check karo. Prompts edit karo, images regenerate karo, phir neeche se video render karo."
    )

    missing = [sc["index"] for sc in scenes if not sc["image_data"]]
    if missing:
        st.warning(
            f"These scenes have no image yet: {missing}. "
            "Har scene k liye image generate karo before rendering."
        )

    for sc in scenes:
        idx = sc["index"]
        start = sc["start"]
        end = sc["end"]

        with st.expander(f"Scene {idx} – {start:.2f}s → {end:.2f}s", expanded=False):
            st.write("**Voiceover text:**")
            st.write(sc["text"])

            prompt_key = f"scene_prompt_{idx}"

            new_prompt_final = st.text_area(
                "Final prompt (you can still edit)",
                key=prompt_key + "_final",
                value=sc["prompt"],
                height=80,
            )

            sc["prompt"] = new_prompt_final

            col1, col2 = st.columns([1, 2])
            with col1:
                gen_label = "Generate image" if not sc["image_data"] else "Regenerate image"
                if st.button(gen_label, key=f"final_gen_btn_{idx}"):
                    try:
                        with st.spinner(f"Generating image for scene {idx} (final review)..."):
                            fname, data, _ = generate_image_from_prompt(
                                sc["prompt"],
                                aspect_ratio,
                                mode_label,
                                IMAGE_API_KEY,
                                timeout=timeout,
                            )
                            sc["image_name"] = fname
                            sc["image_data"] = data
                            st.session_state.scenes = scenes
                            st.rerun()
                    except Exception as e:
                        st.error(f"Image error (scene {idx}): {e}")

            with col2:
                if sc["image_data"]:
                    st.image(
                        sc["image_data"],
                        caption=f"{sc['image_name'] or 'Scene image'}",
                        width="stretch",
                    )
                else:
                    st.info("No image generated yet for this scene.")

    st.divider()

    # =========================
    # FINAL RENDER BUTTON
    # =========================

    if st.button("✅ Finalize & Render Video", type="primary"):
        if not audio_path or not os.path.exists(audio_path):
            st.error("Audio file not found. Please re-analyze the voiceover.")
        elif missing:
            st.error("Some scenes have no image. Generate all images first.")
        else:
            st.session_state.stop_requested = False
            with st.spinner("Rendering video..."):
                final_vid_path = render_video_from_scenes(
                    st.session_state.scenes,
                    audio_path,
                    color_mode=color_mode,
                    zoom_strength=zoom_strength,
                )

            if final_vid_path is None:
                st.warning("⛔ Video generation failed or stopped.")
            else:
                with open(final_vid_path, "rb") as f:
                    video_bytes = f.read()

                st.session_state.final_video_bytes = video_bytes
                st.session_state.final_video_name = "AI_Video.mp4"

                st.success("🎉 Video rendered successfully! Scroll down to preview & download.")

# =========================
# PERSISTENT PREVIEW + DOWNLOAD
# =========================

if st.session_state.final_video_bytes:
    st.subheader("Preview & Download")
    st.video(st.session_state.final_video_bytes)
    st.download_button(
        "⬇️ Download video",
        data=st.session_state.final_video_bytes,
        file_name=st.session_state.final_video_name,
        mime="video/mp4",
    )

# =========================
# DEBUG EXPANDERS
# =========================

if st.session_state.transcription_raw:
    with st.expander("🧩 Transcription Output (raw JSON + scenes)", expanded=False):
        st.subheader("Raw transcription JSON")
        st.json(st.session_state.transcription_raw)
        st.subheader("Scenes (index, start, end, text)")
        st.json(
            [
                {
                    "index": sc["index"],
                    "start": sc["start"],
                    "end": sc["end"],
                    "text": sc["text"],
                }
                for sc in st.session_state.scenes
            ]
        )

if st.session_state.prompt_raw:
    with st.expander("🧠 Image prompts (raw model output)", expanded=False):
        st.text(st.session_state.prompt_raw)
