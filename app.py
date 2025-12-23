import os
import io
import json
import time
import shutil
import tempfile
from typing import List, Dict, Any

import requests
import numpy as np
import cv2
from groq import Groq
from PIL import Image

import streamlit as st

# -------------------------------------------------
# Page & temp setup
# -------------------------------------------------
st.set_page_config(
    page_title="AI Synced Video Builder",
    page_icon="🎬",
    layout="wide",
)

BASE_TEMP_DIR = "temp_processing"
if os.path.exists(BASE_TEMP_DIR):
    shutil.rmtree(BASE_TEMP_DIR)
os.makedirs(BASE_TEMP_DIR, exist_ok=True)

# -------------------------------------------------
# Secrets / keys
# -------------------------------------------------
def get_secret(name: str, fallback: str = None) -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    if fallback and fallback in st.secrets:
        return st.secrets[fallback]
    return None

DEEPGRAM_KEY = get_secret("DEEPGRAM_API_KEY")
GROQ_KEY = get_secret("GROQ_API_KEY")
IMAGE_API_KEY = get_secret("IMAGE_API_KEY", fallback="YOUSMIND_API_KEY")
IMAGE_API_URL = get_secret("IMAGE_API_URL") or "https://yousmind.com/api/image-generator/generate"

if not DEEPGRAM_KEY or not GROQ_KEY or not IMAGE_API_KEY:
    st.error(
        "Secrets missing! Please set `DEEPGRAM_API_KEY`, `GROQ_API_KEY` and "
        "`IMAGE_API_KEY` (or `YOUSMIND_API_KEY`) in `.streamlit/secrets.toml`."
    )
    st.stop()

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def save_uploaded_file(uploaded, folder: str, filename: str) -> str:
    """Save an uploaded file to disk and return path."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path

# -------------------------------------------------
# 1. Deepgram – transcript + timestamps + 5s chunks
# -------------------------------------------------
def deepgram_transcribe(audio_path: str) -> Dict[str, Any]:
    """Call Deepgram REST API and return JSON response."""
    url = "https://api.deepgram.com/v1/listen"
    params = {
        "model": "nova-2",
        "smart_format": "true",
        "punctuate": "true",
        "diarize": "false",
        "paragraphs": "true",
    }
    headers = {
        "Authorization": f"Token {DEEPGRAM_KEY}",
        "Content-Type": "audio/mpeg",
    }

    with open(audio_path, "rb") as f:
        data = f.read()

    resp = requests.post(url, headers=headers, params=params, data=data, timeout=180)
    resp.raise_for_status()
    return resp.json()


def build_chunks_from_words(words: List[Dict[str, Any]], max_chunk_sec: float = 5.0) -> List[Dict[str, Any]]:
    """
    Make time-based chunks from Deepgram word list.
    Each chunk ~ <= max_chunk_sec, 100% time-sync.
    """
    if not words:
        return []

    chunks = []
    cur_words = []
    cur_start = words[0]["start"]
    last_end = cur_start

    for w in words:
        w_start = w["start"]
        w_end = w["end"]
        last_end = w_end
        cur_words.append(w["word"])
        # if this word pushes us past max_chunk_sec → close the chunk
        if (w_end - cur_start) >= max_chunk_sec:
            chunks.append(
                {
                    "text": " ".join(cur_words).strip(),
                    "start": float(cur_start),
                    "end": float(w_end),
                }
            )
            cur_words = []
            cur_start = w_end

    # remaining tail
    if cur_words:
        chunks.append(
            {
                "text": " ".join(cur_words).strip(),
                "start": float(cur_start),
                "end": float(last_end),
            }
        )

    # make sure duration > 0
    for c in chunks:
        if c["end"] <= c["start"]:
            c["end"] = c["start"] + 0.5

    # add index for UI
    for i, c in enumerate(chunks, start=1):
        c["index"] = i

    return chunks


def extract_words_from_deepgram(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Safely pull word list from Deepgram response.
    """
    try:
        alt = resp["results"]["channels"][0]["alternatives"][0]
    except Exception:
        return []

    if "words" in alt:
        return alt["words"]

    # Fallback: construct single chunk over full transcript
    start = 0.0
    end = float(alt.get("duration", 0.0) or 5.0)
    return [{"word": alt.get("transcript", ""), "start": start, "end": end}]

# -------------------------------------------------
# 2. Groq – scene prompts for each chunk
# -------------------------------------------------
def groq_scene_prompts(chunks: List[Dict[str, Any]], global_style: str) -> List[str]:
    """
    Ask Groq to generate one image prompt per chunk.
    Returns list of prompts (len == len(chunks)).
    """
    client = Groq(api_key=GROQ_KEY)

    # Build a concise description for Groq
    parts = []
    for c in chunks:
        parts.append(
            f"Scene {c['index']} ({c['start']:.2f}s–{c['end']:.2f}s): {c['text']}"
        )
    joined = "\n".join(parts)

    style_suffix = ""
    if global_style:
        style_suffix = f" The overall visual style should be: {global_style}."

    system_msg = (
        "You are an expert storyboard artist for videos. "
        "Given scenes of narration, you write one DALL·E / SD-style image prompt "
        "for each scene, describing what should be shown visually."
    )

    user_msg = (
        "Here is the narration split into numbered scenes with their timings:\n\n"
        f"{joined}\n\n"
        "Write ONE image prompt per scene, in English.\n"
        f"{style_suffix}\n\n"
        "Respond ONLY as plain text in this exact format (no JSON, no markdown):\n"
        "Scene 1: <prompt text>\n"
        "Scene 2: <prompt text>\n"
        "Scene 3: <prompt text>\n"
        "...\n"
        "Use the same scene numbers that I provided."
    )

    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.6,
        max_completion_tokens=2048,
        top_p=1,
    )

    raw_text = completion.choices[0].message.content.strip()

    # Parse "Scene X: prompt"
    prompts_by_index = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not line.lower().startswith("scene"):
            continue
        # Expected: "Scene N: prompt..."
        try:
            label, prompt = line.split(":", 1)
        except ValueError:
            continue
        label = label.strip().lower()
        prompt = prompt.strip()
        # extract index
        idx = None
        for token in label.split():
            if token.isdigit():
                idx = int(token)
                break
        if idx is not None and prompt:
            prompts_by_index[idx] = prompt

    # Build a final list aligned with chunks; fallback = generic from text
    prompts = []
    for c in chunks:
        idx = c["index"]
        base_prompt = prompts_by_index.get(
            idx,
            f"An illustrative shot matching this narration: {c['text']}",
        )
        prompts.append(base_prompt)

    return prompts

# -------------------------------------------------
# 3. Image generation API (generic engine)
# -------------------------------------------------
def call_image_engine(prompt: str, aspect_ratio: str, engine_name: str) -> bytes:
    """
    Calls external image engine API and returns raw image bytes.
    We only ask for 1 image per prompt.
    """
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": IMAGE_API_KEY,
    }
    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "provider": engine_name,
        "n": 1,
    }

    resp = requests.post(IMAGE_API_URL, headers=headers, json=payload, timeout=180)
    try:
        data = resp.json()
    except Exception:
        raise RuntimeError(f"Image API returned non-JSON: {resp.text[:300]}")

    if resp.status_code != 200:
        raise RuntimeError(
            f"Image API HTTP {resp.status_code}: {str(data)[:300]}"
        )

    urls = data.get("image_urls") or []
    if not urls:
        raise RuntimeError(f"No image_urls in response: {str(data)[:300]}")

    url = urls[0]
    if not url.lower().startswith("http"):
        from urllib.parse import urljoin
        url = urljoin(IMAGE_API_URL, url)

    img_resp = requests.get(url, timeout=180)
    img_resp.raise_for_status()
    return img_resp.content

# -------------------------------------------------
# 4. Video rendering (OpenCV + ffmpeg)
# -------------------------------------------------
def render_video(
    scenes: List[Dict[str, Any]],
    audio_path: str,
    color_mode: str = "Color",
    zoom_strength: float = 0.05,
    fps: int = 24,
    resolution=(854, 480),
) -> str:
    """
    Render video from scenes + audio.
    Each scene: {"start", "end", "image_bytes"} – completely time-synced with audio.
    """
    width, height = resolution
    temp_silent = os.path.join(BASE_TEMP_DIR, "temp_silent.mp4")
    final_path = os.path.join(BASE_TEMP_DIR, "final_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_silent, fourcc, fps, (width, height))

    # total duration just for logging
    total_sec = scenes[-1]["end"] - scenes[0]["start"]
    total_frames_approx = int(total_sec * fps) + 1
    progress = st.progress(0.0)
    frame_counter = 0

    st.write("🎥 Rendering video frames...")

    for scene in scenes:
        duration = scene["end"] - scene["start"]
        frames = max(1, int(duration * fps))

        # load image bytes into cv2
        image_bytes = scene["image_bytes"]
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            # fallback: black frame
            img = np.zeros((height, width, 3), dtype=np.uint8)

        img = cv2.resize(img, (width, height))

        for i in range(frames):
            # simple zoom in/out
            # t from 0..1
            t = i / max(1, frames - 1)
            scale = 1.0 + zoom_strength * (t - 0.5)  # small zoom variation
            M = cv2.getRotationMatrix2D((width // 2, height // 2), 0, scale)
            frame = cv2.warpAffine(img, M, (width, height))

            if color_mode == "Black & white":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            out.write(frame)
            frame_counter += 1
            if frame_counter % 50 == 0:
                progress.progress(min(frame_counter / total_frames_approx, 1.0))

    out.release()
    progress.progress(1.0)

    # Merge audio with ffmpeg
    st.write("🎵 Merging audio with video (ffmpeg)...")
    cmd = (
        f'ffmpeg -y -i "{temp_silent}" -i "{audio_path}" '
        f'-c:v libx264 -c:a aac -shortest "{final_path}"'
    )
    # Run ffmpeg
    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        st.error("❌ FFmpeg failed. See logs below.")
        st.code(result.stderr[:1000])
        raise RuntimeError("ffmpeg merge failed")

    return final_path

# -------------------------------------------------
# Session state init
# -------------------------------------------------
if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []
if "scene_prompts" not in st.session_state:
    st.session_state["scene_prompts"] = []
if "scene_images" not in st.session_state:
    st.session_state["scene_images"] = []  # list of dict {index, prompt, image_bytes, source}
if "video_path" not in st.session_state:
    st.session_state["video_path"] = None

# -------------------------------------------------
# UI – Layout
# -------------------------------------------------
st.title("🎬 AI Synced Video Builder (Deepgram + Groq + Image Engine)")

st.markdown(
    "1️⃣ Upload voiceover → 2️⃣ Deepgram se transcript + timestamps → "
    "3️⃣ Groq se scene prompts → 4️⃣ Image engine se images → 5️⃣ Render video."
)

with st.sidebar:
    st.header("Render options")

    aspect_ratio = st.selectbox(
        "Image aspect ratio",
        ["16:9", "9:16", "1:1"],
        index=0,
    )

    engine_name = st.selectbox(
        "Image engine preset",
        ["1.5-Fast", "1.0-Slow"],
        index=0,
        help="Yeh sirf backend provider ko batata hai.",
    )

    style_choice = st.selectbox(
        "Visual style (for prompts)",
        [
            "None / neutral",
            "Black & white documentary stills",
            "Vintage film look, slightly grainy",
            "Cinematic, dramatic lighting",
        ],
        index=1,
    )

    if style_choice == "None / neutral":
        global_style = ""
    elif style_choice == "Black & white documentary stills":
        global_style = (
            "High-contrast black and white documentary photography, war-era style."
        )
    elif style_choice == "Vintage film look, slightly grainy":
        global_style = "Vintage film aesthetic, slightly grainy but clear subjects."
    else:
        global_style = "Cinematic, dramatic lighting, high detail, 4k still frame."

    color_mode = st.radio(
        "Video color mode",
        ["Color", "Black & white"],
        index=1,
    )

    zoom_strength = st.slider(
        "Zoom effect strength",
        0.0,
        0.15,
        0.05,
        step=0.01,
    )

    fps = st.slider(
        "Frames per second (FPS)",
        18,
        30,
        24,
    )

# -------------------------------------------------
# Step 1 – Audio upload + Deepgram
# -------------------------------------------------
st.header("1️⃣ Upload voiceover audio")

audio_file = st.file_uploader("Upload MP3 voiceover", type=["mp3"])

col_a1, col_a2 = st.columns(2)
with col_a1:
    analyze_btn = st.button("Analyze with Deepgram", type="primary", disabled=audio_file is None)

if analyze_btn and audio_file:
    st.session_state["video_path"] = None
    local_audio_path = save_uploaded_file(
        audio_file,
        BASE_TEMP_DIR,
        "input.mp3",
    )
    st.session_state["audio_path"] = local_audio_path

    with st.spinner("Sending to Deepgram..."):
        dg_json = deepgram_transcribe(local_audio_path)

    words = extract_words_from_deepgram(dg_json)
    chunks = build_chunks_from_words(words, max_chunk_sec=5.0)
    st.session_state["chunks"] = chunks
    st.session_state["scene_prompts"] = []
    st.session_state["scene_images"] = []

    st.success(f"Deepgram result loaded. Found {len(chunks)} scenes / chunks.")

# Show transcript + chunks debug
if st.session_state["chunks"]:
    with st.expander("🔍 Transcript chunks (Deepgram)", expanded=False):
        st.write(f"Total chunks: {len(st.session_state['chunks'])}")
        for c in st.session_state["chunks"]:
            st.markdown(
                f"**Scene {c['index']}** "
                f"({c['start']:.2f}s → {c['end']:.2f}s) – "
                f"Duration: {c['end'] - c['start']:.2f}s"
            )
            st.write(c["text"])

# -------------------------------------------------
# Step 2 – Groq prompts
# -------------------------------------------------
st.header("2️⃣ Scene prompts from Groq")

if st.session_state["chunks"]:
    if not st.session_state["scene_prompts"]:
        if st.button("Generate prompts with AI", type="primary"):
            with st.spinner("Groq se prompts mangwa rahe hain..."):
                prompts = groq_scene_prompts(st.session_state["chunks"], global_style)
            st.session_state["scene_prompts"] = prompts
            st.success("Scene prompts generated!")
    else:
        st.info("Prompts already generated. Aap niche edit kar sakte hain.")

if st.session_state["scene_prompts"]:
    st.subheader("✏️ Edit scene prompts (before generating images)")
    edited_prompts = []
    for c, prompt in zip(st.session_state["chunks"], st.session_state["scene_prompts"]):
        idx = c["index"]
        text_key = f"scene_prompt_{idx}"
        val = st.text_area(
            f"Scene {idx} ({c['start']:.2f}s–{c['end']:.2f}s)",
            value=prompt,
            key=text_key,
            height=80,
        )
        edited_prompts.append(val)

    st.session_state["scene_prompts"] = edited_prompts

# -------------------------------------------------
# Step 3 – Generate / edit / replace images per scene
# -------------------------------------------------
st.header("3️⃣ Generate images per scene")

if st.session_state["scene_prompts"]:
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        gen_images_btn = st.button(
            "Generate / Regenerate ALL images",
            help="Har scene ke current prompt se nayi image banayega.",
        )

    if gen_images_btn:
        st.session_state["scene_images"] = []  # wipe
        images = []
        progress = st.progress(0.0)

        for i, (c, prompt) in enumerate(
            zip(st.session_state["chunks"], st.session_state["scene_prompts"]), start=1
        ):
            status = st.empty()
            status.write(f"Generating image for Scene {c['index']}...")
            try:
                img_bytes = call_image_engine(prompt, aspect_ratio, engine_name)
                images.append(
                    {
                        "index": c["index"],
                        "prompt": prompt,
                        "image_bytes": img_bytes,
                        "source": "api",
                    }
                )
            except Exception as e:
                st.error(f"Scene {c['index']} image error: {e}")
                # placeholder black frame
                blank = np.zeros((480, 854, 3), dtype=np.uint8)
                _, buf = cv2.imencode(".png", blank)
                images.append(
                    {
                        "index": c["index"],
                        "prompt": prompt,
                        "image_bytes": buf.tobytes(),
                        "source": "error_placeholder",
                    }
                )
            progress.progress(i / len(st.session_state["scene_prompts"]))
            status.empty()

        st.session_state["scene_images"] = images
        st.success("All scene images generated / updated.")

# Per-scene controls: preview, prompt edit, regenerate, manual upload
if st.session_state["scene_images"]:
    st.subheader("🖼 Scene images – edit, regenerate, or replace manually")

    imgs = st.session_state["scene_images"]
    new_imgs = []

    for i, scene in enumerate(imgs):
        idx = scene["index"]
        c = st.session_state["chunks"][idx - 1]
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(
                f"**Scene {idx}** ({c['start']:.2f}s–{c['end']:.2f}s)"
                f" – Duration: {c['end'] - c['start']:.2f}s"
            )
            st.image(scene["image_bytes"], caption=f"Scene {idx}", width="stretch")

        with col2:
            # prompt is already in session_state["scene_prompts"]
            prompt_key = f"scene_prompt_{idx}"
            current_prompt = st.session_state["scene_prompts"][idx - 1]

            st.markdown("Current prompt:")
            st.text_area(
                f"Prompt S{idx}",
                value=current_prompt,
                key=prompt_key,
                height=100,
            )

            regen_btn = st.button(f"Regenerate S{idx}", key=f"regen_{idx}")
            upload_file = st.file_uploader(
                f"Replace S{idx} image",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"upload_{idx}",
            )

            updated_scene = dict(scene)

            # Handle manual upload (wins over regen if both triggered this run)
            if upload_file is not None:
                updated_scene["image_bytes"] = upload_file.read()
                updated_scene["source"] = "upload"

            # Handle regenerate
            if regen_btn and upload_file is None:
                try:
                    new_prompt = st.session_state[prompt_key]
                except KeyError:
                    new_prompt = current_prompt
                st.session_state["scene_prompts"][idx - 1] = new_prompt
                try:
                    img_bytes = call_image_engine(new_prompt, aspect_ratio, engine_name)
                    updated_scene["image_bytes"] = img_bytes
                    updated_scene["prompt"] = new_prompt
                    updated_scene["source"] = "api_regen"
                    st.success(f"Scene {idx} image regenerated.")
                except Exception as e:
                    st.error(f"Regenerate S{idx} failed: {e}")

            new_imgs.append(updated_scene)

        st.markdown("---")

    # save back
    st.session_state["scene_images"] = new_imgs

# -------------------------------------------------
# Step 4 – Final confirmation + render video
# -------------------------------------------------
st.header("4️⃣ Final review & render")

ready = (
    st.session_state["audio_path"]
    and st.session_state["chunks"]
    and st.session_state["scene_prompts"]
    and st.session_state["scene_images"]
)

if not ready:
    st.warning(
        "Voiceover, chunks, prompts, aur images sab complete honi chahiye "
        "pehle. Upar ke steps complete karo."
    )
else:
    # Debug: show summary
    with st.expander("📋 Final scene summary", expanded=False):
        for c, prompt, img in zip(
            st.session_state["chunks"],
            st.session_state["scene_prompts"],
            st.session_state["scene_images"],
        ):
            st.markdown(
                f"**Scene {c['index']}** ({c['start']:.2f}s–{c['end']:.2f}s) –"
                f" duration {(c['end'] - c['start']):.2f}s – source: {img['source']}"
            )
            st.write(prompt)

    confirm = st.checkbox(
        "I have reviewed prompts & images. Render final video.",
        value=False,
    )

    render_btn = st.button(
        "🎬 Render video now",
        type="primary",
        disabled=not confirm,
    )

    if render_btn and confirm:
        # Build scenes array with image bytes + timings
        scenes_for_render = []
        for c, img in zip(
            st.session_state["chunks"],
            st.session_state["scene_images"],
        ):
            scenes_for_render.append(
                {
                    "index": c["index"],
                    "start": c["start"],
                    "end": c["end"],
                    "image_bytes": img["image_bytes"],
                }
            )

        try:
            final_path = render_video(
                scenes_for_render,
                st.session_state["audio_path"],
                color_mode=color_mode,
                zoom_strength=zoom_strength,
                fps=fps,
            )
            st.session_state["video_path"] = final_path
            st.success("✅ Video render complete!")
        except Exception as e:
            st.error(f"Video render failed: {e}")

# Show final video if exists
if st.session_state["video_path"] and os.path.exists(st.session_state["video_path"]):
    st.subheader("🎞 Final video")
    st.video(st.session_state["video_path"], format="video/mp4")
    with open(st.session_state["video_path"], "rb") as f:
        st.download_button(
            "Download MP4",
            data=f.read(),
            file_name="ai_video.mp4",
            mime="video/mp4",
        )
