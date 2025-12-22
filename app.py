import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from PIL import Image
import json
import subprocess
import os
import shutil
import time
import requests  # Deepgram REST API ke liye

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Video Generator", page_icon="🎬", layout="wide")

# --- LOAD SECRETS ---
try:
    DEEPGRAM_KEY = st.secrets["DEEPGRAM_API_KEY"]
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    GCP_CREDS = st.secrets["gcp_service_account"]
    ROOT_FOLDER_ID = st.secrets["ROOT_FOLDER_ID"]
except Exception as e:
    st.error("🚨 Secrets missing! Please configure .streamlit/secrets.toml / Cloud Secrets.")
    st.stop()

# --- TEMP FOLDERS ---
TEMP_DIR = "temp_processing"
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)  # Clean start every run
os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================================
# 1. GOOGLE DRIVE FUNCTIONS (Auto-Storage)
# ==========================================
def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(GCP_CREDS)
    return build("drive", "v3", credentials=creds)


def get_or_create_subfolder(service, folder_name, parent_id):
    query = (
        f"mimeType='application/vnd.google-apps.folder' "
        f"and name='{folder_name}' and '{parent_id}' in parents and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]
    else:
        file_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = service.files().create(body=file_metadata, fields="id").execute()
        return folder["id"]


def upload_to_drive(file_path, file_name, folder_id):
    service = get_drive_service()
    file_metadata = {"name": file_name, "parents": [folder_id]}
    media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)

    st.info(f"☁️ Uploading {file_name} to Google Drive...")
    file = (
        service.files()
        .create(body=file_metadata, media_body=media, fields="id")
        .execute()
    )
    return file.get("id")


# ==========================================
# 2. AI INTELLIGENCE (Vision + Hearing)
# ==========================================
def get_transcript(audio_path):
    """
    Deepgram REST API se Audio -> Text
    (SDK ke bina, isliye Streamlit par import error nahi aayega)
    """
    url = "https://api.deepgram.com/v1/listen"

    headers = {
        "Authorization": f"Token {DEEPGRAM_KEY}",
        "Content-Type": "audio/mpeg",  # mp3 ke liye
    }

    params = {
        "model": "nova-2",       # agar account support kare to 'nova-3' bhi use kar sakte ho
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
        return []

    text = alt.get("transcript", "").strip()
    if not text:
        return []

    words = alt.get("words", [])
    start = words[0]["start"] if words else 0.0
    end = words[-1]["end"] if words else 0.0

    # Tumhara baaki code list-of-dicts expect karta hai
    return [{"text": text, "start": start, "end": end}]


def analyze_images_batch(image_files):
    """Gemini Vision: 5 Images per call to save Rate Limit"""
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    descriptions = {}
    batch_size = 5
    total_images = len(image_files)

    if total_images == 0:
        return descriptions

    progress_bar = st.progress(0)
    status = st.empty()

    # List of tuples (filename, file_object)
    img_list = [(img.name, img) for img in image_files]

    for i in range(0, total_images, batch_size):
        batch = img_list[i : i + batch_size]
        status.text(
            f"👁️ Analyzing Images {i+1} to {min(i+batch_size, total_images)}..."
        )

        inputs = [
            "Describe each image briefly with its filename. Format: Filename: Description"
        ]

        for name, file_obj in batch:
            file_obj.seek(0)
            img = Image.open(file_obj)
            inputs.append(f"Filename: {name}")
            inputs.append(img)

        try:
            # Single API Call for up to 5 Images
            response = model.generate_content(inputs)
            descriptions[f"Batch_{i}"] = response.text
            time.sleep(2)  # Safety Sleep for Rate Limit
        except Exception as e:
            st.warning(f"Batch {i} failed: {e}")

        progress_bar.progress(min((i + batch_size) / total_images, 1.0))

    return descriptions


def create_sync_plan(script_text, image_descriptions_dict):
    """Gemini Brain: Match Script with Image Descriptions"""
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

    desc_text = json.dumps(image_descriptions_dict)

    prompt = f"""
    SCRIPT: "{script_text}"
    IMAGE DESCRIPTIONS: {desc_text}

    TASK: Create a JSON timeline. Map images to the script based on context.
    FORMAT: [{{ "image": "filename.jpg", "duration_seconds": 5.0 }}]
    Ensure total duration roughly matches script length.
    RETURN ONLY JSON.
    """

    try:
        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except Exception as e:
        st.warning(f"⚠️ Plan creation failed, using fallback. Error: {e}")
        return []


# ==========================================
# 3. VIDEO RENDERING (OpenCV + FFmpeg)
# ==========================================
def render_video(timeline, audio_path, image_map, output_name="final_video.mp4"):
    # Settings: 480p is BEST for free cloud servers (Fast & Low RAM)
    width, height = 854, 480
    fps = 24

    temp_video = os.path.join(TEMP_DIR, "temp_silent.mp4")
    final_output = os.path.join(TEMP_DIR, output_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    st.write("🎥 Rendering Video Frames...")
    my_bar = st.progress(0)

    # Calculate total frames for progress bar
    total_sec = sum([t["duration_seconds"] for t in timeline])
    total_frames_all = max(int(total_sec * fps), 1)
    current_frame = 0

    for item in timeline:
        img_name = item["image"]
        duration = item["duration_seconds"]

        if img_name not in image_map:
            continue

        # Convert UploadedFile to OpenCV Image
        image_map[img_name].seek(0)
        file_bytes = np.asarray(bytearray(image_map[img_name].read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        if img is None:
            continue

        # Resize
        img = cv2.resize(img, (width, height))

        frames_in_clip = max(int(duration * fps), 1)

        for i in range(frames_in_clip):
            # Light Zoom Effect
            scale = 1.0 + (0.05 * i / frames_in_clip)
            M = cv2.getRotationMatrix2D((width // 2, height // 2), 0, scale)
            zoomed = cv2.warpAffine(img, M, (width, height))

            out.write(zoomed)
            current_frame += 1

            # Update UI every 50 frames
            if current_frame % 50 == 0:
                my_bar.progress(
                    min(current_frame / float(total_frames_all), 1.0)
                )

    out.release()

    # FFmpeg Merge (RAM Safe Method)
    st.write("🎵 Merging Audio...")
    command = f'ffmpeg -y -i "{temp_video}" -i "{audio_path}" -c:v copy -c:a aac -shortest "{final_output}"'
    subprocess.run(command, shell=True)

    return final_output


# ==========================================
# 4. MAIN UI LOGIC
# ==========================================
st.title("🤖 Free AI Video Generator")
st.markdown(
    "Stack: **Deepgram** (Audio) + **Gemini** (Vision/Logic) + **OpenCV** (Render) + **Google Drive**"
)

col1, col2 = st.columns(2)
with col1:
    audio_file = st.file_uploader("Upload Voiceover (MP3)", type=["mp3"])
with col2:
    uploaded_images = st.file_uploader(
        "Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True
    )

if st.button("🚀 Generate Video", type="primary"):
    if not audio_file or not uploaded_images:
        st.error("Please upload both audio and images!")
    else:
        status = st.status("Starting Process...", expanded=True)

        # 1. Setup Drive
        status.write("📂 Setting up Google Drive...")
        service = get_drive_service()
        final_folder_id = get_or_create_subfolder(
            service, "AI_Final_Videos", ROOT_FOLDER_ID
        )

        # 2. Save Audio Local
        local_audio = os.path.join(TEMP_DIR, "input.mp3")
        with open(local_audio, "wb") as f:
            f.write(audio_file.getbuffer())

        # 3. Transcribe
        status.write("👂 Listening to Audio (Deepgram)...")
        transcript = get_transcript(local_audio)
        if not transcript:
            st.error("❌ Could not get transcript from audio.")
            st.stop()

        full_text = " ".join([t["text"] for t in transcript])

        # 4. Analyze Images
        status.write("👁️ Analyzing Images (Gemini Vision Batch)...")
        image_desc = analyze_images_batch(uploaded_images)

        # 5. Plan Video
        status.write("🧠 Planning Scenes (Gemini Logic)...")
        plan = create_sync_plan(full_text, image_desc)

        # Fallback if AI fails
        if not plan:
            status.write("⚠️ AI Plan failed, using simple slideshow.")
            image_names = [img.name for img in uploaded_images]
            avg_dur = 5.0
            try:
                avg_dur = max(len(full_text.split()) / 2.5 / len(image_names), 1.0)
            except Exception:
                pass
            plan = [{"image": name, "duration_seconds": avg_dur} for name in image_names]

        # 6. Render
        status.write("🎬 Rendering Video (OpenCV)...")
        image_map = {img.name: img for img in uploaded_images}
        final_vid_path = render_video(plan, local_audio, image_map)

        # 7. Upload
        status.write("☁️ Uploading to Drive...")
        file_id = upload_to_drive(final_vid_path, "AI_Gen_Video.mp4", final_folder_id)

        status.update(label="✅ Video Completed!", state="complete", expanded=False)
        st.success(f"Video Saved to Google Drive! (File ID: {file_id})")
        st.video(final_vid_path)
