import streamlit as st
import cv2
import numpy as np
import google.generativeai as genai
from deepgram import DeepgramClient, PrerecordedOptions
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from PIL import Image
import json
import subprocess
import os
import shutil
import time

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Video Generator", page_icon="🎬", layout="wide")

# --- LOAD SECRETS ---
try:
    DEEPGRAM_KEY = st.secrets["DEEPGRAM_API_KEY"]
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
    GCP_CREDS = st.secrets["gcp_service_account"]
    ROOT_FOLDER_ID = st.secrets["ROOT_FOLDER_ID"]
except:
    st.error("🚨 Secrets missing! Please configure .streamlit/secrets.toml")
    st.stop()

# --- TEMP FOLDERS ---
TEMP_DIR = "temp_processing"
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR) # Clean start
os.makedirs(TEMP_DIR)

# ==========================================
# 1. GOOGLE DRIVE FUNCTIONS (Auto-Storage)
# ==========================================
def get_drive_service():
    creds = service_account.Credentials.from_service_account_info(GCP_CREDS)
    return build('drive', 'v3', credentials=creds)

def get_or_create_subfolder(service, folder_name, parent_id):
    query = f"mimeType='application/vnd.google-apps.folder' and name='{folder_name}' and '{parent_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    
    if files:
        return files[0]['id']
    else:
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        return folder['id']

def upload_to_drive(file_path, file_name, folder_id):
    service = get_drive_service()
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    media = MediaFileUpload(file_path, mimetype='video/mp4', resumable=True)
    
    st.info(f"☁️ Uploading {file_name} to Google Drive...")
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return file.get('id')

# ==========================================
# 2. AI INTELLIGENCE (Vision + Hearing)
# ==========================================
def get_transcript(audio_path):
    """Deepgram se Audio -> Text + Timestamps"""
    deepgram = DeepgramClient(DEEPGRAM_KEY)
    with open(audio_path, "rb") as file:
        buffer_data = file.read()
    payload = {"buffer": buffer_data}
    options = PrerecordedOptions(model="nova-2", smart_format=True, punctuate=True)
    response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
    sentences = response.results.channels[0].alternatives[0].paragraphs.paragraphs[0].sentences
    return [{"text": s.text, "start": s.start, "end": s.end} for s in sentences]

def analyze_images_batch(image_files):
    """Gemini Vision: 5 Images per call to save Rate Limit"""
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    descriptions = {}
    batch_size = 5
    total_images = len(image_files)
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    # List of tuples (filename, file_object)
    img_list = [(img.name, img) for img in image_files]
    
    for i in range(0, total_images, batch_size):
        batch = img_list[i : i + batch_size]
        status.text(f"👁️ Analyzing Images {i+1} to {min(i+batch_size, total_images)}...")
        
        inputs = ["Describe each image briefly with its filename. Format: Filename: Description"]
        
        for name, file_obj in batch:
            file_obj.seek(0)
            img = Image.open(file_obj)
            inputs.append(f"Filename: {name}")
            inputs.append(img)
            
        try:
            # Single API Call for 5 Images
            response = model.generate_content(inputs)
            descriptions[f"Batch_{i}"] = response.text
            time.sleep(2) # Safety Sleep for Rate Limit
        except Exception as e:
            st.warning(f"Batch {i} failed: {e}")
            
        progress_bar.progress(min((i + batch_size) / total_images, 1.0))
        
    return descriptions

def create_sync_plan(script_text, image_descriptions_dict):
    """Gemini Brain: Match Script with Image Descriptions"""
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    desc_text = json.dumps(image_descriptions_dict)
    
    prompt = f"""
    SCRIPT: "{script_text}"
    IMAGE DESCRIPTIONS: {desc_text}
    
    TASK: Create a JSON timeline. Map images to the script based on context.
    FORMAT: [{{ "image": "filename.jpg", "duration_seconds": 5.0 }}]
    Ensure total duration matches script length roughly.
    RETURN ONLY JSON.
    """
    
    try:
        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except:
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
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    st.write("🎥 Rendering Video Frames... (Please wait)")
    my_bar = st.progress(0)
    
    # Calculate total frames for progress bar
    total_sec = sum([t['duration_seconds'] for t in timeline])
    total_frames_all = int(total_sec * fps)
    current_frame = 0
    
    for item in timeline:
        img_name = item['image']
        duration = item['duration_seconds']
        
        if img_name in image_map:
            # Convert UploadedFile to OpenCV Image
            image_map[img_name].seek(0)
            file_bytes = np.asarray(bytearray(image_map[img_name].read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            # Resize
            img = cv2.resize(img, (width, height))
            
            frames_in_clip = int(duration * fps)
            
            for i in range(frames_in_clip):
                # Light Zoom Effect (Maths)
                scale = 1.0 + (0.05 * i / frames_in_clip)
                M = cv2.getRotationMatrix2D((width//2, height//2), 0, scale)
                zoomed = cv2.warpAffine(img, M, (width, height))
                
                out.write(zoomed)
                current_frame += 1
                
                # Update UI every 50 frames
                if current_frame % 50 == 0:
                    my_bar.progress(min(current_frame / total_frames_all, 1.0))
        
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
st.markdown("Stack: **Deepgram** (Audio) + **Gemini** (Vision/Logic) + **OpenCV** (Render) + **G-Drive**")

col1, col2 = st.columns(2)
with col1:
    audio_file = st.file_uploader("Upload Voiceover (MP3)", type=["mp3"])
with col2:
    uploaded_images = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if st.button("🚀 Generate Video", type="primary"):
    if not audio_file or not uploaded_images:
        st.error("Please upload audio and images!")
    else:
        status = st.status("Starting Process...", expanded=True)
        
        # 1. Setup Drive
        status.write("📂 Setting up Google Drive...")
        service = get_drive_service()
        final_folder_id = get_or_create_subfolder(service, "AI_Final_Videos", ROOT_FOLDER_ID)
        
        # 2. Save Audio Local
        local_audio = os.path.join(TEMP_DIR, "input.mp3")
        with open(local_audio, "wb") as f:
            f.write(audio_file.getbuffer())
            
        # 3. Transcribe
        status.write("👂 Listening to Audio (Deepgram)...")
        transcript = get_transcript(local_audio)
        full_text = " ".join([t['text'] for t in transcript])
        
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
            avg_dur = len(transcript) / len(image_names) if transcript else 5
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
