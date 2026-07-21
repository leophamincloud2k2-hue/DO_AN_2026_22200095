import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

# ═══════════════════════════════════════════════════════════
# 1. CẤU HÌNH GIAO DIỆN WEB
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="AI Key Detector", page_icon="🎵", layout="centered")

st.title("🎵 AI Music Key Detector")
st.markdown("""
**Chào mừng đến với hệ thống nhận diện Tone nhạc bằng Trí tuệ Nhân tạo!**
Hệ thống sử dụng mạng nơ-ron tích chập hồi quy (CRNN) kết hợp phân tích dải tần số kép (Full-Chroma & Bass-Chroma).
""")

# ═══════════════════════════════════════════════════════════
# 2. ĐỊNH NGHĨA MẠNG CRNN & NHÃN
# ═══════════════════════════════════════════════════════════
TONE_CLASSES = [
    "C_Major",  "Db_Major", "D_Major",  "Eb_Major", "E_Major",  "F_Major",
    "F#_Major", "G_Major",  "Ab_Major", "A_Major",  "Bb_Major", "B_Major",
    "C_Minor",  "C#_Minor", "D_Minor",  "Eb_Minor", "E_Minor",  "F_Minor",
    "F#_Minor", "G_Minor",  "G#_Minor", "A_Minor",  "Bb_Minor", "B_Minor",
]
N_CLASSES = len(TONE_CLASSES)
MODEL_PATH = "key_detector_crnn_fullbai_v2.pth"  # Đảm bảo file này nằm cùng thư mục

class KeyCRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.rnn = nn.LSTM(
            input_size=64 * 6, # Hỗ trợ 25 features 
            hidden_size=128, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3,
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(64, N_CLASSES),
        )

    def forward(self, x):
        B, T, F = x.shape
        x = x.unsqueeze(1)                    
        x = self.cnn(x)                       
        _, C, T2, W = x.shape
        x = x.permute(0, 2, 1, 3)            
        x = x.reshape(B, T2, C * W)          
        x, _ = self.rnn(x)                    
        x = x.mean(dim=1)                     
        return self.head(x)

# Cache model để không phải load lại mỗi khi ấn nút trên Web
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = KeyCRNN()
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

model = load_model()

# ═══════════════════════════════════════════════════════════
# 3. HÀM TRÍCH XUẤT ĐẶC TRƯNG DSP TỪ FILE UPLOAD
# ═══════════════════════════════════════════════════════════
def process_audio(file_bytes):
    # Load audio từ bộ nhớ tạm (upload) thay vì từ ổ cứng
    y, sr = librosa.load(file_bytes, sr=22050, mono=True, duration=60) # Cắt 60s cho Web chạy nhanh
    
    y_harm = librosa.effects.harmonic(y, margin=4)
    
    # DSP: Trích xuất 25 Features
    chroma_full = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=512, bins_per_octave=36)
    chroma_bass = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'), n_octaves=3, bins_per_octave=36)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    mel_mean = librosa.power_to_db(mel, ref=np.max).mean(axis=0, keepdims=True)
    
    feat = np.concatenate([chroma_full, chroma_bass, mel_mean], axis=0).astype(np.float32)
    
    for i in range(feat.shape[0]):
        mn, mx = feat[i].min(), feat[i].max()
        if mx > mn: 
            feat[i] = (feat[i] - mn) / (mx - mn)
    feat = feat.T 
    
    return feat, chroma_full # Trả về chroma_full để vẽ đồ thị

# ═══════════════════════════════════════════════════════════
# 4. GIAO DIỆN TƯƠNG TÁC
# ═══════════════════════════════════════════════════════════
if model is None:
    st.error(f"❌ Không tìm thấy file model: {MODEL_PATH}. Vui lòng copy file .pth vào cùng thư mục với app.py!")
else:
    uploaded_file = st.file_uploader("📂 Tải lên bài hát (Định dạng: WAV, MP3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("🚀 Bắt đầu Phân tích Tone", use_container_width=True):
            with st.spinner("⏳ Đang xử lý DSP và chạy AI phân tích..."):
                try:
                    # Chạy DSP
                    feat, chroma_plot = process_audio(uploaded_file)
                    
                    # Chạy Model Inference
                    x = torch.tensor(feat).unsqueeze(0).float()   
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)[0]
                    
                    # Lấy kết quả Top 1 và Top 2
                    top2_prob, top2_idx = torch.topk(probs, 2)
                    
                    pred_tone_1 = TONE_CLASSES[top2_idx[0].item()]
                    conf_1 = top2_prob[0].item() * 100
                    
                    pred_tone_2 = TONE_CLASSES[top2_idx[1].item()]
                    conf_2 = top2_prob[1].item() * 100

                    # Hiển thị Kết quả
                    st.success("✅ Phân tích hoàn tất!")
                    st.markdown("### 🏆 Kết quả Nhận diện:")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Top 1:** {pred_tone_1}\n\n**Độ tin cậy:** {conf_1:.1f}%")
                    with col2:
                        st.warning(f"**Top 2 (Dự phòng):** {pred_tone_2}\n\n**Độ tin cậy:** {conf_2:.1f}%")

                    # Vẽ đồ thị biểu diễn phổ âm nhạc
                    st.markdown("### 📊 Bản đồ Đặc trưng Tần số (Chromagram)")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    img = librosa.display.specshow(chroma_plot, y_axis='chroma', x_axis='time', ax=ax, cmap='coolwarm')
                    fig.colorbar(img, ax=ax)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"❌ Xảy ra lỗi trong quá trình phân tích: {e}")