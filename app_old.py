import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import librosa
import os
import sys
import asyncio

# --- THUỐC GIẢI TRỊ LỖI WINERROR 10054 TRÊN WINDOWS ---
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# -----------------------------------------------------

# ═══════════════════════════════════════════════════════════
# 1. CẤU HÌNH GIAO DIỆN WEB & CSS BẢNG (UI/UX)
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="Song Key Finder", page_icon="🎵", layout="wide")

# Tiêm CSS Custom để làm bảng giống y hệt thiết kế
st.markdown("""
<style>
    /* Bảng hiển thị kết quả */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 16px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        color: #ffffff;
        background-color: #1a1a2e;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        border-radius: 8px;
        overflow: hidden;
    }
    .styled-table thead tr {
        background-color: #7b61ff; /* Màu tím theo ảnh */
        color: #ffffff;
        text-align: center;
        font-weight: bold;
    }
    .styled-table th, .styled-table td {
        padding: 15px 20px;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #2d2d44;
        background-color: #1e1e32;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #25253b;
    }
    .styled-table tbody tr:hover {
        background-color: #33334c;
        transition: 0.3s;
    }
    .tone-highlight {
        font-weight: bold;
        color: #00d2d3;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo bộ nhớ (Session State) để lưu lại các bài hát đã phân tích
if 'analyzed_files' not in st.session_state:
    st.session_state.analyzed_files = {}

# ═══════════════════════════════════════════════════════════
# 2. ĐỊNH NGHĨA MẠNG CRNN & NHÃN (13 FEATURES)
# ═══════════════════════════════════════════════════════════
TONE_CLASSES = [
    "C Major",  "Db Major", "D Major",  "Eb Major", "E Major",  "F Major",
    "F# Major", "G Major",  "Ab Major", "A Major",  "Bb Major", "B Major",
    "C Minor",  "C# Minor", "D Minor",  "Eb Minor", "E Minor",  "F Minor",
    "F# Minor", "G Minor",  "G# Minor", "A Minor",  "Bb Minor", "B Minor",
]
N_CLASSES = len(TONE_CLASSES)
MODEL_PATH = "key_detector_crnn_fullbai_v2.pth"

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
            input_size=64 * 3, 
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
# 3. HÀM XỬ LÝ DSP & FORMAT SỐ LIỆU
# ═══════════════════════════════════════════════════════════
def process_audio(file_bytes):
    # Rút gọn lại, bỏ hoàn toàn Matplotlib
    y, sr = librosa.load(file_bytes, sr=22050, mono=True, duration=30) 
    y_harm = librosa.effects.harmonic(y, margin=4)
    
    chroma_full = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=512, bins_per_octave=36)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    mel_mean = librosa.power_to_db(mel, ref=np.max).mean(axis=0, keepdims=True)
    
    feat = np.concatenate([chroma_full, mel_mean], axis=0).astype(np.float32)
    for i in range(feat.shape[0]):
        mn, mx = feat[i].min(), feat[i].max()
        if mx > mn: 
            feat[i] = (feat[i] - mn) / (mx - mn)
    return feat.T 

def format_confidence(conf_val):
    """
    Hàm làm tròn thông minh:
    - Nếu >= 99.9% -> Lấy 99.9%
    - Còn lại: Lấy tối đa 5 số sau dấu phẩy, tự động cắt bỏ các số 0 vô nghĩa ở đuôi
    """
    if conf_val >= 99.9:
        return "99.9%"
    else:
        # Format 5 số thập phân, sau đó rstrip cắt bỏ số 0 và dấu . dư thừa
        formatted = f"{conf_val:.5f}".rstrip('0').rstrip('.')
        return f"{formatted}%"

# ═══════════════════════════════════════════════════════════
# 4. GIAO DIỆN CHÍNH
# ═══════════════════════════════════════════════════════════
st.markdown("<h1 style='text-align: center; color: #1abc9c;'>Song Key Finder</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #bdc3c7; font-size:18px;'>Analyzer by AI to find the key of any song</p>", unsafe_allow_html=True)
st.write("---")

if model is None:
    st.error(f"❌ Không tìm thấy file model: {MODEL_PATH}")
else:
    # Cho phép tải lên nhiều file cùng lúc
    uploaded_files = st.file_uploader("Thêm file âm thanh (Add tracks)", type=["wav", "mp3"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            # Chỉ phân tích những file chưa có trong bộ nhớ tạm
            if file.name not in st.session_state.analyzed_files:
                with st.spinner(f"⏳ Đang phân tích: {file.name}..."):
                    try:
                        feat = process_audio(file)
                        x = torch.tensor(feat).unsqueeze(0).float()   
                        
                        with torch.no_grad():
                            probs = torch.softmax(model(x), dim=1)[0]
                        
                        top2_prob, top2_idx = torch.topk(probs, 2)
                        
                        top1_name = TONE_CLASSES[top2_idx[0].item()]
                        top1_conf = top2_prob[0].item() * 100
                        
                        top2_name = TONE_CLASSES[top2_idx[1].item()]
                        top2_conf = top2_prob[1].item() * 100
                        
                        # Lưu kết quả vào bộ nhớ
                        st.session_state.analyzed_files[file.name] = {
                            "top1_name": top1_name,
                            "top1_conf": format_confidence(top1_conf),
                            "top2_name": top2_name,
                            "top2_conf": format_confidence(top2_conf)
                        }
                    except Exception as e:
                        st.error(f"Lỗi khi đọc file {file.name}: {e}")

    # Nếu đã có dữ liệu trong bộ nhớ, tiến hành vẽ Bảng HTML
    # Nếu đã có dữ liệu trong bộ nhớ, tiến hành vẽ Bảng HTML
    if st.session_state.analyzed_files:
        st.markdown("<br>", unsafe_allow_html=True) # Tạo khoảng trống
        
        # Bắt đầu chuỗi HTML tạo bảng (Ép phẳng hoàn toàn để không bị lỗi Markdown)
        table_html = '<table class="styled-table">\n'
        table_html += '<thead>\n<tr>\n<th>File</th>\n<th>Top 1 Key</th>\n<th>Top 2 Key</th>\n</tr>\n</thead>\n'
        table_html += '<tbody>\n'
        
        # Duyệt qua các bài hát đã lưu và điền vào hàng
        for filename, data in st.session_state.analyzed_files.items():
            table_html += '<tr>\n'
            table_html += f'<td>{filename}</td>\n'
            table_html += f'<td><span class="tone-highlight">{data["top1_name"]}</span> <br> <span style="font-size:12px; color:#a4b0be;">({data["top1_conf"]})</span></td>\n'
            table_html += f'<td>{data["top2_name"]} <br> <span style="font-size:12px; color:#a4b0be;">({data["top2_conf"]})</span></td>\n'
            table_html += '</tr>\n'
            
        # Đóng thẻ bảng
        table_html += '</tbody>\n</table>'
        
        # Hiển thị bảng lên Web
        st.markdown(table_html, unsafe_allow_html=True)