import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import pandas as pd

# ═══════════════════════════════════════════════════════════
# 1. CẤU HÌNH GIAO DIỆN WEB & HẰNG SỐ
# ═══════════════════════════════════════════════════════════
st.set_page_config(page_title="AI Key Detector & Evaluator", page_icon="🎵", layout="wide")

TONE_CLASSES = [
    "C_Major",  "Db_Major", "D_Major",  "Eb_Major", "E_Major",  "F_Major",
    "F#_Major", "G_Major",  "Ab_Major", "A_Major",  "Bb_Major", "B_Major",
    "C_Minor",  "C#_Minor", "D_Minor",  "Eb_Minor", "E_Minor",  "F_Minor",
    "F#_Minor", "G_Minor",  "G#_Minor", "A_Minor",  "Bb_Minor", "B_Minor",
]
N_CLASSES = len(TONE_CLASSES)

# Định nghĩa 3 mô hình theo đúng cấu hình của bạn
MODELS_CONFIG = {
    "CRNN (Toàn bài)": "key_detector_crnn_fullbai_v2.pth",
    "CNN (30s Đầu)": "model_CNN_30fs.pth",
    "CNN (30s Cuối)": "model_CNN_30ls.pth"
}

# Khởi tạo Session State để lưu kết quả phân tích tránh load lại
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# ═══════════════════════════════════════════════════════════
# 2. ĐỊNH NGHĨA KIẾN TRÚC CÁC MẠNG NƠ-RON
# ═══════════════════════════════════════════════════════════

# ---- MẠNG CNN 13 BINS CỤC BỘ ----
class KeyCNN_13Bins(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32), nn.ELU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32), nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64), nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64), nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)), 
            
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64), nn.ELU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64), nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((13, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(832, N_CLASSES) 
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[2] > x.shape[3]:
            x = x.permute(0, 1, 3, 2)
        return self.head(self.features(x))

# ---- LỚP ATTENTION CHO CRNN ----
class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, lstm_outputs):
        attn_weights = self.attention(lstm_outputs)
        attn_weights = torch.softmax(attn_weights, dim=1)
        return torch.sum(attn_weights * lstm_outputs, dim=1)

# ---- MẠNG CRNN V3 (CÓ ATTENTION - 13 BINS) ----
class KeyCRNN_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.rnn = nn.LSTM(input_size=1664, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = SelfAttention(hidden_size=256)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(128, N_CLASSES),
        )

    def forward(self, x):
        B, T, F = x.shape
        x = x.unsqueeze(1)                    
        x = x.permute(0, 1, 3, 2)
        x = self.cnn(x)                       
        _, C, F_out, T_out = x.shape
        x = x.permute(0, 3, 1, 2)            
        x = x.reshape(B, T_out, C * F_out)          
        x, _ = self.rnn(x)                    
        x = self.attention(x)
        return self.head(x)

# ---- MẠNG CRNN V2 CŨ (KHÔNG ATTENTION - 25 BINS) ----
class KeyCRNN_V2(nn.Module):
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
        # Đã sửa input_size = 64 * 3 để đồng bộ với 13 Bins (Fix lỗi [512, 384] vs [512, 192])
        self.rnn = nn.LSTM(input_size=64 * 3, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
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

# ═══════════════════════════════════════════════════════════
# 3. TRÌNH NẠP MODEL SIÊU AN TOÀN (ANTI-CRASH)
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_model(model_name):
    model_path = MODELS_CONFIG[model_name]
    
    if not os.path.exists(model_path):
        return None, model_path, 13, "Lỗi: Không tìm thấy file"
        
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model_state", ckpt)
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        req_bins = 13
        
        # Auto-detect kiến trúc thông minh
        if "CRNN" in model_name:
            has_attention = any("attention" in k for k in clean_state_dict.keys())
            if has_attention:
                model = KeyCRNN_Attention()
            else:
                model = KeyCRNN_V2()
        else:
            model = KeyCNN_13Bins()
            
        model.load_state_dict(clean_state_dict, strict=False)
        model.eval()
        return model, model_path, req_bins, "Success"
        
    except Exception as e:
        return None, model_path, 13, f"Lỗi nạp trọng số: {e}"

# ═══════════════════════════════════════════════════════════
# 4. HÀM TRÍCH XUẤT ĐẶC TRƯNG DSP ĐỘNG (TỐI ƯU HÓA SLICING)
# ═══════════════════════════════════════════════════════════
def process_audio(file_bytes, model_name, req_bins):
    # Lọc thời gian chính xác cho từng loại mô hình
    if "30s Đầu" in model_name:
        y, sr = librosa.load(file_bytes, sr=22050, mono=True, duration=30.0)
    elif "30s Cuối" in model_name:
        # Load toàn bài để lấy được Outro chuẩn xác nhất
        y, sr = librosa.load(file_bytes, sr=22050, mono=True, duration=None)
        if len(y) > 30 * sr:
            y = y[-(30 * sr):]
    else:
        # CRNN (Toàn bài) - Giữ nguyên 60s như code cũ của bạn
        y, sr = librosa.load(file_bytes, sr=22050, mono=True, duration=60.0)

    # HPSS Lọc tạp âm
    y_harm = librosa.effects.harmonic(y, margin=4)
    
    # Ma trận 13 Bins
    chroma_full = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=512, bins_per_octave=36)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
    mel_mean = librosa.power_to_db(mel, ref=np.max).mean(axis=0, keepdims=True)
    
    if req_bins == 25:
        chroma_bass = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=512, fmin=librosa.note_to_hz('C1'), n_octaves=3, bins_per_octave=36)
        feat = np.concatenate([chroma_full, chroma_bass, mel_mean], axis=0).astype(np.float32)
    else:
        feat = np.concatenate([chroma_full, mel_mean], axis=0).astype(np.float32)
        
    # Min-Max Normalization
    for i in range(feat.shape[0]):
        mn, mx = feat[i].min(), feat[i].max()
        if mx > mn: 
            feat[i] = (feat[i] - mn) / (mx - mn)
    feat = feat.T 
    
    return feat, chroma_full

# ═══════════════════════════════════════════════════════════
# 5. GIAO DIỆN TƯƠNG TÁC
# ═══════════════════════════════════════════════════════════
st.title("🎵 Hệ Thống Phân Tích & Đánh Giá Tone Nhạc Hàng Loạt")

# --- SIDEBAR: CHỌN MÔ HÌNH ---
st.sidebar.header("⚙️ Cấu hình Hệ thống")
selected_model_name = st.sidebar.selectbox("Lựa chọn Mô hình dự đoán:", list(MODELS_CONFIG.keys()))

if st.sidebar.button("🧹 Xóa Cache & Tải lại Mô hình"):
    st.cache_resource.clear()
    st.session_state.predictions = {}
    st.rerun()

model, current_model_path, req_bins, load_status = load_model(selected_model_name)

if model is None:
    st.sidebar.error(f"❌ KHÔNG THỂ NẠP MÔ HÌNH: `{current_model_path}`")
    st.sidebar.warning(f"**Chi tiết lỗi:**\n{load_status}")
    st.sidebar.info("Ứng dụng sẽ sinh ra kết quả mô phỏng (Mock) để bạn test giao diện.")
else:
    st.sidebar.success(f"✅ Đã nạp mô hình: `{selected_model_name}`")
    st.sidebar.info(f"🧠 Cấu trúc trích xuất: `{req_bins} Bins`")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Hướng dẫn sử dụng:**
1. Upload nhiều bài hát cùng lúc.
2. Bấm "Phân tích tất cả".
3. Nghe thử và đánh giá Đúng/Sai.
4. Bấm "Tổng hợp Báo cáo".
""")

# --- MAIN AREA: UPLOAD & PROCESS ---
uploaded_files = st.file_uploader("📂 Tải lên danh sách bài hát (WAV, MP3)", type=["wav", "mp3"], accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 Bắt đầu Phân tích tất cả", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            file_name = file.name
            status_text.text(f"Đang phân tích: {file_name} ({i+1}/{len(uploaded_files)})...")
            
            cache_key = f"{file_name}_{selected_model_name}"
            
            if cache_key not in st.session_state.predictions:
                try:
                    feat, chroma_plot = process_audio(file, selected_model_name, req_bins)
                    
                    if model is not None:
                        x = torch.tensor(feat).unsqueeze(0).float()   
                        with torch.no_grad():
                            probs = torch.softmax(model(x), dim=1)[0]
                    else:
                        probs = torch.rand(N_CLASSES)
                        probs = torch.softmax(probs, dim=0)

                    top2_prob, top2_idx = torch.topk(probs, 2)
                    
                    st.session_state.predictions[cache_key] = {
                        "top1_tone": TONE_CLASSES[top2_idx[0].item()],
                        "top1_conf": top2_prob[0].item() * 100,
                        "top2_tone": TONE_CLASSES[top2_idx[1].item()],
                        "top2_conf": top2_prob[1].item() * 100,
                        "status": "success",
                        "chroma_plot": chroma_plot
                    }
                except Exception as e:
                    st.session_state.predictions[cache_key] = {"status": "error", "message": str(e)}
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.success("✅ Phân tích hoàn tất!")

# --- FEEDBACK AREA ---
if st.session_state.predictions and uploaded_files:
    st.markdown("### 📝 Đánh Giá Kết Quả")
    
    with st.form("evaluation_form"):
        eval_data = {} 
        
        for file in uploaded_files:
            file_name = file.name
            cache_key = f"{file_name}_{selected_model_name}"
            
            if cache_key in st.session_state.predictions:
                pred_info = st.session_state.predictions[cache_key]
                
                if pred_info["status"] == "error":
                    st.error(f"Lỗi phân tích bài {file_name}: {pred_info['message']}")
                    continue
                
                st.markdown(f"**Bài hát:** `{file_name}`")
                
                col_audio, col_pred, col_eval = st.columns([1, 1, 1.5])
                
                with col_audio:
                    st.audio(file, format='audio/wav')
                
                with col_pred:
                    st.info(f"**Dự đoán:** {pred_info['top1_tone']} ({pred_info['top1_conf']:.1f}%)\n\n"
                            f"*(Top 2: {pred_info['top2_tone']} - {pred_info['top2_conf']:.1f}%)*")
                
                with col_eval:
                    radio_val = st.radio("Đánh giá mô hình:", ["Đúng", "Sai", "Chưa kiểm tra"], index=2, horizontal=True, key=f"radio_{cache_key}")
                    
                    actual_key = pred_info['top1_tone'] 
                    if radio_val == "Sai":
                        actual_key = st.selectbox("Chọn Tone chuẩn xác:", TONE_CLASSES, key=f"correct_{cache_key}")
                    
                    eval_data[file_name] = {
                        "Dự đoán": pred_info['top1_tone'],
                        "Đánh giá": radio_val,
                        "Thực tế": actual_key if radio_val == "Sai" else pred_info['top1_tone']
                    }
                st.divider()
        
        submitted = st.form_submit_button("📊 Tính toán và Tổng hợp Báo cáo", type="primary")

# --- REPORT AREA ---
if 'submitted' in locals() and submitted:
    st.markdown("### 📈 Bảng Thống Kê Tổng Hợp")
    
    evaluated_records = []
    correct_count = 0
    
    for file_name, data in eval_data.items():
        if data["Đánh giá"] != "Chưa kiểm tra":
            evaluated_records.append({
                "Tên bài hát": file_name,
                "Mô hình dự đoán": data["Dự đoán"],
                "Tone thực tế": data["Thực tế"],
                "Kết quả": "✅ Đúng" if data["Đánh giá"] == "Đúng" else "❌ Sai"
            })
            if data["Đánh giá"] == "Đúng":
                correct_count += 1
                
    total_evaluated = len(evaluated_records)
    
    if total_evaluated > 0:
        accuracy = (correct_count / total_evaluated) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng số bài đã đánh giá", total_evaluated)
        col2.metric("Số bài dự đoán Đúng", correct_count)
        col3.metric("Độ chính xác (Accuracy)", f"{accuracy:.2f}%")
        
        df_report = pd.DataFrame(evaluated_records)
        st.dataframe(df_report, use_container_width=True)
        
        csv = df_report.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải Báo cáo (CSV)",
            data=csv,
            file_name=f'Report_{selected_model_name.replace(" ", "_")}.csv',
            mime='text/csv',
        )
    else:
        st.warning("Vui lòng đánh giá (Đúng/Sai) ít nhất 1 bài hát để xem báo cáo thống kê.")