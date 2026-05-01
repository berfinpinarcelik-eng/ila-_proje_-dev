import streamlit as st
import os
from PIL import Image
import io
from agents import PharmaGuardAgents
from utils import create_pdf_report
from dotenv import load_dotenv

# Force reload environment
load_dotenv(override=True)

# Page Config
st.set_page_config(
    page_title="Pharma-Guard AI | Groq Edition",
    page_icon="💊",
    layout="wide"
)

# Helpers & Caching
@st.cache_resource
def get_agents():
    return PharmaGuardAgents()

# Session State
if "quota_hit" not in st.session_state:
    st.session_state.quota_hit = False

def check_api_keys():
    groq_ok = os.getenv("GROQ_API_KEY") and "your_" not in os.getenv("GROQ_API_KEY")
    return bool(groq_ok)

groq_ready = check_api_keys()
demo_active = not groq_ready

# Custom CSS
st.markdown("""
    <style>
    .main { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .report-card { background-color: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); border-left: 5px solid #007bff; }
    .status-badge { padding: 5px 15px; border-radius: 10px; font-size: 0.8em; font-weight: bold; color: white; }
    .ready { background-color: #28a745; }
    .warning { background-color: #ffc107; color: black; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/pill.png", width=150)
    st.title("Pharma-Guard AI")
    st.subheader("Groq/Llama Edition")
    st.markdown("---")
    
    st.markdown("### 🟢 Sistem Durumu")
    if groq_ready:
        st.markdown('<span class="status-badge ready">Groq/Llama 3.3: BAĞLI</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge warning">Groq: Demo Modu</span>', unsafe_allow_html=True)
    
    st.markdown('<span class="status-badge ready">Embeddings: Yerel Aktif</span>', unsafe_allow_html=True)

    if demo_active or st.session_state.quota_hit:
        st.markdown('<span class="status-badge warning">Mod: Otomatik Demo</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge ready">Mod: CANLI (GROQ)</span>', unsafe_allow_html=True)

# Main UI
st.header("💊 Pharma-Guard AI: Akıllı İlaç Denetçisi (GROQ V2)")
st.info("Sistem tamamen Groq/Llama altyapısına taşınmıştır. Gemini bağlantısı kaldırıldı.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Veri Girişi")
    uploaded_file = st.file_uploader("İlaç Kutusunun Fotoğrafını Seçin", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Görsel", width="stretch")
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format if image.format else "PNG")
        img_bytes = img_byte_arr.getvalue()

    if st.button("Analizi Başlat 🚀"):
        if uploaded_file is None:
            st.error("Lütfen önce bir görüntü yükleyin!")
        else:
            with col2:
                try:
                    with st.status("Ajanlar Çalışıyor (Groq)...", expanded=True) as status:
                        agents = get_agents()
                        report_markdown = agents.orchestrate(img_bytes)
                        
                        if "KOTA DOLDU" in report_markdown:
                            st.session_state.quota_hit = True
                            st.rerun()

                        status.update(label="Analiz Tamamlandı!", state="complete", expanded=False)
                        
                        st.markdown('<div class="report-card">', unsafe_allow_html=True)
                        st.markdown(report_markdown)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        pdf_path = create_pdf_report(report_markdown, f"rapor_{uploaded_file.name}.pdf")
                        with open(pdf_path, "rb") as f:
                            st.download_button("📄 PDF Raporu İndir", f, file_name=f"PharmaGuard_{uploaded_file.name}.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Hata: {str(e)}")
