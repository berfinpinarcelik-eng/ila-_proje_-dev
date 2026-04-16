import streamlit as st
import os
from PIL import Image
import io
from agents import PharmaGuardAgents
from utils import create_pdf_report
from dotenv import load_dotenv

load_dotenv()

# Page Config
st.set_page_config(
    page_title="Pharma-Guard AI | Akıllı İlaç Denetçisi",
    page_icon="💊",
    layout="wide"
)

# Helpers & Caching
@st.cache_resource
def get_agents():
    return PharmaGuardAgents()

def check_api_keys():
    google_ok = os.getenv("GOOGLE_API_KEY") and "your_" not in os.getenv("GOOGLE_API_KEY")
    groq_ok = os.getenv("GROQ_API_KEY") and "your_" not in os.getenv("GROQ_API_KEY")
    demo_mode = not (google_ok and groq_ok)
    return bool(google_ok), bool(groq_ok), demo_mode

google_ready, groq_ready, demo_active = check_api_keys()

# Custom CSS for Premium Design
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: scale(1.02);
    }
    .report-card {
        background-color: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #007bff;
    }
    .status-badge {
        padding: 5px 15px;
        border-radius: 10px;
        font-size: 0.8em;
        font-weight: bold;
        color: white;
    }
    .ready { background-color: #28a745; }
    .warning { background-color: #ffc107; color: black; }
    .error { background-color: #dc3545; }
    </style>
    """, unsafe_allow_html=True)

# App Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/pill.png", width=150)
    st.title("Pharma-Guard AI")
    st.markdown("---")
    st.info("📌 **Nasıl Çalışır?**\n1. İlaç kutusunun net bir fotoğrafını yükleyin.\n2. Ajanlar analize başlasın.\n3. RAG ile prospektüs doğrulaması yapılsın.\n4. Raporu indirin.")
    
    # Status Indicators
    st.markdown("### 🟢 Sistem Durumu")
    
    if google_ready:
        st.markdown('<span class="status-badge ready">Gemini 2.0: Bağlı</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge warning">Gemini 2.0: Demo Modu</span>', unsafe_allow_html=True)
        
    if groq_ready:
        st.markdown('<span class="status-badge ready">Groq/Llama3: Hazır</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge warning">Groq/Llama3: Demo Modu</span>', unsafe_allow_html=True)
        
    st.markdown('<span class="status-badge ready">Sistem: Aktif (Demo)</span>', unsafe_allow_html=True)

# Main Interface
st.header("💊 Pharma-Guard AI: Akıllı İlaç Denetçisi")
st.write("Yapay Zeka Destekli Çoklu Ajan Sistemi ile İlaç Doğrulama ve Analiz")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Veri Girişi")
    uploaded_file = st.file_uploader("İlaç Kutusunun Fotoğrafını Seçin veya Sürükleyin", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Görsel", use_container_width=True)
        
        # Convert image to bytes for processing
        img_byte_arr = io.BytesIO()
        save_format = image.format if image.format else "PNG"
        image.save(img_byte_arr, format=save_format)
        img_bytes = img_byte_arr.getvalue()

    # Button is always enabled now, but shows a warning if in demo mode
    process_btn = st.button("Analizi Başlat 🚀")
    
    if demo_active:
        st.info("ℹ️ **Demo Modu Aktif**: Gerçek API anahtarları bulunamadı. Sistem simülasyon üzerinden çalışacaktır.")

with col2:
    st.subheader("📋 Analiz Raporu")
    
    if process_btn:
        if uploaded_file is None:
            st.error("Lütfen önce bir görüntü yükleyin!")
        else:
            try:
                with st.status("Ajanlar Çalışıyor...", expanded=True) as status:
                    st.write("🔍 [Vision-Scanner] Görsel taranıyor...")
                    agents = get_agents()
                    
                    # Logic Execution
                    report_markdown = agents.orchestrate(img_bytes)
                    
                    st.write("📚 [RAG-Specialist] Prospektüs havuzu sorgulanıyor...")
                    st.write("🛡️ [Safety-Auditor] Güvenlik kontrolleri yapılıyor...")
                    st.write("📝 [Report-Synthesizer] Rapor sentezleniyor...")
                    status.update(label="Analiz Tamamlandı!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"❌ Analiz sırasında bir hata oluştu: {str(e)}")
                st.stop()

            # Display Report
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown(report_markdown)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # PDF Generation
            pdf_path = create_pdf_report(report_markdown, f"rapor_{uploaded_file.name}.pdf")
            
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📄 Profesyonel Analiz Raporunu İndir (PDF)",
                    data=f,
                    file_name=f"PharmaGuard_Report_{uploaded_file.name}.pdf",
                    mime="application/pdf"
                )
    else:
        st.info("Analiz sonuçları burada görünecektir. Lütfen sol taraftan 'Analizi Başlat' butonuna tıklayın.")

# Footer
st.markdown("---")
st.caption("⚠️ Uyarı: Bu sistem bir yapay zeka asistanıdır. Tıbbi kararlar almadan önce mutlaka bir doktora danışın.")
