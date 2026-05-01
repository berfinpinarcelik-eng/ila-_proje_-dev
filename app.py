import streamlit as st
import os
import json
 
# ── Page config (must be first) ──
st.set_page_config(
    page_title="LearnerAI Coach",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800;900&display=swap');
 
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #07101f !important;
    color: #e2e8f0 !important;
}
 
/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2rem 2rem 2rem !important; max-width: 900px; }
 
/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3d5fc4, #5b7fff) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    transition: opacity 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
 
/* Selectbox & Multiselect */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #0d1a2e !important;
    border: 1px solid rgba(91,127,255,0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}
 
/* Text area */
.stTextArea textarea {
    background: #0d1a2e !important;
    border: 1px solid rgba(91,127,255,0.25) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
 
/* Text input (chat) */
.stChatInput > div {
    background: #0d1a2e !important;
    border: 1px solid rgba(91,127,255,0.25) !important;
    border-radius: 14px !important;
}
 
/* Select slider */
.stSlider { color: #5b7fff !important; }
 
/* Cards */
.card {
    background: rgba(13,26,46,0.85);
    border: 1px solid rgba(91,127,255,0.18);
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.card-accent {
    background: rgba(91,127,255,0.07);
    border: 1px solid rgba(91,127,255,0.25);
    border-radius: 14px;
    padding: 14px 18px;
    margin: 8px 0;
}
.tag {
    display: inline-block;
    background: rgba(91,127,255,0.12);
    color: #818cf8;
    border: 1px solid rgba(91,127,255,0.25);
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 700;
    margin: 3px;
}
.tag-green {
    background: rgba(74,222,128,0.1);
    color: #4ade80;
    border-color: rgba(74,222,128,0.2);
}
.tag-amber {
    background: rgba(245,158,11,0.1);
    color: #f59e0b;
    border-color: rgba(245,158,11,0.2);
}
.lesson-row {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(91,127,255,0.14);
    border-radius: 14px;
    padding: 14px 18px;
    margin: 8px 0;
    transition: border-color 0.2s;
}
.chat-bubble-ai {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.65;
}
.chat-bubble-user {
    background: linear-gradient(135deg, #3d5fc4, #5b7fff);
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 14px;
    line-height: 1.65;
    text-align: right;
}
.header-logo {
    font-size: 28px;
    font-weight: 900;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.step-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #5b7fff;
    margin-bottom: 6px;
}
.muted { color: #64748b; font-size: 13px; }
</style>
""", unsafe_allow_html=True)
 
# ── Groq client (lazy init) ──
def get_client():
    try:
        from groq import Groq
        key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
        if not key:
            return None
        return Groq(api_key=key)
    except Exception:
        return None
 
def call_groq(messages, json_mode=False):
    client = get_client()
    if not client:
        st.error("⚠️ GROQ_API_KEY bulunamadı. Streamlit Cloud → Settings → Secrets bölümüne ekleyin.")
        st.stop()
    kwargs = dict(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content
 
# ── Session state init ──
if "step" not in st.session_state:
    st.session_state.step = 1
if "grade" not in st.session_state:
    st.session_state.grade = ""
if "subjects" not in st.session_state:
    st.session_state.subjects = []
if "goal" not in st.session_state:
    st.session_state.goal = ""
if "days" not in st.session_state:
    st.session_state.days = "5"
if "plan" not in st.session_state:
    st.session_state.plan = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_initialized" not in st.session_state:
    st.session_state.chat_initialized = False
 
DAYS_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
 
# ════════════════════════════════════════
# ONBOARDING
# ════════════════════════════════════════
if st.session_state.plan is None:
 
    col_c, col_r = st.columns([2, 1])
    with col_c:
        st.markdown('<div class="header-logo">🎓 LearnerAI Coach</div>', unsafe_allow_html=True)
        st.markdown('<div class="muted">Kişisel AI eğitim asistanın — planını birlikte oluşturalım.</div>', unsafe_allow_html=True)
 
    st.markdown("---")
 
    # ── STEP 1 ──
    if st.session_state.step == 1:
        st.markdown('<div class="step-label">Adım 1 / 3</div>', unsafe_allow_html=True)
        st.markdown("### Hangi sınıftasın? 🎓")
        st.markdown('<div class="muted">Seviyene göre en uygun içerikleri hazırlayacağım.</div>', unsafe_allow_html=True)
        st.markdown("")
 
        grade = st.selectbox(
            "Sınıf seçin",
            ["", "İlkokul (1-4. Sınıf)", "Ortaokul 5. Sınıf", "Ortaokul 6. Sınıf",
             "Ortaokul 7. Sınıf", "Ortaokul 8. Sınıf", "Lise 9. Sınıf", "Lise 10. Sınıf",
             "Lise 11. Sınıf", "Lise 12. Sınıf", "Üniversite"],
            label_visibility="collapsed"
        )
        st.markdown("")
        if st.button("Devam Et →", disabled=(grade == "")):
            st.session_state.grade = grade
            st.session_state.step = 2
            st.rerun()
 
    # ── STEP 2 ──
    elif st.session_state.step == 2:
        st.markdown('<div class="step-label">Adım 2 / 3</div>', unsafe_allow_html=True)
        st.markdown("### Hangi dersleri çalışmak istiyorsun? 📚")
        st.markdown('<div class="muted">Birden fazla seçebilirsin.</div>', unsafe_allow_html=True)
        st.markdown("")
 
        subjects = st.multiselect(
            "Ders seçin",
            ["Matematik", "Fizik", "Kimya", "Biyoloji", "Türkçe",
             "Tarih", "Coğrafya", "İngilizce", "Python Programlama", "Veri Bilimi", "Felsefe"],
            label_visibility="collapsed"
        )
        st.markdown("")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("← Geri"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Devam Et →", disabled=(len(subjects) == 0)):
                st.session_state.subjects = subjects
                st.session_state.step = 3
                st.rerun()
 
    # ── STEP 3 ──
    elif st.session_state.step == 3:
        st.markdown('<div class="step-label">Adım 3 / 3</div>', unsafe_allow_html=True)
        st.markdown("### Hedefin ne? 🎯")
        st.markdown('<div class="muted">Kısa bir hedef yaz — planını buna göre kişiselleştireceğim.</div>', unsafe_allow_html=True)
        st.markdown("")
 
        goal = st.text_area(
            "Hedef",
            placeholder="Örn: YKS'ye hazırlanıyorum, matematikten yüksek puan almak istiyorum...",
            label_visibility="collapsed",
            height=100
        )
        days = st.select_slider(
            "Haftada kaç gün çalışmak istiyorsun?",
            options=["3", "5", "7"],
            value="5"
        )
        st.markdown("")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("← Geri"):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("✨ Planımı Oluştur"):
                st.session_state.goal = goal or "Derslerimi geliştirmek istiyorum"
                st.session_state.days = days
 
                with st.spinner("🤖 AI planın hazırlanıyor... (15-20 saniye)"):
                    prompt = f"""Sen bir Türkçe AI eğitim koçusun.
Sınıf: {st.session_state.grade}
Dersler: {', '.join(st.session_state.subjects)}
Hedef: {st.session_state.goal}
Haftada çalışma günü: {days}
 
SADECE aşağıdaki JSON formatında yanıt ver:
{{
  "headline": "kısa motivasyon cümlesi max 8 kelime",
  "subline": "planı açıklayan 1 cümle",
  "weeklyPlan": {{
    "Pazartesi": [],
    "Salı": [],
    "Çarşamba": [],
    "Perşembe": [],
    "Cuma": [],
    "Cumartesi": [],
    "Pazar": []
  }},
  "tips": ["ipucu1", "ipucu2", "ipucu3"]
}}
 
Her günün dizisindeki ders objesi:
{{"subject":"DersAdı", "topic":"KonuBaşlığı", "duration":"X dk", "color":"#hexrenk"}}
 
Kurallar:
- Sadece şu dersleri ekle: {', '.join(st.session_state.subjects)}
- Tam olarak {days} güne dersleri yay, diğer günler boş []
- {st.session_state.grade} seviyesine uygun Türkçe konular
- Her ders için farklı hex renk"""
                    try:
                        raw = call_groq([{"role": "user", "content": prompt}], json_mode=True)
                        st.session_state.plan = json.loads(raw)
                        st.session_state.chat_initialized = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Plan oluşturulamadı: {e}")
 
# ════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════
else:
    plan = st.session_state.plan
 
    # ── Header ──
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(f'<div class="header-logo">🎓 LearnerAI</div>', unsafe_allow_html=True)
        st.markdown(f"**{plan['headline']}**")
        st.markdown(f'<div class="muted">{plan["subline"]}</div>', unsafe_allow_html=True)
    with col_h2:
        st.markdown(f'<div class="card" style="text-align:center;padding:12px;"><div class="muted">Sınıf</div><div style="font-weight:800;font-size:14px;">{st.session_state.grade}</div></div>', unsafe_allow_html=True)
        if st.button("🔄 Yeni Plan"):
            for k in ["plan", "step", "grade", "subjects", "goal", "days", "messages", "chat_initialized"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
 
    st.markdown("---")
 
    # ── Two column layout ──
    left, right = st.columns([3, 2])
 
    # ── LEFT: Weekly Plan ──
    with left:
        st.markdown("#### 📅 Haftalık Planın")
 
        tabs = st.tabs(DAYS_TR)
        for i, tab in enumerate(tabs):
            with tab:
                day_name = DAYS_TR[i]
                lessons = plan.get("weeklyPlan", {}).get(day_name, [])
                if not lessons:
                    st.markdown('<div class="muted" style="padding:12px 0;">Bu gün dinlenme günü 🌙</div>', unsafe_allow_html=True)
                else:
                    for l in lessons:
                        color = l.get("color", "#5b7fff")
                        st.markdown(f"""
                        <div class="lesson-row">
                            <div style="font-size:11px;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:.06em;">{l.get('subject','')}</div>
                            <div style="font-size:15px;font-weight:800;margin:4px 0;">{l.get('topic','')}</div>
                            <div style="font-size:12px;color:#64748b;">⏱ {l.get('duration','')}</div>
                        </div>
                        """, unsafe_allow_html=True)
 
        st.markdown("")
        st.markdown("#### 💡 AI Koçundan İpuçları")
        for i, tip in enumerate(plan.get("tips", [])):
            st.markdown(f"""
            <div class="card-accent" style="display:flex;gap:12px;align-items:flex-start;">
                <span style="font-weight:800;color:#818cf8;font-size:13px;">{i+1}.</span>
                <span style="font-size:13.5px;">{tip}</span>
            </div>
            """, unsafe_allow_html=True)
 
    # ── RIGHT: AI Chat ──
    with right:
        st.markdown("#### 💬 AI Koçun")
        st.markdown('<div class="muted" style="margin-bottom:12px;">Herhangi bir konuyu sorabilirsin</div>', unsafe_allow_html=True)
 
        # Chat geçmişini göster
        chat_container = st.container(height=380)
        with chat_container:
            if not st.session_state.chat_initialized:
                dersler = ", ".join(st.session_state.subjects)
                welcome = f"Merhaba! 👋 {st.session_state.grade} planın hazır. Dersler: {dersler}. Herhangi bir konuda soru sorabilirsin!"
                st.session_state.messages.append({"role": "assistant", "content": welcome})
                st.session_state.chat_initialized = True
 
            for msg in st.session_state.messages:
                if msg["role"] == "assistant":
                    with st.chat_message("assistant", avatar="🤖"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("user", avatar="👤"):
                        st.write(msg["content"])
 
        # Chat input
        if user_input := st.chat_input("Bir şey sor..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
 
            sys_msg = f"Sen bir Türkçe AI eğitim koçusun. Öğrenci: {st.session_state.grade}, dersler: {', '.join(st.session_state.subjects)}, hedef: {st.session_state.goal}. Kısa, net, teşvik edici yanıtlar ver. Türkçe konuş."
            groq_msgs = [{"role": "system", "content": sys_msg}]
            for m in st.session_state.messages:
                groq_msgs.append({"role": m["role"], "content": m["content"]})
 
            with st.spinner("Düşünüyorum..."):
                try:
                    reply = call_groq(groq_msgs)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Yanıt alınamadı: {e}"})
            st.rerun()
 