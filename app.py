import streamlit as st
from groq import Groq
from PIL import Image
import pdfplumber
import plotly.graph_objects as go
from fpdf import FPDF
import base64
import json
import re
import io

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%); color: #e2e8f0; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a 0%, #112240 100%); border-right: 1px solid #1e3a5f; }
.metric-card { background: linear-gradient(135deg, #112240 0%, #0d1b2a 100%); border: 1px solid #1e3a5f; border-radius: 16px; padding: 20px; margin: 10px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
.severity-normal { background: linear-gradient(135deg, #064e3b 0%, #065f46 100%); border: 1px solid #10b981; border-radius: 12px; padding: 12px 18px; margin: 6px 0; }
.severity-borderline { background: linear-gradient(135deg, #78350f 0%, #92400e 100%); border: 1px solid #f59e0b; border-radius: 12px; padding: 12px 18px; margin: 6px 0; }
.severity-critical { background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); border: 1px solid #ef4444; border-radius: 12px; padding: 12px 18px; margin: 6px 0; }
.section-header { background: linear-gradient(90deg, #1e3a5f, transparent); border-left: 4px solid #38bdf8; padding: 10px 16px; border-radius: 0 8px 8px 0; margin: 20px 0 10px 0; }
.chat-user { background: linear-gradient(135deg, #1e3a5f, #1e4080); border-radius: 16px 16px 4px 16px; padding: 12px 16px; margin: 8px 0; border: 1px solid #2563eb; }
.chat-ai { background: linear-gradient(135deg, #112240, #0d2137); border-radius: 16px 16px 16px 4px; padding: 12px 16px; margin: 8px 0; border: 1px solid #1e3a5f; }
.stButton > button { background: linear-gradient(135deg, #0ea5e9, #2563eb) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 10px 24px !important; font-weight: 600 !important; transition: all 0.3s ease !important; }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4) !important; }
.stTextInput > div > div > input, .stTextArea > div > div > textarea { background: #112240 !important; border: 1px solid #1e3a5f !important; color: #e2e8f0 !important; border-radius: 10px !important; }
.stSelectbox > div > div { background: #112240 !important; border: 1px solid #1e3a5f !important; color: #e2e8f0 !important; border-radius: 10px !important; }
.stTabs [data-baseweb="tab-list"] { background: #112240; border-radius: 12px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #94a3b8; border-radius: 8px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #0ea5e9, #2563eb) !important; color: white !important; }
hr { border-color: #1e3a5f !important; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── GROQ FUNCTIONS ──────────────────────────────────────────────────────────
def get_groq_client(api_key):
    return Groq(api_key=api_key)

def groq_chat(client, prompt, max_tokens=2000):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.3
    )
    return response.choices[0].message.content

# ─── EXTRACT TEXT FROM PDF ───────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ─── EXTRACT TEXT FROM IMAGE ─────────────────────────────────────────────────
def extract_text_from_image(client, image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": "Extract ALL text from this medical report image exactly as it appears. Include all values, units, reference ranges, and test names. Return only the extracted text."}
            ]
        }],
        max_tokens=2000
    )
    return response.choices[0].message.content

# ─── ANALYZE REPORT ──────────────────────────────────────────────────────────
def analyze_report(client, report_text):
    prompt = f"""You are an expert medical report analyzer. Analyze this medical report and respond ONLY in valid JSON format with no extra text, no markdown, no code blocks.

Medical Report:
{report_text}

Return ONLY this exact JSON structure:
{{
  "patient_summary": "Brief 2-3 sentence overview in simple language",
  "overall_health_score": 75,
  "parameters": [
    {{
      "name": "Hemoglobin",
      "value": "13.5",
      "unit": "g/dL",
      "normal_range": "12-17",
      "normal_min": 12,
      "normal_max": 17,
      "patient_value_num": 13.5,
      "status": "Normal",
      "simple_explanation": "Protein in red blood cells that carries oxygen"
    }}
  ],
  "critical_findings": [],
  "borderline_findings": [],
  "diet_suggestions": [
    {{"title": "Increase Iron Rich Foods", "description": "Eat spinach, lentils, red meat", "icon": "🥗"}}
  ],
  "lifestyle_suggestions": [
    {{"title": "Exercise Regularly", "description": "30 min walk daily", "icon": "🏃"}}
  ],
  "doctor_to_visit": "General Physician",
  "follow_up": "Repeat test after 3 months"
}}

RULES:
- status must be exactly: "Normal", "Borderline", or "Critical"
- overall_health_score must be a number 0-100
- Return ONLY valid JSON. No explanation. No markdown. No code blocks."""

    raw = groq_chat(client, prompt, max_tokens=3000)
    raw = raw.strip()
    raw = re.sub(r'```json|```', '', raw).strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group()
    return json.loads(raw)

# ─── TRANSLATE ───────────────────────────────────────────────────────────────
def translate_summary(client, text, language):
    if language == "English":
        return text
    prompt = f"Translate this medical summary to {language} in simple words. Return only the translated text:\n\n{text}"
    return groq_chat(client, prompt, max_tokens=500)

# ─── CHART ───────────────────────────────────────────────────────────────────
def create_range_chart(parameters):
    valid = []
    for p in parameters:
        try:
            float(p.get('normal_min', ''))
            float(p.get('normal_max', ''))
            float(p.get('patient_value_num', ''))
            valid.append(p)
        except (ValueError, TypeError):
            continue
    if not valid:
        return None

    names  = [p['name'] for p in valid]
    mins   = [float(p['normal_min']) for p in valid]
    maxs   = [float(p['normal_max']) for p in valid]
    vals   = [float(p['patient_value_num']) for p in valid]
    colors = ['#10b981' if p['status']=='Normal' else '#f59e0b' if p['status']=='Borderline' else '#ef4444' for p in valid]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Normal Range', x=names, y=[mx-mn for mn,mx in zip(mins,maxs)], base=mins,
                         marker_color='rgba(30,58,95,0.8)', marker_line_color='#38bdf8', marker_line_width=1))
    fig.add_trace(go.Scatter(name='Your Value', x=names, y=vals, mode='markers',
                             marker=dict(color=colors, size=14, symbol='diamond', line=dict(color='white', width=2))))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,34,64,0.8)',
                      font=dict(color='#e2e8f0', family='Plus Jakarta Sans'),
                      legend=dict(bgcolor='rgba(17,34,64,0.8)', bordercolor='#1e3a5f'),
                      xaxis=dict(gridcolor='#1e3a5f', tickangle=-30),
                      yaxis=dict(gridcolor='#1e3a5f'), height=380, margin=dict(t=20, b=20))
    return fig

# ─── PDF GENERATOR ───────────────────────────────────────────────────────────
def generate_pdf_report(analysis):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(10, 100, 180)
    pdf.cell(0, 12, "MediScan AI - Medical Report Analysis", ln=True, align='C')
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "AI-Powered Medical Report Analyzer", ln=True, align='C')
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(10, 100, 180)
    pdf.cell(0, 10, f"Overall Health Score: {analysis.get('overall_health_score','N/A')}/100", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "Patient Summary:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    summary = analysis.get('patient_summary', '').encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, summary)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "Test Parameters:", ln=True)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(10, 100, 180)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(60, 8, "Parameter", border=1, fill=True)
    pdf.cell(30, 8, "Value", border=1, fill=True)
    pdf.cell(40, 8, "Normal Range", border=1, fill=True)
    pdf.cell(30, 8, "Status", border=1, fill=True, ln=True)
    pdf.set_font("Helvetica", "", 10)
    for p in analysis.get('parameters', []):
        status = p.get('status', '')
        if status == 'Normal': pdf.set_text_color(6, 78, 59)
        elif status == 'Borderline': pdf.set_text_color(120, 53, 15)
        else: pdf.set_text_color(127, 29, 29)
        pdf.cell(60, 7, str(p.get('name','')), border=1)
        pdf.cell(30, 7, f"{p.get('value','')} {p.get('unit','')}", border=1)
        pdf.cell(40, 7, str(p.get('normal_range','')), border=1)
        pdf.cell(30, 7, status, border=1, ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, "Diet Suggestions:", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    for d in analysis.get('diet_suggestions', []):
        line = f"- {d.get('title','')}: {d.get('description','')}".encode('latin-1','replace').decode('latin-1')
        pdf.multi_cell(0, 7, line)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, f"Recommended Doctor: {analysis.get('doctor_to_visit','')}", ln=True)
    pdf.cell(0, 8, f"Follow Up: {analysis.get('follow_up','')}", ln=True)
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 7, "Disclaimer: AI-generated analysis. Always consult a qualified doctor.", ln=True, align='C')
    return bytes(pdf.output())

# ─── SESSION STATE ───────────────────────────────────────────────────────────
for key, default in [('analysis', None), ('report_text', ""), ('chat_history', []), ('all_reports', [])]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0;'>
        <div style='font-size: 48px;'>🩺</div>
        <h2 style='color: #38bdf8; margin: 8px 0; font-family: Syne, sans-serif;'>MediScan AI</h2>
        <p style='color: #64748b; font-size: 13px;'>AI Medical Report Analyzer</p>
        <p style='color: #10b981; font-size: 11px; background:#064e3b; padding:4px 8px; border-radius:6px; display:inline-block;'>⚡ Powered by Groq (Free)</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="Enter your Groq API key...")
    st.markdown("[🆓 Get FREE Groq API key →](https://console.groq.com/keys)")
    st.divider()
    language = st.selectbox("🌐 Language", ["English", "Hindi", "Tamil", "Telugu", "Malayalam", "Bengali"])
    st.divider()
    st.markdown("""
    <div style='padding: 12px; background: #112240; border-radius: 10px; border: 1px solid #1e3a5f;'>
        <p style='color: #64748b; font-size: 12px; margin: 0;'>⚠️ <b>Disclaimer</b><br>For educational purposes only. Always consult a qualified doctor.</p>
    </div>
    """, unsafe_allow_html=True)

# ─── MAIN HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 30px 0 20px;'>
    <h1 style='font-size: 42px; color: #f1f5f9; margin: 0;'>🩺 MediScan <span style='color: #38bdf8;'>AI</span></h1>
    <p style='color: #64748b; font-size: 16px; margin-top: 8px;'>Upload your medical report · Get instant AI analysis · Understand your health</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Analyze", "📊 Results & Charts", "💬 Ask AI Doctor", "📋 Report History"])

# ════════════════════════════════════ TAB 1 ═══════════════════════════════════
with tab1:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("<div class='section-header'><b>📁 Upload Medical Report</b></div>", unsafe_allow_html=True)
        uploaded_files = st.file_uploader("Upload PDF or Image (JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
        simple_language = st.toggle("🔤 Show explanation in simple language", value=True)

        if st.button("🔍 Analyze Report", use_container_width=True):
            if not api_key:
                st.error("❌ Please enter your Groq API Key in the sidebar!")
            elif not uploaded_files:
                st.error("❌ Please upload at least one medical report!")
            else:
                with st.spinner("🔄 AI is analyzing your report..."):
                    try:
                        client   = get_groq_client(api_key)
                        all_text = ""
                        for f in uploaded_files:
                            if f.type == "application/pdf":
                                all_text += extract_text_from_pdf(f) + "\n"
                            else:
                                img = Image.open(f)
                                all_text += extract_text_from_image(client, img) + "\n"

                        if not all_text.strip():
                            st.error("❌ Could not extract text. Try a clearer image or PDF.")
                        else:
                            analysis = analyze_report(client, all_text)
                            if language != "English":
                                analysis['patient_summary'] = translate_summary(client, analysis['patient_summary'], language)
                            st.session_state.analysis    = analysis
                            st.session_state.report_text = all_text
                            st.session_state.chat_history = []
                            st.session_state.all_reports.append({
                                "name": uploaded_files[0].name,
                                "score": analysis.get('overall_health_score', 0),
                                "analysis": analysis
                            })
                            st.success("✅ Analysis complete! Go to **Results & Charts** tab.")
                    except json.JSONDecodeError:
                        st.error("❌ AI response parsing failed. Please try again.")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    with col2:
        st.markdown("<div class='section-header'><b>ℹ️ How It Works</b></div>", unsafe_allow_html=True)
        for icon, step in [("1️⃣","Get FREE Groq API key from console.groq.com"),("2️⃣","Enter API key in sidebar"),("3️⃣","Upload medical report (PDF or image)"),("4️⃣","Click Analyze Report"),("5️⃣","View results, charts & suggestions"),("6️⃣","Ask questions & Download PDF")]:
            st.markdown(f"<div style='display:flex;align-items:center;gap:12px;padding:10px;background:#112240;border-radius:10px;margin:6px 0;border:1px solid #1e3a5f;'><span style='font-size:20px;'>{icon}</span><span style='color:#cbd5e1;font-size:14px;'>{step}</span></div>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:16px;padding:12px;background:#064e3b;border-radius:10px;border:1px solid #10b981;'><p style='color:#6ee7b7;font-size:13px;margin:0;'>⚡ <b>Groq is FREE!</b><br>No credit card · No quota issues · Super fast</p></div>", unsafe_allow_html=True)

# ════════════════════════════════════ TAB 2 ═══════════════════════════════════
with tab2:
    if st.session_state.analysis is None:
        st.markdown("<div style='text-align:center;padding:60px;color:#475569;'><div style='font-size:64px;'>📊</div><h3 style='color:#475569;'>No analysis yet</h3><p>Upload and analyze a report in the first tab</p></div>", unsafe_allow_html=True)
    else:
        analysis = st.session_state.analysis
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            score = analysis.get('overall_health_score', 0)
            color = '#10b981' if score >= 70 else '#f59e0b' if score >= 50 else '#ef4444'
            st.markdown(f"<div class='metric-card' style='text-align:center;border-color:{color};'><div style='font-size:48px;font-weight:800;color:{color};'>{score}</div><div style='color:#94a3b8;font-size:13px;'>Health Score / 100</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='section-header'><b>📝 Summary</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><p style='color:#cbd5e1;line-height:1.7;margin:0;'>{analysis.get('patient_summary','')}</p></div>", unsafe_allow_html=True)
        with col3:
            critical   = len(analysis.get('critical_findings', []))
            borderline = len(analysis.get('borderline_findings', []))
            st.markdown(f"<div class='metric-card' style='text-align:center;'><div style='color:#ef4444;font-size:28px;font-weight:700;'>{critical}</div><div style='color:#94a3b8;font-size:12px;'>Critical</div><hr style='border-color:#1e3a5f;margin:8px 0;'><div style='color:#f59e0b;font-size:28px;font-weight:700;'>{borderline}</div><div style='color:#94a3b8;font-size:12px;'>Borderline</div></div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("<div class='section-header'><b>🔬 Test Parameters & Severity</b></div>", unsafe_allow_html=True)
        params = analysis.get('parameters', [])
        if params:
            cols = st.columns(2)
            for i, p in enumerate(params):
                status    = p.get('status', 'Normal')
                css_class = 'severity-normal' if status=='Normal' else 'severity-borderline' if status=='Borderline' else 'severity-critical'
                icon      = '🟢' if status=='Normal' else '🟡' if status=='Borderline' else '🔴'
                with cols[i % 2]:
                    st.markdown(f"<div class='{css_class}'><div style='display:flex;justify-content:space-between;align-items:center;'><b style='color:#f1f5f9;'>{icon} {p.get('name','')}</b><span style='font-size:18px;font-weight:700;color:#f1f5f9;'>{p.get('value','')} <small style='font-size:12px;'>{p.get('unit','')}</small></span></div><div style='color:#94a3b8;font-size:12px;margin-top:4px;'>Normal: {p.get('normal_range','N/A')} | {p.get('simple_explanation','')}</div></div>", unsafe_allow_html=True)

        st.divider()
        st.markdown("<div class='section-header'><b>📈 Normal Range Comparison Chart</b></div>", unsafe_allow_html=True)
        chart = create_range_chart(params)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.info("Chart not available — numeric values not found in this report.")

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-header'><b>🥗 Diet Suggestions</b></div>", unsafe_allow_html=True)
            for d in analysis.get('diet_suggestions', []):
                st.markdown(f"<div class='metric-card'><b style='color:#38bdf8;'>{d.get('icon','')} {d.get('title','')}</b><p style='color:#94a3b8;margin:4px 0 0;font-size:13px;'>{d.get('description','')}</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='section-header'><b>🏃 Lifestyle Suggestions</b></div>", unsafe_allow_html=True)
            for ls in analysis.get('lifestyle_suggestions', []):
                st.markdown(f"<div class='metric-card'><b style='color:#38bdf8;'>{ls.get('icon','')} {ls.get('title','')}</b><p style='color:#94a3b8;margin:4px 0 0;font-size:13px;'>{ls.get('description','')}</p></div>", unsafe_allow_html=True)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<div class='metric-card'><b style='color:#38bdf8;'>👨‍⚕️ Recommended Specialist</b><p style='color:#f1f5f9;margin:8px 0 0;font-size:16px;'>{analysis.get('doctor_to_visit','General Physician')}</p><p style='color:#64748b;font-size:13px;'>Follow up: {analysis.get('follow_up','As advised by doctor')}</p></div>", unsafe_allow_html=True)
        with col2:
            if st.button("📥 Download PDF Report", use_container_width=True):
                try:
                    pdf_bytes = generate_pdf_report(analysis)
                    b64  = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="MediScan_Report.pdf" style="display:block;text-align:center;background:linear-gradient(135deg,#0ea5e9,#2563eb);color:white;padding:12px;border-radius:10px;text-decoration:none;font-weight:600;">📄 Click to Download PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"PDF error: {e}")

# ════════════════════════════════════ TAB 3 ═══════════════════════════════════
with tab3:
    if st.session_state.analysis is None:
        st.markdown("<div style='text-align:center;padding:60px;color:#475569;'><div style='font-size:64px;'>💬</div><h3 style='color:#475569;'>Analyze a report first</h3><p>Upload your report in Tab 1 to enable AI Doctor chat</p></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='section-header'><b>💬 Ask Questions About Your Report</b></div>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b;font-size:13px;'>Quick questions:</p>", unsafe_allow_html=True)
        qcols = st.columns(3)
        quick_qs = ["What are my critical values?","What foods should I eat?","Which doctor should I visit?","What does my report mean overall?","What tests should I repeat?","Is my health score good?"]
        for i, q in enumerate(quick_qs):
            with qcols[i % 3]:
                if st.button(q, key=f"quick_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": q})
                    try:
                        client  = get_groq_client(api_key)
                        prompt  = f"Medical Report: {json.dumps(st.session_state.analysis)}\n\nQuestion: {q}\n\nAnswer in simple friendly language in 3-4 sentences."
                        answer  = groq_chat(client, prompt, max_tokens=300)
                        st.session_state.chat_history.append({"role": "ai", "content": answer})
                        st.rerun()
                    except Exception as e:
                        st.session_state.chat_history.append({"role": "ai", "content": f"Error: {str(e)}"})
                        st.rerun()

        st.divider()
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"<div class='chat-user'><span style='color:#94a3b8;font-size:11px;'>You</span><br><span style='color:#f1f5f9;'>{msg['content']}</span></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-ai'><span style='color:#38bdf8;font-size:11px;'>🩺 AI Doctor</span><br><span style='color:#cbd5e1;'>{msg['content']}</span></div>", unsafe_allow_html=True)

        user_q = st.text_input("Ask anything about your report...", placeholder="e.g. What does my creatinine level mean?", key="chat_input")
        if st.button("Send 📨", use_container_width=True) and user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            try:
                client = get_groq_client(api_key)
                prompt = f"Medical Report: {json.dumps(st.session_state.analysis)}\n\nQuestion: {user_q}\n\nAnswer in simple friendly language."
                answer = groq_chat(client, prompt, max_tokens=400)
                st.session_state.chat_history.append({"role": "ai", "content": answer})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ════════════════════════════════════ TAB 4 ═══════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'><b>📋 Report History & Comparison</b></div>", unsafe_allow_html=True)
    if not st.session_state.all_reports:
        st.markdown("<div style='text-align:center;padding:60px;color:#475569;'><div style='font-size:64px;'>📋</div><h3 style='color:#475569;'>No reports yet</h3><p>Analyzed reports will appear here</p></div>", unsafe_allow_html=True)
    else:
        for i, r in enumerate(st.session_state.all_reports):
            score = r.get('score', 0)
            color = '#10b981' if score >= 70 else '#f59e0b' if score >= 50 else '#ef4444'
            st.markdown(f"<div class='metric-card' style='border-color:{color};'><div style='display:flex;justify-content:space-between;align-items:center;'><div><b style='color:#f1f5f9;font-size:16px;'>📄 Report {i+1}: {r.get('name','')}</b><p style='color:#64748b;font-size:13px;margin:4px 0 0;'>Critical: {len(r['analysis'].get('critical_findings',[]))} | Borderline: {len(r['analysis'].get('borderline_findings',[]))}</p></div><div style='text-align:center;'><div style='font-size:32px;font-weight:800;color:{color};'>{score}</div><div style='color:#64748b;font-size:11px;'>Health Score</div></div></div></div>", unsafe_allow_html=True)

        if len(st.session_state.all_reports) > 1:
            st.markdown("<div class='section-header'><b>📈 Health Score Trend</b></div>", unsafe_allow_html=True)
            scores = [r['score'] for r in st.session_state.all_reports]
            names  = [f"Report {i+1}" for i in range(len(scores))]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=names, y=scores, mode='lines+markers', line=dict(color='#38bdf8', width=3), marker=dict(size=12, color='#0ea5e9', line=dict(color='white', width=2))))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,34,64,0.8)', font=dict(color='#e2e8f0'), yaxis=dict(range=[0,100], gridcolor='#1e3a5f'), xaxis=dict(gridcolor='#1e3a5f'), height=300)
            st.plotly_chart(fig, use_container_width=True)
