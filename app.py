import { useState } from "react";

const APP_PY = `import streamlit as st
import anthropic
import base64
import json
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import io

st.set_page_config(page_title="Medical Report Analyzer", page_icon="🩺", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f172a; color: #e2e8f0; }
[data-testid="stHeader"] { background: transparent; }
.main-header { background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
  padding: 2rem; border-radius: 14px; margin-bottom: 2rem; border: 1px solid #1e293b; }
.result-card { background: #1e293b; border-radius: 14px; padding: 1.5rem;
  border: 1px solid #334155; margin-bottom: 1rem; }
.risk-high   { background: #ef444422; border: 1px solid #ef444466; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }
.risk-medium { background: #f59e0b22; border: 1px solid #f59e0b66; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }
.risk-low    { background: #22c55e22; border: 1px solid #22c55e66; border-radius: 10px; padding: 12px 16px; margin-bottom: 8px; }
.disclaimer  { background: #fbbf2411; border: 1px solid #fbbf2444;
  border-radius: 10px; padding: 12px 16px; color: #fbbf24; font-size: 13px; }
.stButton > button {
  background: linear-gradient(135deg, #0ea5e9, #2563eb) !important;
  color: white !important; border: none !important; border-radius: 10px !important;
  font-weight: 700 !important; width: 100% !important; }
.stTextArea textarea { background: #0f172a !important; color: #e2e8f0 !important;
  border: 1px solid #334155 !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

SYSTEM_PROMPT = """You are a highly experienced clinical AI assistant trained on medical literature.
Analyze the provided medical report and return ONLY a valid JSON object with exactly these keys:
{
  "summary": "A clear 3-5 sentence plain-language summary of the report findings.",
  "disease_risk": [
    {"condition": "Condition Name", "risk": "High|Moderate|Low", "reason": "Brief explanation"}
  ],
  "abnormalities": [
    {"parameter": "Parameter name", "value": "Reported value", "normal_range": "Normal range", "interpretation": "What it means"}
  ],
  "predicted_conditions": [
    {"condition": "Condition", "confidence": "High|Moderate|Low", "basis": "Why this is predicted"}
  ],
  "recommendations": ["Actionable recommendation 1", "Actionable recommendation 2"],
  "overall_health_score": 75,
  "urgency": "Routine|Soon|Urgent"
}
Return ONLY valid JSON. No markdown fences, no preamble."""

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\\n".join(page.get_text() for page in doc)

def extract_text_from_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    return pytesseract.image_to_string(img)

def encode_b64(data):
    return base64.standard_b64encode(data).decode("utf-8")

def analyze_report(text=None, file_bytes=None, file_type=None):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    if file_bytes and file_type:
        if file_type == "application/pdf":
            content = [
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": encode_b64(file_bytes)}},
                {"type": "text", "text": "Analyze this medical report and return the structured JSON."}
            ]
        else:
            content = [
                {"type": "image", "source": {"type": "base64", "media_type": file_type, "data": encode_b64(file_bytes)}},
                {"type": "text", "text": "Analyze this medical report image and return the structured JSON."}
            ]
    else:
        content = f"Analyze this medical report:\\n\\n{text}"
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}]
    )
    raw = msg.content[0].text.strip().replace("\`\`\`json","").replace("\`\`\`","")
    return json.loads(raw)

def risk_badge(risk):
    r = risk.lower()
    cls = "badge-high" if r == "high" else "badge-medium" if r == "moderate" else "badge-low"
    color = "#ef4444" if r=="high" else "#f59e0b" if r=="moderate" else "#22c55e"
    return f'<span style="background:{color}33;color:{color};border-radius:6px;padding:2px 12px;font-size:12px;font-weight:700">{risk}</span>'

def score_svg(score):
    color = "#22c55e" if score >= 70 else "#f59e0b" if score >= 40 else "#ef4444"
    r, cx, cy = 54, 64, 64
    circ = 2 * 3.14159 * r
    dash = (min(max(score,0),100)/100) * circ
    return f"""<svg width="128" height="128" viewBox="0 0 128 128">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#1e293b" stroke-width="10"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="10"
        stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-linecap="round"
        transform="rotate(-90 {cx} {cy})"/>
      <text x="{cx}" y="{cy+8}" text-anchor="middle" fill="{color}" font-size="24" font-weight="bold">{score}</text>
    </svg><p style="color:#94a3b8;font-size:12px;margin-top:-8px;text-align:center">Health Score</p>"""

st.markdown("""<div class="main-header">
  <h1 style="color:#38bdf8;margin:0;font-size:2rem">🩺 Medical Report Analyzer</h1>
  <p style="color:#64748b;margin-top:6px">AI-powered clinical analysis • Powered by Claude Sonnet 4</p>
</div>""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**📁 Upload Report** (PDF / Image / Text)")
    uploaded = st.file_uploader("", type=["pdf","png","jpg","jpeg","txt"], label_visibility="collapsed")
with col2:
    st.markdown("**✏️ Or Paste Report Text**")
    report_text = st.text_area("", height=150, placeholder="Paste lab report, blood work, radiology report...", label_visibility="collapsed")

if st.button("🧠 Analyze Medical Report"):
    with st.spinner("🔬 Analyzing report with AI..."):
        try:
            result = None
            if uploaded:
                fbytes = uploaded.read()
                if uploaded.type == "text/plain":
                    result = analyze_report(text=fbytes.decode("utf-8"))
                else:
                    result = analyze_report(file_bytes=fbytes, file_type=uploaded.type)
            elif report_text.strip():
                result = analyze_report(text=report_text)
            else:
                st.error("⚠️ Please upload a file or paste report text.")
            if result:
                st.session_state["result"] = result
        except Exception as e:
            st.error(f"Analysis failed: {e}")

if "result" in st.session_state:
    result = st.session_state["result"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(score_svg(result.get("overall_health_score", 0)), unsafe_allow_html=True)
    with c2:
        st.metric("Urgency", result.get("urgency","N/A"))
    with c3:
        st.metric("Conditions Detected", len(result.get("predicted_conditions",[])))

    tabs = st.tabs(["📋 Summary","⚠️ Disease Risk","🔍 Abnormalities","🧬 Predictions","💊 Recommendations"])
    with tabs[0]:
        st.markdown(f'<div class="result-card"><p style="line-height:1.8;font-size:15px">{result.get("summary","")}</p></div>', unsafe_allow_html=True)
    with tabs[1]:
        for d in result.get("disease_risk",[]):
            r = d["risk"].lower()
            cls = "risk-high" if r=="high" else "risk-medium" if r=="moderate" else "risk-low"
            st.markdown(f'<div class="{cls}"><strong>{d["condition"]}</strong> {risk_badge(d["risk"])}<br><small style="color:#94a3b8">{d["reason"]}</small></div>', unsafe_allow_html=True)
    with tabs[2]:
        if not result.get("abnormalities"):
            st.success("✅ No significant abnormalities detected.")
        for a in result.get("abnormalities",[]):
            st.markdown(f'<div class="result-card"><strong>{a["parameter"]}</strong> <span style="background:#ef444422;color:#ef4444;border-radius:6px;padding:2px 10px;font-size:12px">{a["value"]}</span><br><small style="color:#64748b">Normal: {a["normal_range"]}</small><br><span style="color:#94a3b8;font-size:13px">{a["interpretation"]}</span></div>', unsafe_allow_html=True)
    with tabs[3]:
        for c in result.get("predicted_conditions",[]):
            conf = c["confidence"].lower()
            cls = "risk-high" if conf=="high" else "risk-medium" if conf=="moderate" else "risk-low"
            st.markdown(f'<div class="{cls}"><strong>{c["condition"]}</strong> {risk_badge(c["confidence"])}<br><small style="color:#94a3b8">{c["basis"]}</small></div>', unsafe_allow_html=True)
    with tabs[4]:
        for rec in result.get("recommendations",[]):
            st.markdown(f'<div class="result-card">→ {rec}</div>', unsafe_allow_html=True)

    st.markdown('<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This AI analysis is for informational purposes only. Always consult a qualified healthcare professional.</div>', unsafe_allow_html=True)
`;

const REQUIREMENTS = `streamlit>=1.35.0
anthropic>=0.25.0
PyMuPDF>=1.24.0
Pillow>=10.0.0
pytesseract>=0.3.10
`;

const SECRETS = `ANTHROPIC_API_KEY = "sk-ant-your-key-here"
`;

const CONFIG = `[theme]
base = "dark"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#e2e8f0"
primaryColor = "#0ea5e9"

[server]
maxUploadSize = 10
`;

const README = `# 🩺 Medical Report Analyzer — AI Powered

## Features
- Upload PDF, Image (JPG/PNG), or Text medical reports
- Claude Sonnet 4 AI analysis via Anthropic API
- Disease risk assessment (High / Moderate / Low)
- Abnormality detection with normal ranges
- Predicted conditions with confidence scores
- Overall health score + urgency level
- Actionable clinical recommendations
- Dark medical-grade UI theme

## Quick Setup

### 1. Install Python dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 2. Install Tesseract OCR (for image reports)
- Windows: https://github.com/UB-Mannheim/tesseract/wiki
- macOS:   brew install tesseract
- Linux:   sudo apt install tesseract-ocr

### 3. Add your Anthropic API key
Edit .streamlit/secrets.toml:
\`\`\`
ANTHROPIC_API_KEY = "sk-ant-your-actual-key-here"
\`\`\`

### 4. Run the app
\`\`\`bash
streamlit run app.py
\`\`\`

## Deploy to Streamlit Cloud
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo
4. Add ANTHROPIC_API_KEY in the Secrets section

## Project Structure
\`\`\`
medical-report-analyzer/
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    ├── secrets.toml
    └── config.toml
\`\`\`

## Get Anthropic API Key
Visit: https://console.anthropic.com
`;

  const allFiles = [
    { name: "app.py",                      icon: "🐍", desc: "Main Streamlit application",    size: "~8 KB",   content: APP_PY,      color: "#3b82f6" },
    { name: "requirements.txt",             icon: "📦", desc: "Python dependencies",           size: "~120 B",  content: REQUIREMENTS, color: "#8b5cf6" },
    { name: "secrets.toml",                 icon: "🔑", desc: "API key config (edit this)",    size: "~60 B",   content: SECRETS,      color: "#f59e0b" },
    { name: "config.toml",                  icon: "⚙️", desc: "Streamlit theme config",        size: "~160 B",  content: CONFIG,       color: "#0ea5e9" },
    { name: "README.md",                    icon: "📖", desc: "Setup & deployment guide",      size: "~1.2 KB", content: README,       color: "#22c55e" },
  ];

  const [done, setDone] = useState({});

  const dl = (filename, content) => {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    setDone(p => ({ ...p, [filename]: true }));
  };

  const dlAll = () => {
    allFiles.forEach((f, i) => setTimeout(() => dl(f.name, f.content), i * 400));
  };

  const allDone = Object.keys(done).length === allFiles.length;

  return (
    <div style={{ background:"#0f172a", minHeight:"100vh", color:"#e2e8f0", fontFamily:"Inter,sans-serif" }}>
      {/* Header */}
      <div style={{ background:"linear-gradient(135deg,#1e3a5f,#0f172a)", padding:"24px 28px", borderBottom:"1px solid #1e293b" }}>
        <div style={{ display:"flex", alignItems:"center", gap:14 }}>
          <span style={{ fontSize:36 }}>🩺</span>
          <div>
            <div style={{ fontSize:22, fontWeight:800, color:"#38bdf8" }}>Medical Report Analyzer</div>
            <div style={{ fontSize:12, color:"#64748b", marginTop:2 }}>AI-Powered • Claude Sonnet 4 • Streamlit</div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth:680, margin:"0 auto", padding:"28px 20px" }}>

        {/* Download All Button */}
        <button onClick={dlAll} style={{
          width:"100%", background:"linear-gradient(135deg,#0ea5e9,#2563eb)",
          color:"#fff", border:"none", borderRadius:12, padding:"16px",
          fontSize:16, fontWeight:800, cursor:"pointer", marginBottom:24,
          boxShadow:"0 4px 24px #0ea5e944", letterSpacing:0.5
        }}>
          ⬇️ Download All {allFiles.length} Files at Once
        </button>

        {/* Tech stack badges */}
        <div style={{ background:"#1e293b", borderRadius:12, padding:"16px 20px", marginBottom:20, border:"1px solid #334155" }}>
          <div style={{ fontSize:13, color:"#64748b", fontWeight:600, marginBottom:10, textTransform:"uppercase", letterSpacing:1 }}>Tech Stack</div>
          <div style={{ display:"flex", flexWrap:"wrap", gap:8 }}>
            {["Python 3.10+","Streamlit","Anthropic Claude SDK","PyMuPDF (PDF)","Pytesseract (OCR)","Dark UI Theme","Deployable to Cloud"].map(t=>(
              <span key={t} style={{ background:"#0ea5e918", color:"#38bdf8", border:"1px solid #0ea5e933", borderRadius:6, padding:"3px 12px", fontSize:12, fontWeight:600 }}>{t}</span>
            ))}
          </div>
        </div>

        {/* File list */}
        <div style={{ fontSize:13, color:"#64748b", fontWeight:600, marginBottom:12, textTransform:"uppercase", letterSpacing:1 }}>📁 Individual Files</div>
        {allFiles.map(f => (
          <div key={f.name} style={{
            background:"#1e293b", borderRadius:12, padding:"16px 18px", marginBottom:10,
            border:`1px solid ${done[f.name] ? "#22c55e44" : "#334155"}`,
            display:"flex", alignItems:"center", gap:14, transition:"border 0.3s"
          }}>
            <span style={{ fontSize:26 }}>{f.icon}</span>
            <div style={{ flex:1 }}>
              <div style={{ fontWeight:700, fontSize:15, color: done[f.name] ? "#22c55e" : "#e2e8f0" }}>{f.name}</div>
              <div style={{ fontSize:12, color:"#64748b", marginTop:2 }}>{f.desc} • {f.size}</div>
            </div>
            {done[f.name]
              ? <span style={{ color:"#22c55e", fontSize:13, fontWeight:700 }}>✅ Saved</span>
              : <button onClick={() => dl(f.name, f.content)} style={{
                  background: f.color + "22", color: f.color,
                  border:`1px solid ${f.color}55`, borderRadius:8,
                  padding:"7px 18px", fontSize:13, fontWeight:700, cursor:"pointer", whiteSpace:"nowrap"
                }}>
                  ⬇️ Download
                </button>
            }
          </div>
        ))}

        {/* Setup steps */}
        <div style={{ background:"#1e293b", borderRadius:12, padding:"18px 20px", marginTop:8, border:"1px solid #334155" }}>
          <div style={{ fontSize:14, fontWeight:700, color:"#38bdf8", marginBottom:12 }}>🚀 Quick Setup After Download</div>
          {[
            ["1","Create a folder","mkdir medical-report-analyzer && cd medical-report-analyzer"],
            ["2","Save files","Put app.py & requirements.txt in root. Put secrets.toml & config.toml in a .streamlit/ folder"],
            ["3","Install deps","pip install -r requirements.txt"],
            ["4","Add API key","Edit secrets.toml → replace sk-ant-your-key-here with real key from console.anthropic.com"],
            ["5","Run app","streamlit run app.py"],
          ].map(([num, title, cmd]) => (
            <div key={num} style={{ display:"flex", gap:12, marginBottom:12 }}>
              <span style={{ background:"#0ea5e9", color:"#fff", borderRadius:"50%", width:22, height:22, display:"flex", alignItems:"center", justifyContent:"center", fontSize:11, fontWeight:800, flexShrink:0 }}>{num}</span>
              <div>
                <div style={{ fontSize:13, fontWeight:600, color:"#e2e8f0" }}>{title}</div>
                <div style={{ fontSize:11, color:"#64748b", fontFamily:"monospace", marginTop:2, background:"#0f172a", padding:"3px 8px", borderRadius:4 }}>{cmd}</div>
              </div>
            </div>
          ))}
        </div>

        <div style={{ marginTop:16, background:"#fbbf2411", border:"1px solid #fbbf2433", borderRadius:10, padding:"12px 16px", fontSize:12, color:"#fbbf24" }}>
          ⚠️ secrets.toml contains your API key — never upload it to GitHub. Add it to .gitignore.
        </div>
      </div>
    </div>
  );
}
