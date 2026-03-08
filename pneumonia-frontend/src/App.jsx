import { useState, useRef, useCallback } from "react";

const API_URL = "http://localhost:8000";

// ─── Colour tokens ────────────────────────────────────────────────────────────
const C = {
  bg:        "#0a0d14",
  surface:   "#111520",
  card:      "#161b2c",
  border:    "#1e2640",
  accent:    "#3b82f6",
  accentGlow:"#3b82f640",
  danger:    "#ef4444",
  safe:      "#22c55e",
  muted:     "#4b5680",
  text:      "#e2e8f0",
  textDim:   "#7c87a6",
};

// ─── Pulse ring for the result badge ─────────────────────────────────────────
const pulseKF = `@keyframes pulse-ring{0%{transform:scale(.85);opacity:.6}70%{transform:scale(1.15);opacity:0}100%{transform:scale(1.15);opacity:0}}`;
const fadeIn   = `@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}`;
const spin     = `@keyframes spin{to{transform:rotate(360deg)}}`;
const scanLine = `@keyframes scanLine{0%{top:0}100%{top:100%}}`;

const globalCSS = `
  ${pulseKF}${fadeIn}${spin}${scanLine}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:${C.bg};color:${C.text};font-family:'DM Sans',sans-serif;min-height:100vh}
  ::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:${C.surface}}
  ::-webkit-scrollbar-thumb{background:${C.border};border-radius:3px}
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');
`;

// ─── Sub-components ───────────────────────────────────────────────────────────
function Spinner() {
  return (
    <div style={{width:40,height:40,border:`3px solid ${C.border}`,borderTopColor:C.accent,
      borderRadius:"50%",animation:"spin 0.9s linear infinite"}} />
  );
}

function ConfidenceMeter({ value, label, color }) {
  return (
    <div style={{display:"flex",flexDirection:"column",gap:6}}>
      <div style={{display:"flex",justifyContent:"space-between",fontSize:12,color:C.textDim}}>
        <span style={{fontFamily:"Space Mono"}}>{label}</span>
        <span style={{color,fontFamily:"Space Mono",fontWeight:700}}>{(value*100).toFixed(1)}%</span>
      </div>
      <div style={{height:6,background:C.border,borderRadius:3,overflow:"hidden"}}>
        <div style={{
          height:"100%",width:`${value*100}%`,background:color,
          borderRadius:3,transition:"width 1s cubic-bezier(0.34,1.56,0.64,1)",
          boxShadow:`0 0 8px ${color}80`
        }}/>
      </div>
    </div>
  );
}

function ImageCard({ title, src, tag }) {
  const tagColors = {
    "Original":"#6366f1", "Heatmap":"#f59e0b", "Overlay":"#ec4899"
  };
  return (
    <div style={{
      background:C.card,border:`1px solid ${C.border}`,borderRadius:12,overflow:"hidden",
      display:"flex",flexDirection:"column",
      boxShadow:`0 4px 20px #00000060`,
      animation:"fadeIn 0.5s ease forwards",
    }}>
      <div style={{
        padding:"10px 14px",background:`${C.surface}`,borderBottom:`1px solid ${C.border}`,
        display:"flex",alignItems:"center",justifyContent:"space-between"
      }}>
        <span style={{fontFamily:"Space Mono",fontSize:11,color:C.textDim,letterSpacing:1}}>{title}</span>
        <span style={{
          fontSize:9,fontFamily:"Space Mono",background:`${tagColors[tag]}22`,
          color:tagColors[tag],padding:"2px 8px",borderRadius:99,border:`1px solid ${tagColors[tag]}44`,
          letterSpacing:1,textTransform:"uppercase"
        }}>{tag}</span>
      </div>
      <img src={`data:image/png;base64,${src}`} alt={title}
        style={{width:"100%",aspectRatio:"1/1",objectFit:"cover",display:"block"}}/>
    </div>
  );
}

function VisualisationSection({ title, icon, heatmap, overlay, original }) {
  const [tab, setTab] = useState("overlay");
  const tabs = [
    { id:"overlay", label:"Overlay" },
    { id:"heatmap", label:"Heatmap" },
    { id:"original", label:"Original" },
  ];
  const imgs = { overlay, heatmap, original };
  const tags = { overlay:"Overlay", heatmap:"Heatmap", original:"Original" };
  return (
    <div style={{
      background:C.card,border:`1px solid ${C.border}`,borderRadius:16,overflow:"hidden",
      animation:"fadeIn 0.5s ease forwards",
    }}>
      {/* Header */}
      <div style={{
        padding:"14px 18px",background:C.surface,borderBottom:`1px solid ${C.border}`,
        display:"flex",alignItems:"center",gap:10
      }}>
        <span style={{fontSize:18}}>{icon}</span>
        <span style={{fontFamily:"Space Mono",fontSize:13,fontWeight:700,letterSpacing:1}}>{title}</span>
      </div>
      {/* Tabs */}
      <div style={{display:"flex",gap:6,padding:"12px 16px 0",background:C.card}}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setTab(t.id)} style={{
            padding:"6px 16px",borderRadius:8,border:"none",cursor:"pointer",
            fontSize:11,fontFamily:"Space Mono",letterSpacing:1,
            background: tab===t.id ? C.accent : C.border,
            color: tab===t.id ? "#fff" : C.textDim,
            transition:"all 0.2s",boxShadow: tab===t.id ? `0 0 10px ${C.accentGlow}` : "none"
          }}>{t.label}</button>
        ))}
      </div>
      {/* Image */}
      <div style={{padding:16}}>
        <div style={{position:"relative",borderRadius:10,overflow:"hidden"}}>
          <img src={`data:image/png;base64,${imgs[tab]}`} alt={tab}
            style={{width:"100%",aspectRatio:"1/1",objectFit:"cover",display:"block",borderRadius:10}}/>
          <span style={{
            position:"absolute",bottom:8,right:8,
            fontSize:9,fontFamily:"Space Mono",letterSpacing:1,textTransform:"uppercase",
            background:"#000000aa",color:C.textDim,padding:"4px 10px",borderRadius:6,
          }}>{tags[tab]}</span>
        </div>
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [dragging, setDragging] = useState(false);
  const [preview,  setPreview]  = useState(null);
  const [file,     setFile]     = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [result,   setResult]   = useState(null);
  const [error,    setError]    = useState(null);
  const inputRef = useRef();

  const handleFile = useCallback((f) => {
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = e => setPreview(e.target.result);
    reader.readAsDataURL(f);
  }, []);

  const onDrop = useCallback(e => {
    e.preventDefault(); setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const analyse = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API_URL}/predict`, { method:"POST", body:fd });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      setResult(await res.json());
    } catch(e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const isPneumonia = result?.prediction === "PNEUMONIA";
  const resultColor = isPneumonia ? C.danger : C.safe;

  return (
    <>
      <style>{globalCSS}</style>

      {/* ── Scanline background effect ── */}
      <div style={{
        position:"fixed",inset:0,pointerEvents:"none",zIndex:0,
        backgroundImage:`repeating-linear-gradient(0deg,transparent,transparent 2px,${C.bg}40 2px,${C.bg}40 4px)`,
        opacity:0.3
      }}/>

      <div style={{position:"relative",zIndex:1,minHeight:"100vh",
        display:"flex",flexDirection:"column",alignItems:"center",
        padding:"40px 20px 80px",gap:32,maxWidth:1100,margin:"0 auto"}}>

        {/* ── Header ── */}
        <header style={{textAlign:"center",animation:"fadeIn 0.6s ease"}}>
          <div style={{
            display:"inline-flex",alignItems:"center",gap:8,
            background:`${C.accent}15`,border:`1px solid ${C.accent}30`,
            borderRadius:99,padding:"4px 16px",marginBottom:16,
            fontSize:11,fontFamily:"Space Mono",color:C.accent,letterSpacing:2
          }}>⬡ DENSENET-121 · CHEST X-RAY</div>
          <h1 style={{
            fontSize:"clamp(28px,5vw,52px)",fontWeight:600,letterSpacing:-1,
            background:`linear-gradient(135deg,${C.text} 30%,${C.accent})`,
            WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"
          }}>Pneumonia<br/>Detection</h1>
          <p style={{marginTop:10,color:C.textDim,fontSize:14,maxWidth:420,margin:"10px auto 0",lineHeight:1.6}}>
            Upload a chest X-ray to receive an AI-assisted diagnosis with<br/>
            explainability via <strong style={{color:C.text}}>GradCAM</strong> &amp; <strong style={{color:C.text}}>ScoreCAM</strong> visualisations.
          </p>
        </header>

        {/* ── Upload card ── */}
        <div style={{width:"100%",maxWidth:520}}>
          <div
            onDragOver={e => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current.click()}
            style={{
              position:"relative",cursor:"pointer",
              border:`2px dashed ${dragging ? C.accent : C.border}`,
              borderRadius:18,padding:"32px 20px",textAlign:"center",
              background: dragging ? `${C.accent}08` : C.card,
              transition:"all 0.2s",
              boxShadow: dragging ? `0 0 30px ${C.accentGlow}` : "none",
            }}
          >
            {preview ? (
              <img src={preview} alt="preview" style={{
                maxHeight:260,maxWidth:"100%",borderRadius:10,objectFit:"contain"
              }}/>
            ) : (
              <>
                <div style={{fontSize:48,marginBottom:12}}>🫁</div>
                <p style={{color:C.text,fontWeight:500,marginBottom:4}}>Drop your X-ray here</p>
                <p style={{color:C.textDim,fontSize:13}}>or click to browse &nbsp;·&nbsp; JPEG / PNG</p>
              </>
            )}
            <input ref={inputRef} type="file" accept="image/*" style={{display:"none"}}
              onChange={e => handleFile(e.target.files[0])}/>
          </div>

          {/* Analyse button */}
          {file && !loading && (
            <button onClick={analyse} style={{
              marginTop:14,width:"100%",padding:"14px 0",borderRadius:12,
              border:"none",cursor:"pointer",
              background:`linear-gradient(135deg,${C.accent},#2563eb)`,
              color:"#fff",fontSize:14,fontWeight:600,letterSpacing:1,
              fontFamily:"Space Mono",
              boxShadow:`0 6px 24px ${C.accentGlow}`,
              transition:"transform 0.1s",
            }}
            onMouseDown={e=>e.currentTarget.style.transform="scale(0.98)"}
            onMouseUp={e=>e.currentTarget.style.transform="scale(1)"}
            >▶ &nbsp;ANALYSE X-RAY</button>
          )}
          {loading && (
            <div style={{marginTop:14,display:"flex",justifyContent:"center",
              alignItems:"center",gap:12,padding:20}}>
              <Spinner/>
              <span style={{color:C.textDim,fontFamily:"Space Mono",fontSize:12}}>
                PROCESSING…
              </span>
            </div>
          )}
          {error && (
            <div style={{marginTop:12,padding:"12px 16px",borderRadius:10,
              background:`${C.danger}18`,border:`1px solid ${C.danger}40`,
              color:C.danger,fontSize:13,fontFamily:"Space Mono"}}>
              ⚠ {error}
            </div>
          )}
        </div>

        {/* ── Results ── */}
        {result && (
          <div style={{width:"100%",display:"flex",flexDirection:"column",gap:28,animation:"fadeIn 0.5s ease"}}>

            {/* Diagnosis banner */}
            <div style={{
              background:C.card,border:`1px solid ${resultColor}40`,
              borderRadius:18,padding:"28px 32px",
              boxShadow:`0 0 40px ${resultColor}18`,
              display:"flex",flexWrap:"wrap",gap:24,alignItems:"center"
            }}>
              {/* Badge */}
              <div style={{position:"relative",width:80,height:80,flexShrink:0}}>
                <div style={{
                  position:"absolute",inset:-6,borderRadius:"50%",
                  border:`2px solid ${resultColor}`,
                  animation:"pulse-ring 1.5s cubic-bezier(0.215,0.61,0.355,1) infinite",
                }}/>
                <div style={{
                  width:80,height:80,borderRadius:"50%",
                  background:`${resultColor}20`,border:`2px solid ${resultColor}`,
                  display:"flex",alignItems:"center",justifyContent:"center",
                  fontSize:32
                }}>
                  {isPneumonia ? "🔴" : "🟢"}
                </div>
              </div>

              {/* Text */}
              <div style={{flex:1,minWidth:160}}>
                <div style={{
                  fontFamily:"Space Mono",fontSize:11,letterSpacing:3,
                  color:C.textDim,textTransform:"uppercase",marginBottom:4
                }}>Diagnosis</div>
                <div style={{
                  fontSize:"clamp(22px,4vw,34px)",fontWeight:700,
                  color:resultColor,letterSpacing:-0.5,lineHeight:1.1
                }}>{result.prediction}</div>
                <div style={{marginTop:4,fontSize:13,color:C.textDim}}>
                  Raw probability: <span style={{color:C.text,fontFamily:"Space Mono"}}>
                    {(result.probability*100).toFixed(2)}%
                  </span>
                </div>
              </div>

              {/* Confidence bars */}
              <div style={{flex:1,minWidth:200,display:"flex",flexDirection:"column",gap:10}}>
                <ConfidenceMeter
                  label="PNEUMONIA"
                  value={result.probability}
                  color={C.danger}/>
                <ConfidenceMeter
                  label="NORMAL"
                  value={1 - result.probability}
                  color={C.safe}/>
              </div>
            </div>

            {/* Clinical note */}
            <div style={{
              padding:"12px 18px",borderRadius:10,
              background:`#f59e0b10`,border:`1px solid #f59e0b30`,
              fontSize:12,color:"#f59e0b",fontFamily:"Space Mono",lineHeight:1.6
            }}>
              ⚠ &nbsp;This tool is for screening assistance only and should not replace a qualified radiologist's assessment.
            </div>

            {/* Visualisations */}
            <h2 style={{fontFamily:"Space Mono",fontSize:13,letterSpacing:2,color:C.textDim,
              textTransform:"uppercase",paddingLeft:2}}>Explainability Visualisations</h2>

            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(280px,1fr))",gap:20}}>
              <VisualisationSection
                title="GRAD-CAM"
                icon="🔥"
                heatmap={result.images.gradcam_heatmap}
                overlay={result.images.gradcam_overlay}
                original={result.images.original}
              />
              <VisualisationSection
                title="SCORE-CAM"
                icon="🎯"
                heatmap={result.images.scorecam_heatmap}
                overlay={result.images.scorecam_overlay}
                original={result.images.original}
              />
            </div>

            {/* Technique explanations */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(240px,1fr))",gap:16}}>
              {[
                { icon:"🔥", name:"Grad-CAM", color:"#f97316",
                  desc:"Uses gradient flow through the last convolutional layer to highlight regions most responsible for the prediction. Fast and reliable." },
                { icon:"🎯", name:"Score-CAM", color:"#a855f7",
                  desc:"Masks each feature map individually and measures impact on score. More accurate than Grad-CAM but computationally heavier." },
              ].map(t => (
                <div key={t.name} style={{
                  background:C.card,border:`1px solid ${C.border}`,borderRadius:14,
                  padding:"18px 20px",display:"flex",gap:14
                }}>
                  <span style={{fontSize:24,flexShrink:0}}>{t.icon}</span>
                  <div>
                    <div style={{fontWeight:600,fontSize:14,color:t.color,marginBottom:4,
                      fontFamily:"Space Mono"}}>{t.name}</div>
                    <p style={{color:C.textDim,fontSize:12,lineHeight:1.6}}>{t.desc}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* Reset */}
            <div style={{textAlign:"center"}}>
              <button onClick={() => { setResult(null); setFile(null); setPreview(null); }}
                style={{
                  background:"transparent",border:`1px solid ${C.border}`,
                  color:C.textDim,padding:"10px 28px",borderRadius:10,
                  cursor:"pointer",fontFamily:"Space Mono",fontSize:11,letterSpacing:1,
                  transition:"all 0.2s"
                }}
                onMouseEnter={e => { e.currentTarget.style.borderColor=C.accent; e.currentTarget.style.color=C.accent; }}
                onMouseLeave={e => { e.currentTarget.style.borderColor=C.border; e.currentTarget.style.color=C.textDim; }}
              >↩ ANALYSE ANOTHER</button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
