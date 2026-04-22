import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import "@/App.css";
import { PatternGlyph, GridViz, MetricPill, VerdictChip } from "@/components/WatermarkUI";
import { HowItWorksPanel } from "@/components/HowItWorksPanel";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const BUILTIN_PATTERNS = ["square", "checkerboard", "circle", "diamond", "cross"];
const MODELS = [
  "HuggingFaceTB/SmolLM2-135M-Instruct",
];
const MODEL_LABELS = {
  "HuggingFaceTB/SmolLM2-135M-Instruct": "SmolLM2 · 135M · instruct",
};

function AdvancedPopover({ open, onClose, state, setState }) {
  if (!open) return null;
  const Slider = ({ id, label, min, max, step, value, onChange, fmt }) => (
    <div className="flex flex-col gap-1.5" data-testid={`adv-${id}`}>
      <div className="flex items-baseline justify-between">
        <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#888884]">{label}</span>
        <span className="text-xs font-mono font-semibold text-[#111]">{fmt ? fmt(value) : value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-[#111]"
      />
    </div>
  );
  return (
    <>
      <div className="fixed inset-0 z-30" onClick={onClose} aria-hidden="true" />
      <div
        data-testid="advanced-popover"
        className="absolute bottom-full mb-3 left-0 w-[340px] bg-white border border-[#E5E5DF] rounded-xl shadow-[0_16px_48px_rgba(0,0,0,0.08)] p-5 z-40"
      >
        <div className="flex items-baseline justify-between mb-4">
          <span className="font-heading font-bold text-sm">Advanced</span>
          <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-[#888884]">watermark + sampling</span>
        </div>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <span className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#888884]">secret key</span>
            <input
              data-testid="adv-secret-key"
              type="text"
              value={state.secret_key}
              onChange={(e) => setState({ ...state, secret_key: e.target.value })}
              className="h-9 bg-[#F5F5F0] border-none rounded-md font-mono text-sm px-3 w-full focus:outline-none focus:ring-2 focus:ring-[#111]/20"
              placeholder="llmwatermark"
            />
          </div>
          <Slider id="gamma" label="gamma (green fraction)" min={0.1} max={0.9} step={0.05} value={state.gamma} onChange={(v) => setState({ ...state, gamma: v })} fmt={(v) => v.toFixed(2)} />
          <Slider id="delta" label="delta (logit bias)" min={0.5} max={10.0} step={0.5} value={state.delta} onChange={(v) => setState({ ...state, delta: v })} fmt={(v) => v.toFixed(1)} />
          <Slider id="tau" label="tau (z-score threshold)" min={1.0} max={10.0} step={0.5} value={state.tau} onChange={(v) => setState({ ...state, tau: v })} fmt={(v) => v.toFixed(1)} />
          <div className="grid grid-cols-2 gap-3">
            <Slider id="temperature" label="temperature" min={0.1} max={1.5} step={0.1} value={state.temperature} onChange={(v) => setState({ ...state, temperature: v })} fmt={(v) => v.toFixed(1)} />
            <Slider id="top_p" label="top-p" min={0.5} max={1.0} step={0.05} value={state.top_p} onChange={(v) => setState({ ...state, top_p: v })} fmt={(v) => v.toFixed(2)} />
          </div>
        </div>
      </div>
    </>
  );
}

function ChatInput({
  prompt, setPrompt,
  pattern, setPattern,
  uploadB64, setUploadB64,
  model, setModel,
  maxTokens, setMaxTokens,
  advanced, setAdvanced,
  onSubmit, busy,
}) {
  const fileRef = useRef(null);
  const [advOpen, setAdvOpen] = useState(false);

  const handleFile = (f) => {
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      setUploadB64(reader.result);
      setPattern("upload");
    };
    reader.readAsDataURL(f);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!busy && prompt.trim()) onSubmit();
    }
  };

  return (
    <div className="pointer-events-auto max-w-3xl mx-auto w-full">
      <div
        className="bg-white border border-[#E5E5DF] rounded-2xl shadow-[0_8px_32px_rgba(17,17,17,0.06)] p-3 sm:p-4 flex flex-col gap-3"
        data-testid="chat-input-card"
      >
        {/* Top row — pattern chips + model + max tokens */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="flex items-center gap-1.5 flex-wrap" data-testid="pattern-picker">
            {BUILTIN_PATTERNS.map((p) => {
              const selected = pattern === p;
              return (
                <button
                  key={p}
                  data-testid={`pattern-chip-${p}`}
                  onClick={() => { setPattern(p); setUploadB64(null); }}
                  className={`h-8 px-3 rounded-full text-[11px] font-heading font-semibold tracking-tight flex items-center gap-1.5 transition-colors duration-150 border ${
                    selected
                      ? "bg-[#111] text-white border-[#111]"
                      : "bg-transparent text-[#666661] border-[#E5E5DF] hover:border-[#111] hover:text-[#111]"
                  }`}
                >
                  <PatternGlyph name={p} />
                  <span className="capitalize">{p}</span>
                </button>
              );
            })}
            <button
              data-testid="pattern-chip-upload"
              onClick={() => fileRef.current?.click()}
              className={`h-8 px-3 rounded-full text-[11px] font-heading font-semibold tracking-tight flex items-center gap-1.5 transition-colors duration-150 border ${
                pattern === "upload"
                  ? "bg-[#111] text-white border-[#111]"
                  : "bg-transparent text-[#666661] border-[#E5E5DF] hover:border-[#111] hover:text-[#111]"
              }`}
            >
              <PatternGlyph name="upload" />
              <span>{uploadB64 ? "Image" : "Upload"}</span>
            </button>
            <input
              ref={fileRef}
              type="file"
              accept="image/png,image/jpeg"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0])}
              data-testid="pattern-upload-input"
            />
          </div>

          <div className="ml-auto flex items-center gap-2">
            <select
              data-testid="model-selector"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="h-8 bg-transparent border border-transparent hover:border-[#E5E5DF] text-[11px] font-mono text-[#444440] rounded-md px-2 focus:outline-none"
            >
              {MODELS.map((m) => (
                <option key={m} value={m}>{MODEL_LABELS[m] || m}</option>
              ))}
            </select>
            <div className="hidden sm:flex items-center gap-2 text-[11px] font-mono text-[#666661]">
              <span className="uppercase tracking-[0.18em] text-[9px] text-[#888884]">tokens</span>
              <input
                type="range"
                min={40}
                max={320}
                step={10}
                value={maxTokens}
                onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
                className="w-24 accent-[#111]"
                data-testid="max-tokens-slider"
              />
              <span className="w-6 text-right">{maxTokens}</span>
            </div>
          </div>
        </div>

        {/* Textarea row */}
        <div className="flex items-end gap-2">
          <textarea
            data-testid="chat-input-textarea"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={handleKey}
            placeholder="Start a sentence and the model will complete it with a watermark woven in… (or ask a question and it'll be auto-converted)"
            rows={2}
            className="flex-1 resize-none bg-transparent text-[15px] font-body leading-relaxed text-[#111] placeholder:text-[#A8A8A1] focus:outline-none px-1 py-1 max-h-[200px]"
            style={{ minHeight: 48 }}
          />

          <div className="relative">
            <button
              data-testid="advanced-settings-trigger"
              onClick={() => setAdvOpen((v) => !v)}
              className="h-10 w-10 rounded-full border border-[#E5E5DF] hover:border-[#111] hover:bg-[#F5F5F0] transition-colors flex items-center justify-center"
              aria-label="Advanced settings"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 1 V4 M8 12 V15 M1 8 H4 M12 8 H15 M3 3 L5 5 M11 11 L13 13 M3 13 L5 11 M11 5 L13 3" stroke="#111" strokeWidth="1.4" strokeLinecap="round" />
                <circle cx="8" cy="8" r="2.2" stroke="#111" strokeWidth="1.4" fill="none" />
              </svg>
            </button>
            <AdvancedPopover
              open={advOpen}
              onClose={() => setAdvOpen(false)}
              state={advanced}
              setState={setAdvanced}
            />
          </div>

          <button
            data-testid="generate-button"
            onClick={onSubmit}
            disabled={busy || !prompt.trim()}
            className="h-10 px-5 rounded-full bg-[#111] text-white text-[12px] font-heading font-bold uppercase tracking-[0.14em] hover:bg-[#2a2a28] disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {busy ? (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-white pulse-dot" />
                <span>Working</span>
              </>
            ) : (
              <>
                <span>Generate</span>
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                  <path d="M2 6 H10 M7 3 L10 6 L7 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </>
            )}
          </button>
        </div>
      </div>

      <div className="text-[10px] font-mono text-[#888884] text-center mt-3 tracking-wide uppercase">
        Base completion models (GPT-2 / OPT) may produce incoherent output — use TinyLlama for better factual quality
      </div>
    </div>
  );
}

function StreamingMessage({ text }) {
  return (
    <div className="anim-msg-in w-full">
      <div className="flex items-start gap-3">
        <div className="mt-1 w-7 h-7 rounded-full bg-[#111] text-white flex items-center justify-center shrink-0">
          <span className="text-[10px] font-mono font-bold">WM</span>
        </div>
        <div className="flex-1">
          <div className="text-[11px] font-mono uppercase tracking-[0.2em] text-[#888884] mb-2">
            weaving pattern into tokens…
          </div>
          <div className="bg-white border border-[#E5E5DF] rounded-2xl rounded-tl-sm p-5 text-[15px] leading-[1.65] text-[#1a1a19] font-body whitespace-pre-wrap min-h-[60px]">
            {text
              ? <>{text}<span className="caret" /></>
              : <span className="text-[#A8A8A1]">loading model…<span className="caret" /></span>
            }
          </div>
        </div>
      </div>
    </div>
  );
}

function AssistantMessage({ msg, onVerify, onRedetect }) {
  const d = msg.data;
  const [editText, setEditText] = React.useState(d.generated_text);
  const textChanged = editText !== d.generated_text;
  return (
    <div className="anim-msg-in w-full">
      <div className="flex items-start gap-3">
        <div className="mt-1 w-7 h-7 rounded-full bg-[#111] text-white flex items-center justify-center shrink-0">
          <span className="text-[10px] font-mono font-bold">WM</span>
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-3 mb-2">
            <span className="font-heading font-bold text-sm">Watermarked output</span>
            <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-[#888884]">{d.model_name}</span>
          </div>

          {/* metrics */}
          <div className="flex flex-wrap gap-2 mb-3">
            <MetricPill label="tokens" value={d.token_count} />
            <MetricPill label="grid" value={`${d.rows}×${d.cols}`} sub={`${d.pattern_length} bits`} />
            <MetricPill label="delta" value={d.delta} />
            <MetricPill label="elapsed" value={`${d.elapsed_s}s`} />
          </div>

          <div
            data-testid="generated-text"
            className="relative"
          >
            <textarea
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              className="w-full bg-white border border-[#E5E5DF] rounded-2xl rounded-tl-sm p-5 text-[15px] leading-[1.65] text-[#1a1a19] font-body resize-y focus:outline-none focus:ring-2 focus:ring-[#111]/20"
              style={{ minHeight: 100 }}
              data-testid="generated-text-area"
            />
            {textChanged && (
              <div className="mt-2 flex items-center gap-3 flex-wrap">
                <span className="text-[11px] font-mono text-[#888884]">text modified — watermark may degrade</span>
                <button
                  onClick={() => setEditText(d.generated_text)}
                  className="text-[11px] font-mono text-[#888884] underline hover:text-[#111]"
                >
                  reset
                </button>
              </div>
            )}
          </div>

          {/* Target preview + verify / re-detect */}
          <div className="mt-4 flex flex-wrap items-center gap-4">
            <GridViz
              grid={d.target_grid}
              title="embedded pattern"
              caption="green = 1 · red = 0"
              variant="target"
              ariaLabel="Embedded target watermark pattern"
            />
            {!msg.detection && !textChanged && (
              <button
                data-testid="verify-watermark-button"
                onClick={() => onVerify(msg.id)}
                disabled={msg.verifying}
                className="inline-flex items-center gap-2 px-4 py-2 border border-[#111] text-[11px] font-heading font-bold uppercase tracking-[0.18em] text-[#111] hover:bg-[#111] hover:text-white transition-colors rounded-full disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {msg.verifying ? (
                  <><span className="w-1.5 h-1.5 rounded-full bg-current pulse-dot" /><span>Analysing</span></>
                ) : (
                  <><svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2 6 L5 9 L10 3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg><span>Verify watermark</span></>
                )}
              </button>
            )}
            {textChanged && (
              <button
                data-testid="redetect-button"
                onClick={() => onRedetect(msg.id, editText)}
                disabled={msg.redetecting}
                className="inline-flex items-center gap-2 px-4 py-2 border border-[#111] text-[11px] font-heading font-bold uppercase tracking-[0.18em] text-[#111] hover:bg-[#111] hover:text-white transition-colors rounded-full disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {msg.redetecting ? (
                  <><span className="w-1.5 h-1.5 rounded-full bg-current pulse-dot" /><span>Analysing</span></>
                ) : (
                  <><svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2 6 L5 9 L10 3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" /></svg><span>Re-detect edits</span></>
                )}
              </button>
            )}
          </div>

          {/* Detection verdict card */}
          {msg.detection && (
            <div
              data-testid="verdict-card"
              className="mt-5 p-5 bg-[#F5F5F0] border border-[#E5E5DF] rounded-2xl flex flex-col gap-5 anim-msg-in"
            >
              <div className="flex flex-wrap items-center gap-3">
                <VerdictChip
                  positive={msg.detection.is_watermarked}
                  label={msg.detection.is_watermarked ? "Watermarked" : "Not Watermarked"}
                />
                <MetricPill label="z-score" value={msg.detection.z_score.toFixed(2)} sub={`τ ${msg.detection.tau.toFixed(1)}`} />
                <MetricPill label="bit-acc" value={`${(msg.detection.bit_accuracy * 100).toFixed(1)}%`} sub="vs ~50%" />
                <MetricPill label="LCS" value={msg.detection.lcs_ratio.toFixed(3)} sub="info only" />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <GridViz
                  grid={msg.detection.target_grid}
                  title="target pattern"
                  variant="target"
                />
                <GridViz
                  grid={msg.detection.recovered_grid}
                  title="recovered pattern"
                  variant="recovered"
                />              </div>
              <div className="text-[11px] font-mono text-[#888884]">
                green = bit 1 recovered · red = bit 0 recovered · white = noise / modification outside LCS
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function UserMessage({ text }) {
  return (
    <div className="anim-msg-in flex justify-end">
      <div
        data-testid="user-message"
        className="max-w-[78%] bg-[#111] text-white rounded-2xl rounded-tr-sm px-4 py-3 text-[14px] leading-relaxed font-body whitespace-pre-wrap"
      >
        {text}
      </div>
    </div>
  );
}

function EmptyState() {
  const examples = [
    "Artificial intelligence is transforming research workflows in modern scientific laboratories by",
    "The Renaissance was a cultural movement that profoundly affected European intellectual life. Its legacy includes",
    "Virat Kohli is one of cricket's greatest batsmen. Born in Delhi in 1988,",
  ];
  return (
    <div className="max-w-2xl mx-auto text-center pt-10 sm:pt-16">
      <div className="inline-flex items-center gap-2 text-[10px] font-mono uppercase tracking-[0.28em] text-[#888884] mb-5">
        <span className="w-1.5 h-1.5 rounded-full bg-[#111]" /> Bashir 2025 · visual pattern watermarking
      </div>
      <h1 className="font-heading font-bold text-[40px] sm:text-[56px] leading-[1.02] tracking-[-0.02em] text-[#111]">
        A watermark you can <em className="not-italic" style={{ fontStyle: "italic", fontFamily: "'Cabinet Grotesk', serif" }}>see</em>.
      </h1>
      <p className="mt-5 text-[15px] text-[#555550] leading-relaxed max-w-xl mx-auto">
        Embed a visual binary pattern into the token stream of an LLM, then recover and visualise it as proof of origin — all in a single chat.
      </p>

      <div className="mt-10 grid grid-cols-1 sm:grid-cols-3 gap-2 text-left" data-testid="example-prompts">
        {examples.map((ex, i) => (
          <button
            key={i}
            data-testid={`example-prompt-${i}`}
            className="group px-4 py-3 rounded-xl border border-[#E5E5DF] hover:border-[#111] bg-white/60 hover:bg-white transition-colors text-[13px] leading-snug text-[#333330]"
            onClick={() => window.dispatchEvent(new CustomEvent("wm_example", { detail: ex }))}
          >
            <span className="block text-[9px] font-mono uppercase tracking-[0.22em] text-[#888884] mb-1">try</span>
            {ex}
          </button>
        ))}
      </div>
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [pattern, setPattern] = useState("checkerboard");
  const [uploadB64, setUploadB64] = useState(null);
  const [model, setModel] = useState("HuggingFaceTB/SmolLM2-135M-Instruct");
  const [maxTokens, setMaxTokens] = useState(120);
  const [advanced, setAdvanced] = useState({
    secret_key: "llmwatermark",
    gamma: 0.5,
    delta: 4.0,
    tau: 4.0,
    temperature: 0.8,
    top_p: 0.9,
  });
  const [howOpen, setHowOpen] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    const h = (e) => setPrompt(e.detail);
    window.addEventListener("wm_example", h);
    return () => window.removeEventListener("wm_example", h);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [messages]);

  const handleSubmit = async () => {
    const p = prompt.trim();
    if (!p || busy) return;
    const userMsg = { id: `u-${Date.now()}`, role: "user", text: p };
    const pendingId = `a-${Date.now()}`;
    setMessages((m) => [...m, userMsg, { id: pendingId, role: "assistant", streaming: true, streamText: "", data: null, session_id: null }]);
    setPrompt("");
    setBusy(true);

    try {
      const body = {
        prompt: p,
        model_name: model,
        max_new_tokens: maxTokens,
        pattern,
        pattern_image_b64: pattern === "upload" ? uploadB64 : null,
        ...advanced,
      };

      const resp = await fetch(`${API}/watermark/generate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `request failed: ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop();
        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data: ")) continue;
          const evt = JSON.parse(line.slice(6));
          if (evt.type === "token") {
            setMessages((m) => m.map((x) => x.id === pendingId ? { ...x, streamText: (x.streamText || "") + evt.text } : x));
          } else if (evt.type === "done") {
            setMessages((m) => m.map((x) => x.id === pendingId ? { ...x, streaming: false, data: evt, session_id: evt.session_id } : x));
          } else if (evt.type === "error") {
            throw new Error(evt.text);
          }
        }
      }
    } catch (e) {
      const errText = e.message || "Generation failed.";
      setMessages((m) => m.filter((x) => x.id !== pendingId).concat({ id: `e-${Date.now()}`, role: "error", text: errText }));
    } finally {
      setBusy(false);
    }
  };

  const handleVerify = async (msgId) => {
    setMessages((m) => m.map((x) => (x.id === msgId ? { ...x, verifying: true } : x)));
    const msg = messages.find((x) => x.id === msgId);
    try {
      const res = await axios.post(`${API}/watermark/detect`, { session_id: msg.session_id }, { timeout: 120000 });
      setMessages((m) => m.map((x) => (x.id === msgId ? { ...x, verifying: false, detection: res.data } : x)));
    } catch (e) {
      const errText = e?.response?.data?.detail || e.message || "Detection failed.";
      setMessages((m) =>
        m.map((x) => (x.id === msgId ? { ...x, verifying: false } : x)).concat({
          id: `e-${Date.now()}`,
          role: "error",
          text: errText,
        })
      );
    }
  };

  const handleRedetect = async (msgId, newText) => {
    setMessages((m) => m.map((x) => (x.id === msgId ? { ...x, redetecting: true } : x)));
    const msg = messages.find((x) => x.id === msgId);
    try {
      const res = await axios.post(`${API}/watermark/detect-text`, {
        session_id: msg.session_id,
        text: newText,
      }, { timeout: 120000 });
      setMessages((m) => m.map((x) => (x.id === msgId ? { ...x, redetecting: false, detection: res.data } : x)));
    } catch (e) {
      const errText = e?.response?.data?.detail || e.message || "Detection failed.";
      setMessages((m) => m.map((x) => (x.id === msgId ? { ...x, redetecting: false } : x)).concat({
        id: `e-${Date.now()}`,
        role: "error",
        text: errText,
      }));
    }
  };

  return (
    <div className="App app-grain min-h-screen flex flex-col">
      {/* Header */}
      <header
        data-testid="app-header"
        className="fixed top-0 left-0 right-0 h-16 flex items-center justify-between px-5 sm:px-8 z-30 backdrop-blur-md bg-[#F5F5F0]/75 border-b border-[#E5E5DF]"
      >
        <div className="flex items-baseline gap-3">
          <div className="w-6 h-6 bg-[#111] flex items-center justify-center" aria-hidden="true">
            <div className="w-2 h-2 bg-[#10B981]" />
          </div>
          <span className="font-heading font-bold text-[15px] tracking-tight">Watermark Studio</span>
          <span className="hidden sm:inline text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884]">
            / visual pattern LLM watermarking
          </span>
        </div>
        <div className="flex items-center gap-3">
          <a
            href="https://arxiv.org/"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:inline text-[11px] font-mono text-[#666661] hover:text-[#111] transition-colors"
          >
            Bashir 2025 ↗
          </a>
          <button
            data-testid="how-it-works-trigger"
            onClick={() => setHowOpen(true)}
            className="h-9 px-4 rounded-full text-[11px] font-heading font-bold uppercase tracking-[0.18em] border border-[#111] text-[#111] hover:bg-[#111] hover:text-white transition-colors"
          >
            How it works
          </button>
        </div>
      </header>

      {/* Scrollable chat area */}
      <main
        ref={scrollRef}
        className="flex-1 overflow-y-auto pt-24 pb-56 px-4 sm:px-6"
        data-testid="chat-area"
      >
        <div className="max-w-3xl mx-auto space-y-8">
          {messages.length === 0 && <EmptyState />}

          {messages.map((m) => {
            if (m.role === "user") return <UserMessage key={m.id} text={m.text} />;
            if (m.role === "assistant" && m.streaming) return <StreamingMessage key={m.id} text={m.streamText || ""} />;
            if (m.role === "assistant") return <AssistantMessage key={m.id} msg={m} onVerify={handleVerify} onRedetect={handleRedetect} />;
            return (
              <div
                key={m.id}
                data-testid="error-message"
                className="anim-msg-in max-w-[90%] mx-auto px-4 py-3 rounded-xl border border-[#EF4444]/25 bg-[#EF4444]/5 text-[#B91C1C] text-[13px] font-mono"
              >
                {m.text}
              </div>
            );
          })}

        </div>
      </main>

      {/* Fixed input area */}
      <div className="fixed bottom-0 left-0 right-0 z-20 px-4 sm:px-6 pb-5 pointer-events-none bg-gradient-to-t from-[#F5F5F0] via-[#F5F5F0]/95 to-transparent pt-12">
        <ChatInput
          prompt={prompt}
          setPrompt={setPrompt}
          pattern={pattern}
          setPattern={setPattern}
          uploadB64={uploadB64}
          setUploadB64={setUploadB64}
          model={model}
          setModel={setModel}
          maxTokens={maxTokens}
          setMaxTokens={setMaxTokens}
          advanced={advanced}
          setAdvanced={setAdvanced}
          onSubmit={handleSubmit}
          busy={busy}
        />
      </div>

      <HowItWorksPanel open={howOpen} onClose={() => setHowOpen(false)} />
    </div>
  );
}

export default App;
