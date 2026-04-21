import React from "react";

export function HowItWorksPanel({ open, onClose }) {
  return (
    <>
      <div
        className={`fixed inset-0 z-40 bg-[#111]/20 transition-opacity duration-200 ${
          open ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
        onClick={onClose}
        aria-hidden={!open}
      />
      <aside
        data-testid="how-it-works-panel"
        className={`fixed top-0 right-0 z-50 h-full w-full sm:w-[440px] bg-white border-l border-[#E5E5DF] shadow-2xl transform transition-transform duration-300 ease-out ${
          open ? "translate-x-0" : "translate-x-full"
        }`}
        aria-label="How it works"
      >
        <div className="h-16 flex items-center justify-between px-6 border-b border-[#E5E5DF]">
          <div className="flex items-baseline gap-3">
            <span className="text-[10px] font-mono uppercase tracking-[0.25em] text-[#888884]">Method</span>
            <span className="font-heading font-bold text-base">How it works</span>
          </div>
          <button
            onClick={onClose}
            data-testid="how-it-works-close"
            className="w-9 h-9 rounded-full flex items-center justify-center hover:bg-[#F5F5F0] transition-colors"
            aria-label="Close"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M3 3 L13 13 M13 3 L3 13" stroke="#111" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>
        <div className="overflow-y-auto h-[calc(100%-4rem)] px-6 py-6 space-y-6 text-[14px] leading-relaxed text-[#2a2a28]">
          <section>
            <div className="text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884] mb-2">Paper</div>
            <p className="font-heading font-bold text-xl leading-snug">
              Visual Pattern-Based Watermarking for Large Language Model Generated Text
            </p>
            <p className="text-[12px] text-[#666661] mt-1">Shariq Bashir, IMSIU (2025)</p>
          </section>

          <div className="rule-mono" />

          <section className="space-y-3">
            <div className="text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884]">Embedding</div>
            <ol className="space-y-2 list-decimal pl-5">
              <li>A binary image <b>P</b> is resized to an r×c grid that matches the generation length.</li>
              <li>At each token step, the vocabulary is partitioned into a <b>green list</b> and <b>red list</b> using SHA3-256 keyed on (previous token, step, secret key).</li>
              <li>A logit bias <b>δ</b> is applied: bit=1 boosts green tokens; bit=0 boosts red tokens.</li>
              <li>Nucleus sampling picks the next token under the biased distribution.</li>
              <li>Generation runs via <span className="font-mono text-[12px]">model.generate()</span> with a <span className="font-mono text-[12px]">LogitsProcessor</span> — KV cache preserved.</li>
            </ol>
          </section>

          <section className="space-y-3">
            <div className="text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884]">Detection</div>
            <ol className="space-y-2 list-decimal pl-5">
              <li>The detector rebuilds the identical green/red partitions from the same secret key.</li>
              <li>Each token is assigned bit 1 (green) or 0 (red).</li>
              <li><b>LCS ratio</b> = |LCS(pattern, recovered)| / |pattern|. Values above τ indicate a watermark.</li>
              <li>The recovered pattern is visualised as a 2D image; a KGW-style z-score is also reported.</li>
            </ol>
          </section>

          <div className="rule-mono" />

          <section>
            <div className="text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884] mb-2">Parameters</div>
            <div className="grid grid-cols-3 gap-2 text-[12px]">
              <div className="font-mono text-[#888884]">delta</div><div className="col-span-2">0.2 – 0.4: stronger watermark, lower text quality at high values</div>
              <div className="font-mono text-[#888884]">gamma</div><div className="col-span-2">0.5: fraction of vocab in green list</div>
              <div className="font-mono text-[#888884]">tau</div><div className="col-span-2">0.70 – 0.80: detection threshold (LCS ratio)</div>
              <div className="font-mono text-[#888884]">key</div><div className="col-span-2">any string; must match between embed and detect</div>
            </div>
          </section>

          <div className="rule-mono" />

          <section className="text-[12px] text-[#666661]">
            This demo uses local HuggingFace models (gpt2, gpt2-medium, facebook/opt-125m) because watermarking requires direct logit-level control, which hosted APIs do not expose.
          </section>
        </div>
      </aside>
    </>
  );
}
