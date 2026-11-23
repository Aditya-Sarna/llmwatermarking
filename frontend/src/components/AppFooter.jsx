import React from "react";

const MOTIF = [
  "1", "0", "1", "0", "1",
  "0", "1", "0", "1", "0",
  "1", "0", "1", "0", "1",
];

function FooterHeading({ children }) {
  return (
    <div className="text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884] mb-3">
      {children}
    </div>
  );
}

function ExternalLink({ href, children, testId }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      data-testid={testId}
      className="text-[13px] text-[#444440] hover:text-[#111] transition-colors inline-flex items-center gap-1"
    >
      {children}
      <span aria-hidden="true" className="text-[11px] font-mono text-[#A8A8A1]">↗</span>
    </a>
  );
}

export function AppFooter({ onOpenHowItWorks, onBackToTop }) {
  const year = new Date().getFullYear();

  return (
    <footer
      data-testid="app-footer"
      className="max-w-3xl mx-auto mt-20 pt-8 border-t border-[#E5E5DF]"
    >
      <div className="grid grid-cols-1 sm:grid-cols-[1.4fr_1fr_1fr] gap-8">
        {/* Brand */}
        <div>
          <div className="flex items-center gap-3">
            <div className="w-6 h-6 bg-[#111] flex items-center justify-center" aria-hidden="true">
              <div className="w-2 h-2 bg-[#10B981]" />
            </div>
            <span className="font-heading font-bold text-[15px] tracking-tight">Watermark Studio</span>
          </div>
          <p className="mt-3 text-[13px] leading-relaxed text-[#666661] max-w-xs">
            Weaves a visual binary pattern into the tokens a language model writes, then recovers it as
            visible proof of where the text came from.
          </p>
          <div
            aria-hidden="true"
            className="mt-4 inline-grid bg-[#E5E5DF] p-[1px] rounded-sm"
            style={{ gridTemplateColumns: "repeat(5, 8px)", gridAutoRows: "8px", gap: 1 }}
          >
            {MOTIF.map((v, i) => (
              <div key={i} style={{ background: v === "1" ? "#10B981" : "#EF4444", opacity: 0.85 }} />
            ))}
          </div>
        </div>

        {/* Method */}
        <div>
          <FooterHeading>Method</FooterHeading>
          <ul className="space-y-2">
            <li>
              <button
                type="button"
                data-testid="footer-how-it-works"
                onClick={onOpenHowItWorks}
                className="text-[13px] text-[#444440] hover:text-[#111] transition-colors"
              >
                How it works
              </button>
            </li>
            <li>
              <ExternalLink href="https://arxiv.org/" testId="footer-paper-link">
                Bashir 2025 paper
              </ExternalLink>
            </li>
            <li>
              <ExternalLink href="https://arxiv.org/abs/2301.10226" testId="footer-kgw-link">
                KGW green-list watermark
              </ExternalLink>
            </li>
          </ul>
        </div>

        {/* Model & defaults */}
        <div>
          <FooterHeading>Model</FooterHeading>
          <ul className="space-y-2">
            <li>
              <ExternalLink
                href="https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct"
                testId="footer-model-link"
              >
                SmolLM2 · 135M
              </ExternalLink>
            </li>
          </ul>
          <div className="mt-4 flex flex-col gap-1 text-[11px] font-mono text-[#888884]">
            <span>defaults</span>
            <span className="text-[#444440]">γ 0.50 · δ 1.0 · τ 4.0</span>
          </div>
        </div>
      </div>

      <div className="rule-mono mt-8" />

      <div className="mt-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-[11px] font-mono text-[#888884]">
        <span>© {year} Watermark Studio · research demonstration, not for production use</span>
        <button
          type="button"
          data-testid="footer-back-to-top"
          onClick={onBackToTop}
          className="self-start sm:self-auto inline-flex items-center gap-1.5 uppercase tracking-[0.18em] text-[10px] text-[#666661] hover:text-[#111] transition-colors"
        >
          <svg width="10" height="10" viewBox="0 0 12 12" fill="none" aria-hidden="true">
            <path d="M6 10 V2 M3 5 L6 2 L9 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          Back to top
        </button>
      </div>
    </footer>
  );
}
