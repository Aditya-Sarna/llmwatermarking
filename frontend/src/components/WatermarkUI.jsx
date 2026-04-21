import React from "react";

// Tiny inline SVG previews for each built-in pattern (used in picker chips)
export function PatternGlyph({ name, size = 14 }) {
  const s = size;
  const stroke = "currentColor";
  switch (name) {
    case "square":
      return (
        <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
          <rect x="2" y="2" width="10" height="10" fill={stroke} />
        </svg>
      );
    case "checkerboard":
      return (
        <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
          <rect x="0" y="0" width="7" height="7" fill={stroke} />
          <rect x="7" y="7" width="7" height="7" fill={stroke} />
        </svg>
      );
    case "circle":
      return (
        <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
          <circle cx="7" cy="7" r="5" stroke={stroke} strokeWidth="1.6" fill="none" />
        </svg>
      );
    case "diamond":
      return (
        <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
          <path d="M7 1 L13 7 L7 13 L1 7 Z" stroke={stroke} strokeWidth="1.4" fill="none" />
        </svg>
      );
    case "cross":
      return (
        <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
          <rect x="5.5" y="1" width="3" height="12" fill={stroke} />
          <rect x="1" y="5.5" width="12" height="3" fill={stroke} />
        </svg>
      );
    case "upload":
      return (
        <svg width={s} height={s} viewBox="0 0 14 14" fill="none">
          <path d="M7 10 V3 M4 6 L7 3 L10 6 M2 12 H12" stroke={stroke} strokeWidth="1.4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      );
    default:
      return null;
  }
}

// Visualize a rows x cols grid where each cell is "1" (hit/green), "0" (miss/red), or "-" (noise/white)
// Variant "target": every cell is green(1)/red(0), no noise.
// Variant "recovered": green(1) / red(0) cells inside LCS; white "-" cells for noise.
export function GridViz({ grid, title, caption, variant = "target", ariaLabel }) {
  if (!grid || grid.length === 0) return null;
  const rows = grid.length;
  const cols = grid[0].length;
  // Dynamic cell size: fit comfortably up to ~14 cols
  const cellPx = cols >= 16 ? 14 : cols >= 12 ? 18 : cols >= 9 ? 22 : 26;

  return (
    <div className="flex flex-col gap-2" data-testid={`grid-viz-${variant}`}>
      {title && (
        <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-[#666661]">
          {title}
        </div>
      )}
      <div
        role="img"
        aria-label={ariaLabel || title}
        className="inline-grid bg-[#E5E5DF] p-[1px] border border-[#E5E5DF] rounded-sm"
        style={{
          gridTemplateColumns: `repeat(${cols}, ${cellPx}px)`,
          gridAutoRows: `${cellPx}px`,
          gap: 1,
          width: "fit-content",
        }}
      >
        {grid.flat().map((v, i) => {
          const bg =
            v === "1" ? "#10B981" : v === "0" ? "#EF4444" : "#FFFFFF";
          const delay = (i % cols) * 6 + Math.floor(i / cols) * 4;
          return (
            <div
              key={i}
              className="cell-pop"
              style={{
                width: cellPx,
                height: cellPx,
                background: bg,
                animationDelay: `${delay}ms`,
              }}
            />
          );
        })}
      </div>
      {caption && (
        <div className="text-[11px] font-mono text-[#666661]">{caption}</div>
      )}
    </div>
  );
}

export function MetricPill({ label, value, sub }) {
  return (
    <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded bg-[#F5F5F0] border border-[#E5E5DF]">
      <span className="text-[9px] font-mono uppercase tracking-[0.18em] text-[#888884]">{label}</span>
      <span className="text-xs font-mono font-semibold text-[#111]">{value}</span>
      {sub && <span className="text-[10px] font-mono text-[#888884]">{sub}</span>}
    </div>
  );
}

export function VerdictChip({ positive, label }) {
  const cls = positive
    ? "bg-[#10B981]/10 border-[#10B981]/25 text-[#0E7C5A]"
    : "bg-[#EF4444]/10 border-[#EF4444]/25 text-[#B91C1C]";
  return (
    <span
      data-testid={positive ? "verdict-chip-positive" : "verdict-chip-negative"}
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full border font-mono text-[11px] font-bold tracking-[0.18em] uppercase ${cls}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${positive ? "bg-[#10B981]" : "bg-[#EF4444]"}`} />
      {label}
    </span>
  );
}
