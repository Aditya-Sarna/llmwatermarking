import React from "react";
import { Link } from "react-router-dom";
import "@/App.css";
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from "@/components/ui/accordion";
import { MetricPill, VerdictChip } from "@/components/WatermarkUI";

const SECTIONS = [
  {
    id: "basics",
    label: "The basics",
    items: [
      {
        q: "What is Watermark Studio?",
        a: (
          <>
            <p>
              Watermark Studio is an interactive demo of <b>visual pattern watermarking</b> for AI-generated text. While a language model writes, we quietly steer its word choices so that a hidden picture, such as a checkerboard or a circle, gets woven into the text.
            </p>
            <p>
              Later, anyone with the same secret key can recover that picture from the text alone. The recovered picture is the proof that the text came from a watermarked model.
            </p>
          </>
        ),
      },
      {
        q: "Why would anyone want to watermark AI text?",
        a: (
          <p>
            As AI writing becomes harder to tell apart from human writing, organisations need a reliable way to show where a piece of text came from. A watermark lets a provider say "yes, our model produced this" without storing every output. It does this without changing how the text reads to a person.
          </p>
        ),
      },
      {
        q: "Can I see the watermark by reading the text?",
        a: (
          <p>
            No. The text reads like ordinary model output. The watermark lives in small statistical preferences for some words over others, and only a detector holding the secret key can see the pattern.
          </p>
        ),
      },
      {
        q: "What research is this based on?",
        a: (
          <p>
            The method follows <i>Visual Pattern-Based Watermarking for Large Language Model Generated Text</i> by Shariq Bashir (IMSIU, 2025). For a step-by-step technical summary, open the <b>How it works</b> panel in the studio header.
          </p>
        ),
      },
    ],
  },
  {
    id: "getting-started",
    label: "Getting started",
    items: [
      {
        q: "How do I create my first watermarked text?",
        a: (
          <ol className="list-decimal pl-5 space-y-1.5">
            <li>Choose a pattern chip above the input box. <b>Checkerboard</b> is a good first pick.</li>
            <li>Type the start of a sentence, or click one of the example prompts.</li>
            <li>Press <b>Generate</b>, or hit Enter. The text appears word by word as it is written.</li>
            <li>When it finishes, click <b>Verify watermark</b> to recover the hidden pattern.</li>
          </ol>
        ),
      },
      {
        q: "Should I write a question or the start of a sentence?",
        a: (
          <p>
            The start of a sentence usually works best, for example <i>"The history of the printing press begins with"</i>. If you type a question, the studio converts it into a sentence opener for you. Longer, open-ended prompts give the model more room to write, and more room to hold the watermark.
          </p>
        ),
      },
      {
        q: "Do I need an account, an API key or a payment method?",
        a: (
          <p>
            No. The demo runs a small model on our own server. There is nothing to sign up for and it costs you nothing.
          </p>
        ),
      },
      {
        q: "Can I use my own image as the pattern?",
        a: (
          <p>
            Yes. Click the <b>Upload</b> chip and choose a PNG or JPEG. The image is converted to a simple black-and-white grid sized to fit the length of the text. Bold, high-contrast shapes such as logos, letters and simple icons survive this conversion best. Fine detail is lost.
          </p>
        ),
      },
    ],
  },
  {
    id: "results",
    label: "Reading the results",
    items: [
      {
        q: "What do the green and red squares mean?",
        a: (
          <>
            <p>
              Each square in a grid stands for one word-piece (token) in the generated text. <span className="text-[#0E7C5A] font-semibold">Green</span> means the hidden bit was a 1. <span className="text-[#B91C1C] font-semibold">Red</span> means it was a 0. Together, the squares draw the pattern you picked.
            </p>
            <p>
              In the <b>recovered pattern</b>, white squares mark positions the detector could not match. This is usually caused by noise or by edits to the text.
            </p>
          </>
        ),
      },
      {
        q: "How do I know whether the watermark was found?",
        a: (
          <>
            <p>After verification, you will see one of two verdicts:</p>
            <div className="flex flex-wrap gap-2 my-3">
              <VerdictChip positive label="Watermarked" />
              <VerdictChip positive={false} label="Not Watermarked" />
            </div>
            <p>
              The verdict comes from the <b>z-score</b>, a measure of how far the text departs from pure chance. If the z-score is above the threshold (τ), the text counts as watermarked.
            </p>
          </>
        ),
      },
      {
        q: "What are z-score, bit-acc and LCS?",
        a: (
          <div className="space-y-3">
            <div className="flex flex-wrap gap-2">
              <MetricPill label="z-score" value="6.84" sub="τ 4.0" />
              <MetricPill label="bit-acc" value="82.5%" sub="vs ~50%" />
              <MetricPill label="LCS" value="0.911" sub="info only" />
            </div>
            <ul className="list-disc pl-5 space-y-1.5">
              <li><b>z-score</b> is the main signal. Higher values mean stronger evidence of a watermark. Unwatermarked text usually sits near 0.</li>
              <li><b>bit-acc</b> is the share of squares recovered correctly. Random text lands around 50%, so anything well above that is meaningful.</li>
              <li><b>LCS</b> measures how much of the pattern survives in the right order. It is shown for reference only and does not affect the verdict.</li>
            </ul>
          </div>
        ),
      },
      {
        q: "What happens if I edit the generated text?",
        a: (
          <p>
            Try it. The output box is editable. After you change a few words, a <b>Re-detect edits</b> button appears. It shows how well the watermark holds up against your edits. Small edits usually leave it detectable. Heavy rewriting gradually erases it. Click <b>reset</b> to go back to the original text.
          </p>
        ),
      },
    ],
  },
  {
    id: "settings",
    label: "Settings",
    items: [
      {
        q: "What do the Advanced settings do?",
        a: (
          <div className="grid grid-cols-[110px_1fr] gap-x-4 gap-y-2 text-[13px]">
            <div className="font-mono text-[#888884]">secret key</div><div>The password for the watermark. Detection only works with the same key used during generation.</div>
            <div className="font-mono text-[#888884]">gamma</div><div>The share of the vocabulary placed on the "green list" at each step. 0.5 is a balanced default.</div>
            <div className="font-mono text-[#888884]">delta</div><div>How strongly the model is nudged. Higher values make the watermark stronger but the writing less natural.</div>
            <div className="font-mono text-[#888884]">tau</div><div>The z-score a text must reach to count as watermarked. Raise it to be stricter.</div>
            <div className="font-mono text-[#888884]">temperature</div><div>How adventurous the word choices are. Lower values make the output more predictable.</div>
            <div className="font-mono text-[#888884]">top-p</div><div>Limits choices to the most likely words. Lower values make the output more focused.</div>
          </div>
        ),
      },
      {
        q: "How many tokens should I generate?",
        a: (
          <p>
            The <b>tokens</b> slider sets the length of the output, from 40 to 320 tokens. Longer outputs mean a larger, more detailed grid and a more confident verdict, but they take longer to produce. 120 tokens is a good balance for a first try.
          </p>
        ),
      },
      {
        q: "I'm not sure what to change. What are safe defaults?",
        a: (
          <p>
            The defaults are tuned for this demo: key <span className="font-mono text-[12px]">llmwatermark</span>, gamma 0.50, delta 1.0, tau 4.0, temperature 0.8 and top-p 0.90. If verification fails, raise <b>delta</b> by one step or generate more tokens.
          </p>
        ),
      },
    ],
  },
  {
    id: "troubleshooting",
    label: "Troubleshooting",
    items: [
      {
        q: "Why is the generated text strange or factually wrong?",
        a: (
          <p>
            This is expected. The demo uses a deliberately tiny model, <span className="font-mono text-[12px]">SmolLM2 · 135M</span>, so that it runs quickly and for free. The model often wanders off topic or invents facts. The purpose of the demo is to show the watermarking technique, not to produce useful answers.
          </p>
        ),
      },
      {
        q: "The first generation is slow. Is something broken?",
        a: (
          <p>
            Probably not. The first request may show <i>"loading model…"</i> while the model warms up. Later generations are much faster.
          </p>
        ),
      },
      {
        q: "Verification says \"Not Watermarked\" on my own output. Why?",
        a: (
          <ul className="list-disc pl-5 space-y-1.5">
            <li>The output may be too short. Try 120 tokens or more.</li>
            <li>Delta may be too low to leave a clear signal. Try 1.5 or 2.0.</li>
            <li>Tau may be set very high. Lower it toward 4.0.</li>
            <li>If you edited the text, larger edits remove more of the watermark.</li>
          </ul>
        ),
      },
      {
        q: "Is my prompt stored or shared?",
        a: (
          <p>
            Prompts and outputs are sent to the demo server only to generate and verify text. Please do not enter personal or confidential information into a public demo.
          </p>
        ),
      },
    ],
  },
];

export default function FaqPage() {
  return (
    <div className="App app-grain min-h-screen flex flex-col" data-testid="faq-page">
      <header className="fixed top-0 left-0 right-0 h-16 flex items-center justify-between px-5 sm:px-8 z-30 backdrop-blur-md bg-[#F5F5F0]/75 border-b border-[#E5E5DF]">
        <Link to="/" className="flex items-baseline gap-3">
          <div className="w-6 h-6 bg-[#111] flex items-center justify-center self-center" aria-hidden="true">
            <div className="w-2 h-2 bg-[#10B981]" />
          </div>
          <span className="font-heading font-bold text-[15px] tracking-tight">Watermark Studio</span>
          <span className="hidden sm:inline text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884]">/ faq</span>
        </Link>
        <Link
          to="/"
          data-testid="faq-back-to-studio"
          className="h-9 px-4 rounded-full text-[11px] font-heading font-bold uppercase tracking-[0.18em] border border-[#111] text-[#111] hover:bg-[#111] hover:text-white transition-colors inline-flex items-center"
        >
          Open studio
        </Link>
      </header>

      <main className="flex-1 pt-28 pb-24 px-4 sm:px-6">
        <div className="max-w-5xl mx-auto">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 text-[10px] font-mono uppercase tracking-[0.28em] text-[#888884] mb-5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#111]" /> New here? Start with these
            </div>
            <h1 className="font-heading font-bold text-[40px] sm:text-[52px] leading-[1.02] tracking-[-0.02em] text-[#111]">
              Frequently asked questions
            </h1>
            <p className="mt-5 text-[15px] text-[#555550] leading-relaxed">
              Everything you need to create your first watermarked text, read the results and understand the settings, all in plain language.
            </p>
          </div>

          <div className="mt-12 grid grid-cols-1 md:grid-cols-[200px_1fr] gap-10">
            <nav className="hidden md:block" aria-label="FAQ sections">
              <div className="sticky top-24 flex flex-col gap-1">
                <span className="text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884] mb-2">Sections</span>
                {SECTIONS.map((s, i) => (
                  <a
                    key={s.id}
                    href={`#${s.id}`}
                    className="group flex items-baseline gap-2 py-1.5 text-[13px] text-[#666661] hover:text-[#111] transition-colors"
                  >
                    <span className="font-mono text-[10px] text-[#A8A8A1] group-hover:text-[#111]">0{i + 1}</span>
                    {s.label}
                  </a>
                ))}
              </div>
            </nav>

            <div className="space-y-12">
              {SECTIONS.map((s, i) => (
                <section key={s.id} id={s.id} className="scroll-mt-24" data-testid={`faq-section-${s.id}`}>
                  <div className="flex items-baseline gap-3 mb-3">
                    <span className="text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884]">0{i + 1}</span>
                    <h2 className="font-heading font-bold text-xl tracking-tight">{s.label}</h2>
                  </div>
                  <div className="bg-white border border-[#E5E5DF] rounded-2xl px-5 sm:px-6">
                    <Accordion type="multiple">
                      {s.items.map((item, j) => (
                        <AccordionItem
                          key={j}
                          value={`${s.id}-${j}`}
                          className={`border-[#E5E5DF] ${j === s.items.length - 1 ? "border-b-0" : ""}`}
                        >
                          <AccordionTrigger className="font-heading font-bold text-[15px] text-[#111] py-5 hover:no-underline hover:text-[#444440]">
                            {item.q}
                          </AccordionTrigger>
                          <AccordionContent className="pb-5 text-[14px] leading-relaxed text-[#2a2a28] space-y-3">
                            {item.a}
                          </AccordionContent>
                        </AccordionItem>
                      ))}
                    </Accordion>
                  </div>
                </section>
              ))}

              <div className="rule-mono" />

              <div className="p-6 sm:p-8 bg-[#111] text-white rounded-2xl flex flex-col sm:flex-row sm:items-center gap-5 justify-between">
                <div>
                  <div className="text-[10px] font-mono uppercase tracking-[0.22em] text-white/50 mb-2">Ready?</div>
                  <p className="font-heading font-bold text-xl leading-snug">Weave your first pattern into text.</p>
                  <p className="text-[13px] text-white/60 mt-1">It takes less than a minute and needs no account.</p>
                </div>
                <Link
                  to="/"
                  data-testid="faq-cta"
                  className="h-10 px-5 rounded-full bg-white text-[#111] text-[12px] font-heading font-bold uppercase tracking-[0.14em] hover:bg-[#F5F5F0] transition-colors inline-flex items-center gap-2 self-start sm:self-auto"
                >
                  <span>Try it now</span>
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                    <path d="M2 6 H10 M7 3 L10 6 L7 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
