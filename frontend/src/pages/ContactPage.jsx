import React, { useState } from "react";
import { Link } from "react-router-dom";
import "@/App.css";

const CONTACT_EMAIL = "hello@watermarkstudio.dev";

const TOPICS = [
  { value: "research", label: "Research question" },
  { value: "bug", label: "Bug report" },
  { value: "collaboration", label: "Collaboration / partnership" },
  { value: "press", label: "Press & media" },
  { value: "other", label: "Something else" },
];

const CHANNELS = [
  {
    tag: "research",
    title: "Questions about the method",
    body: "Curious how green/red partitioning, the logit bias or LCS-based detection works? We're happy to walk through the details.",
    meta: "reply within 2 business days",
  },
  {
    tag: "support",
    title: "Something not working",
    body: "If generation stalls, verification fails or a pattern upload is rejected, tell us what you tried and what you saw.",
    meta: "include prompt + pattern if you can",
  },
  {
    tag: "partners",
    title: "Collaborations & press",
    body: "Working on provenance, AI safety or content authenticity? We'd love to hear about pilots, talks and joint research.",
    meta: "reply within 1 week",
  },
];

const FAQ = [
  {
    q: "Can I use the watermarking approach in my own product?",
    a: "The method is described openly in the Bashir 2025 paper. Reach out if you'd like guidance on adapting it to a specific model or deployment.",
  },
  {
    q: "Why are the generated answers sometimes nonsensical?",
    a: "The demo runs a tiny 135M-parameter local model so it stays free and private. It's there to show the watermark, not to give accurate answers.",
  },
  {
    q: "Do you store what I type into the studio?",
    a: "Prompts are only used to generate and verify the watermark for your session. If you have a specific privacy question, send us a note.",
  },
];

const EMPTY = { name: "", email: "", topic: "research", message: "" };

function validate(values) {
  const errors = {};
  if (!values.name.trim()) errors.name = "Please tell us your name.";
  if (!values.email.trim()) errors.email = "We need an email to reply to.";
  else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(values.email.trim())) errors.email = "That email doesn't look quite right.";
  if (values.message.trim().length < 10) errors.message = "A few more words would help us understand (10+ characters).";
  return errors;
}

function FieldLabel({ htmlFor, children, hint }) {
  return (
    <div className="flex items-baseline justify-between mb-1.5">
      <label htmlFor={htmlFor} className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#888884]">
        {children}
      </label>
      {hint && <span className="text-[10px] font-mono text-[#A8A8A1]">{hint}</span>}
    </div>
  );
}

function FieldError({ id, children }) {
  if (!children) return null;
  return (
    <p id={id} className="mt-1.5 text-[11px] font-mono text-[#B91C1C]">
      {children}
    </p>
  );
}

const inputCls = (hasError) =>
  `w-full bg-[#F5F5F0] border rounded-md text-[14px] px-3 text-[#111] placeholder:text-[#A8A8A1] focus:outline-none focus:ring-2 focus:ring-[#111]/20 transition-colors ${
    hasError ? "border-[#EF4444]/50" : "border-transparent"
  }`;

function ContactForm() {
  const [values, setValues] = useState(EMPTY);
  const [errors, setErrors] = useState({});
  const [sent, setSent] = useState(false);

  const update = (key) => (e) => {
    const next = { ...values, [key]: e.target.value };
    setValues(next);
    if (errors[key]) setErrors(validate(next));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const found = validate(values);
    setErrors(found);
    if (Object.keys(found).length > 0) return;

    const topicLabel = TOPICS.find((t) => t.value === values.topic)?.label || "General";
    const subject = `[Watermark Studio] ${topicLabel} from ${values.name.trim()}`;
    const body = `${values.message.trim()}\n\n—\nName: ${values.name.trim()}\nEmail: ${values.email.trim()}\nTopic: ${topicLabel}`;
    window.location.href = `mailto:${CONTACT_EMAIL}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
    setSent(true);
  };

  if (sent) {
    return (
      <div
        data-testid="contact-success"
        className="anim-msg-in bg-white border border-[#E5E5DF] rounded-2xl p-8 shadow-[0_8px_32px_rgba(17,17,17,0.06)]"
      >
        <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border font-mono text-[11px] font-bold tracking-[0.18em] uppercase bg-[#10B981]/10 border-[#10B981]/25 text-[#0E7C5A]">
          <span className="w-1.5 h-1.5 rounded-full bg-[#10B981]" />
          Draft ready
        </span>
        <h2 className="mt-5 font-heading font-bold text-2xl tracking-tight text-[#111]">
          Thanks, {values.name.trim().split(" ")[0]}.
        </h2>
        <p className="mt-3 text-[14px] leading-relaxed text-[#555550]">
          Your email app should have opened with your message filled in — just hit send. If nothing opened,
          write to us directly at{" "}
          <a href={`mailto:${CONTACT_EMAIL}`} className="font-mono text-[13px] text-[#111] underline underline-offset-4">
            {CONTACT_EMAIL}
          </a>
          .
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() => { setValues(EMPTY); setErrors({}); setSent(false); }}
            className="h-10 px-5 rounded-full border border-[#111] text-[11px] font-heading font-bold uppercase tracking-[0.18em] text-[#111] hover:bg-[#111] hover:text-white transition-colors"
          >
            Write another
          </button>
          <Link
            to="/"
            className="h-10 px-5 rounded-full bg-[#111] text-white text-[11px] font-heading font-bold uppercase tracking-[0.18em] hover:bg-[#2a2a28] transition-colors inline-flex items-center"
          >
            Back to studio
          </Link>
        </div>
      </div>
    );
  }

  return (
    <form
      noValidate
      onSubmit={handleSubmit}
      data-testid="contact-form"
      className="bg-white border border-[#E5E5DF] rounded-2xl p-6 sm:p-8 shadow-[0_8px_32px_rgba(17,17,17,0.06)] flex flex-col gap-5"
    >
      <div className="flex items-baseline justify-between">
        <span className="font-heading font-bold text-lg tracking-tight">Send us a message</span>
        <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-[#888884]">all fields required</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <FieldLabel htmlFor="contact-name">name</FieldLabel>
          <input
            id="contact-name"
            data-testid="contact-name"
            type="text"
            autoComplete="name"
            value={values.name}
            onChange={update("name")}
            placeholder="Ada Lovelace"
            aria-invalid={!!errors.name}
            aria-describedby={errors.name ? "contact-name-error" : undefined}
            className={`${inputCls(errors.name)} h-10`}
          />
          <FieldError id="contact-name-error">{errors.name}</FieldError>
        </div>
        <div>
          <FieldLabel htmlFor="contact-email">email</FieldLabel>
          <input
            id="contact-email"
            data-testid="contact-email"
            type="email"
            autoComplete="email"
            value={values.email}
            onChange={update("email")}
            placeholder="ada@university.edu"
            aria-invalid={!!errors.email}
            aria-describedby={errors.email ? "contact-email-error" : undefined}
            className={`${inputCls(errors.email)} h-10`}
          />
          <FieldError id="contact-email-error">{errors.email}</FieldError>
        </div>
      </div>

      <div>
        <FieldLabel htmlFor="contact-topic">topic</FieldLabel>
        <select
          id="contact-topic"
          data-testid="contact-topic"
          value={values.topic}
          onChange={update("topic")}
          className={`${inputCls(false)} h-10`}
        >
          {TOPICS.map((t) => (
            <option key={t.value} value={t.value}>{t.label}</option>
          ))}
        </select>
      </div>

      <div>
        <FieldLabel htmlFor="contact-message" hint={`${values.message.length} / 2000`}>message</FieldLabel>
        <textarea
          id="contact-message"
          data-testid="contact-message"
          rows={6}
          maxLength={2000}
          value={values.message}
          onChange={update("message")}
          placeholder="Tell us what you're working on, what you tried, or what you'd like to know…"
          aria-invalid={!!errors.message}
          aria-describedby={errors.message ? "contact-message-error" : undefined}
          className={`${inputCls(errors.message)} py-2.5 leading-relaxed resize-y min-h-[140px]`}
        />
        <FieldError id="contact-message-error">{errors.message}</FieldError>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
        <p className="text-[11px] font-mono text-[#888884] max-w-xs leading-relaxed">
          Opens your email app with the message pre-filled. Nothing is sent until you press send.
        </p>
        <button
          type="submit"
          data-testid="contact-submit"
          className="h-10 px-5 rounded-full bg-[#111] text-white text-[12px] font-heading font-bold uppercase tracking-[0.14em] hover:bg-[#2a2a28] transition-colors flex items-center gap-2"
        >
          <span>Send message</span>
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
            <path d="M2 6 H10 M7 3 L10 6 L7 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
    </form>
  );
}

export default function ContactPage() {
  return (
    <div className="App app-grain min-h-screen flex flex-col">
      <header
        data-testid="contact-header"
        className="fixed top-0 left-0 right-0 h-16 flex items-center justify-between px-5 sm:px-8 z-30 backdrop-blur-md bg-[#F5F5F0]/75 border-b border-[#E5E5DF]"
      >
        <Link to="/" className="flex items-baseline gap-3">
          <div className="w-6 h-6 bg-[#111] flex items-center justify-center self-center" aria-hidden="true">
            <div className="w-2 h-2 bg-[#10B981]" />
          </div>
          <span className="font-heading font-bold text-[15px] tracking-tight">Watermark Studio</span>
          <span className="hidden sm:inline text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884]">
            / contact
          </span>
        </Link>
        <Link
          to="/"
          data-testid="contact-back-to-studio"
          className="h-9 px-4 rounded-full text-[11px] font-heading font-bold uppercase tracking-[0.18em] border border-[#111] text-[#111] hover:bg-[#111] hover:text-white transition-colors inline-flex items-center"
        >
          Open studio
        </Link>
      </header>

      <main className="flex-1 pt-28 pb-20 px-4 sm:px-6">
        <div className="max-w-5xl mx-auto">
          <section className="max-w-2xl anim-msg-in">
            <div className="inline-flex items-center gap-2 text-[10px] font-mono uppercase tracking-[0.28em] text-[#888884] mb-5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#111]" /> get in touch
            </div>
            <h1 className="font-heading font-bold text-[40px] sm:text-[52px] leading-[1.02] tracking-[-0.02em] text-[#111]">
              Let's talk <em style={{ fontStyle: "italic", fontFamily: "'Cabinet Grotesk', serif" }}>provenance</em>.
            </h1>
            <p className="mt-5 text-[15px] text-[#555550] leading-relaxed">
              Whether you're a researcher testing the method, a team exploring watermarking for your own models,
              or you just spotted something odd in the demo — we read every message.
            </p>
          </section>

          <div className="rule-mono my-10" />

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
            <aside className="lg:col-span-2 flex flex-col gap-3" data-testid="contact-channels">
              {CHANNELS.map((c) => (
                <div key={c.tag} className="bg-white/60 border border-[#E5E5DF] rounded-xl p-5">
                  <span className="block text-[9px] font-mono uppercase tracking-[0.22em] text-[#888884] mb-2">{c.tag}</span>
                  <div className="font-heading font-bold text-[16px] tracking-tight text-[#111]">{c.title}</div>
                  <p className="mt-1.5 text-[13px] leading-relaxed text-[#555550]">{c.body}</p>
                  <div className="mt-3 inline-flex items-center gap-2 px-2.5 py-1 rounded bg-[#F5F5F0] border border-[#E5E5DF]">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#10B981]" />
                    <span className="text-[10px] font-mono text-[#666661]">{c.meta}</span>
                  </div>
                </div>
              ))}

              <div className="bg-[#111] text-white rounded-xl p-5">
                <span className="block text-[9px] font-mono uppercase tracking-[0.22em] text-white/50 mb-2">direct</span>
                <a
                  href={`mailto:${CONTACT_EMAIL}`}
                  data-testid="contact-direct-email"
                  className="font-mono text-[14px] font-semibold hover:text-[#10B981] transition-colors break-all"
                >
                  {CONTACT_EMAIL}
                </a>
                <p className="mt-2 text-[12px] text-white/60 leading-relaxed">
                  Prefer your own email client? Write to us directly — same inbox, same people.
                </p>
              </div>
            </aside>

            <div className="lg:col-span-3">
              <ContactForm />
            </div>
          </div>

          <div className="rule-mono my-12" />

          <section data-testid="contact-faq">
            <div className="text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884] mb-4">Before you write</div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {FAQ.map((f) => (
                <div key={f.q}>
                  <div className="font-heading font-bold text-[15px] leading-snug text-[#111]">{f.q}</div>
                  <p className="mt-2 text-[13px] leading-relaxed text-[#555550]">{f.a}</p>
                </div>
              ))}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}
