import React, { useMemo, useState } from "react";
import { Link } from "react-router-dom";

const FOOD_TYPES = [
  {
    id: "kibble",
    name: "Dry kibble",
    tag: "most common",
    kcal: "340–420 kcal / cup",
    pros: ["Affordable and easy to store", "Long shelf life once opened", "Simple to measure portions"],
    cons: ["Low moisture (around 10%)", "Quality varies widely by brand"],
  },
  {
    id: "wet",
    name: "Wet / canned",
    tag: "high moisture",
    kcal: "85–120 kcal / 100 g",
    pros: ["Very palatable for picky eaters", "Around 75% water, which helps hydration", "Softer texture for senior dogs"],
    cons: ["Costs more per calorie", "Must be refrigerated after opening"],
  },
  {
    id: "fresh",
    name: "Fresh-cooked",
    tag: "subscription",
    kcal: "120–160 kcal / 100 g",
    pros: ["Minimally processed, whole ingredients", "Pre-portioned for your dog"],
    cons: ["Premium price", "Needs freezer and fridge space"],
  },
  {
    id: "raw",
    name: "Raw / freeze-dried",
    tag: "handle with care",
    kcal: "varies by product",
    pros: ["High in protein", "Freeze-dried options are shelf-stable"],
    cons: ["Bacterial risk to pets and people", "Many vets advise against it for puppies"],
  },
];

const LABEL_TIPS = [
  { k: "AAFCO statement", v: "Look for “complete and balanced” for your dog’s life stage. Treat-style or “supplemental” foods are not meant to be a full diet." },
  { k: "First ingredients", v: "Ingredients are listed by weight before cooking. A named protein such as chicken or lamb near the top is a good sign." },
  { k: "Guaranteed analysis", v: "Shows the minimum protein and fat and the maximum fiber and moisture. Compare foods on a dry-matter basis, not straight off the bag." },
  { k: "Calorie content", v: "Printed as kcal/kg and kcal/cup. This is the number that decides how much you should scoop." },
  { k: "Life stage", v: "Puppy, adult, senior or “all life stages”. Large-breed puppies need controlled calcium for healthy joints." },
];

const LIFE_STAGES = [
  { id: "puppy-young", label: "Puppy · under 4 months", factor: 3.0 },
  { id: "puppy-older", label: "Puppy · 4–12 months", factor: 2.0 },
  { id: "adult-neutered", label: "Adult · spayed / neutered", factor: 1.6 },
  { id: "adult-intact", label: "Adult · intact", factor: 1.8 },
  { id: "senior", label: "Senior · less active", factor: 1.4 },
  { id: "weight-loss", label: "Weight loss (vet-guided)", factor: 1.0 },
  { id: "working", label: "Working / very active", factor: 3.0 },
];

const TRANSITION = [
  { days: "Days 1–2", oldPct: 75 },
  { days: "Days 3–4", oldPct: 50 },
  { days: "Days 5–6", oldPct: 25 },
  { days: "Day 7", oldPct: 0 },
];

const FAQ = [
  {
    q: "Is grain-free food healthier?",
    a: "Not for most dogs. True grain allergies are uncommon, and the FDA has looked into a possible link between some grain-free diets and heart disease (DCM). Unless your vet recommends it, grain-inclusive food is a safe default.",
  },
  {
    q: "How many meals a day?",
    a: "Puppies do best on 3–4 small meals a day. Most adults do well on two meals, about 8–12 hours apart. Deep-chested breeds may benefit from smaller, more frequent meals.",
  },
  {
    q: "Which human foods are dangerous?",
    a: "Chocolate, grapes and raisins, onions, garlic, xylitol (found in some peanut butters and gum), macadamia nuts, alcohol and cooked bones. If your dog eats any of these, call your vet or a pet poison line right away.",
  },
  {
    q: "How much should treats count for?",
    a: "Keep treats under about 10% of daily calories. Small pieces of carrot, green beans or plain cooked chicken work well as low-calorie rewards.",
  },
];

function Eyebrow({ children }) {
  return (
    <div className="text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884] mb-3">{children}</div>
  );
}

function PortionCalculator() {
  const [weight, setWeight] = useState(15);
  const [stage, setStage] = useState("adult-neutered");
  const [kcalPerCup, setKcalPerCup] = useState(380);

  const factor = LIFE_STAGES.find((s) => s.id === stage)?.factor ?? 1.6;
  const { rer, daily, cups } = useMemo(() => {
    const r = 70 * Math.pow(weight, 0.75);
    const d = r * factor;
    return { rer: Math.round(r), daily: Math.round(d), cups: d / kcalPerCup };
  }, [weight, factor, kcalPerCup]);

  return (
    <div data-testid="portion-calculator" className="bg-white border border-[#E5E5DF] rounded-2xl p-5 sm:p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="flex flex-col gap-5">
        <div className="flex flex-col gap-1.5">
          <div className="flex items-baseline justify-between">
            <label htmlFor="df-weight" className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#888884]">dog weight</label>
            <span className="text-xs font-mono font-semibold text-[#111]">{weight} kg · {Math.round(weight * 2.205)} lb</span>
          </div>
          <input id="df-weight" type="range" min={1} max={80} step={1} value={weight} onChange={(e) => setWeight(parseInt(e.target.value, 10))} className="w-full accent-[#111]" />
        </div>
        <div className="flex flex-col gap-1.5">
          <label htmlFor="df-stage" className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#888884]">life stage / activity</label>
          <select id="df-stage" value={stage} onChange={(e) => setStage(e.target.value)} className="h-9 bg-[#F5F5F0] border-none rounded-md text-sm px-3 focus:outline-none focus:ring-2 focus:ring-[#111]/20">
            {LIFE_STAGES.map((s) => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
        </div>
        <div className="flex flex-col gap-1.5">
          <div className="flex items-baseline justify-between">
            <label htmlFor="df-kcal" className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#888884]">food energy (from bag)</label>
            <span className="text-xs font-mono font-semibold text-[#111]">{kcalPerCup} kcal / cup</span>
          </div>
          <input id="df-kcal" type="range" min={250} max={550} step={10} value={kcalPerCup} onChange={(e) => setKcalPerCup(parseInt(e.target.value, 10))} className="w-full accent-[#111]" />
        </div>
      </div>

      <div className="bg-[#F5F5F0] border border-[#E5E5DF] rounded-xl p-5 flex flex-col justify-between gap-4">
        <div>
          <div className="text-[10px] font-mono uppercase tracking-[0.2em] text-[#888884]">estimated daily need</div>
          <div className="font-heading font-bold text-[44px] leading-none tracking-[-0.02em] mt-2" data-testid="daily-kcal">
            {daily.toLocaleString()} <span className="text-base font-mono font-medium text-[#666661]">kcal</span>
          </div>
          <div className="mt-3 text-[15px] text-[#2a2a28]">
            ≈ <b>{cups.toFixed(1)} cups</b> per day, or about <b>{(cups / 2).toFixed(1)} cups</b> per meal over two meals.
          </div>
        </div>
        <div className="text-[11px] font-mono text-[#888884] leading-relaxed">
          resting energy {rer} kcal × {factor.toFixed(1)} activity factor. This is a starting point. Adjust by about 10% every two weeks based on your dog’s body condition.
        </div>
      </div>
    </div>
  );
}

export default function DogFood() {
  const [openFaq, setOpenFaq] = useState(0);

  return (
    <div className="App app-grain min-h-screen flex flex-col" data-testid="dog-food-page">
      <header className="fixed top-0 left-0 right-0 h-16 flex items-center justify-between px-5 sm:px-8 z-30 backdrop-blur-md bg-[#F5F5F0]/75 border-b border-[#E5E5DF]">
        <div className="flex items-baseline gap-3">
          <div className="w-6 h-6 bg-[#111] flex items-center justify-center" aria-hidden="true">
            <div className="w-2 h-2 bg-[#10B981]" />
          </div>
          <span className="font-heading font-bold text-[15px] tracking-tight">Watermark Studio</span>
          <span className="hidden sm:inline text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884]">/ dog food guide</span>
        </div>
        <Link to="/" data-testid="nav-home" className="h-9 px-4 rounded-full text-[11px] font-heading font-bold uppercase tracking-[0.18em] border border-[#111] text-[#111] hover:bg-[#111] hover:text-white transition-colors inline-flex items-center">
          ← Back to studio
        </Link>
      </header>

      <main className="flex-1 pt-28 pb-24 px-4 sm:px-6">
        <div className="max-w-4xl mx-auto space-y-16">
          <section className="text-center anim-msg-in">
            <div className="inline-flex items-center gap-2 text-[10px] font-mono uppercase tracking-[0.28em] text-[#888884] mb-5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#10B981]" /> practical guide · updated for 2025
            </div>
            <h1 className="font-heading font-bold text-[40px] sm:text-[56px] leading-[1.02] tracking-[-0.02em] text-[#111]">
              Feeding your dog, <em style={{ fontStyle: "italic" }}>well</em>.
            </h1>
            <p className="mt-5 text-[15px] text-[#555550] leading-relaxed max-w-xl mx-auto">
              How the main types of dog food compare, what the label actually tells you, and how much to put in the bowl each day.
            </p>
          </section>

          <section>
            <Eyebrow>01 · food types</Eyebrow>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {FOOD_TYPES.map((f) => (
                <article key={f.id} data-testid={`food-type-${f.id}`} className="bg-white border border-[#E5E5DF] rounded-2xl p-5 hover:border-[#111] transition-colors">
                  <div className="flex items-baseline justify-between gap-3">
                    <h2 className="font-heading font-bold text-lg">{f.name}</h2>
                    <span className="text-[9px] font-mono uppercase tracking-[0.18em] text-[#888884]">{f.tag}</span>
                  </div>
                  <div className="mt-1 text-[11px] font-mono text-[#666661]">{f.kcal}</div>
                  <div className="rule-mono my-4" />
                  <ul className="space-y-1.5 text-[13px] text-[#2a2a28]">
                    {f.pros.map((p) => (
                      <li key={p} className="flex gap-2"><span className="mt-[7px] w-1.5 h-1.5 rounded-full bg-[#10B981] shrink-0" />{p}</li>
                    ))}
                    {f.cons.map((c) => (
                      <li key={c} className="flex gap-2 text-[#666661]"><span className="mt-[7px] w-1.5 h-1.5 rounded-full bg-[#EF4444] shrink-0" />{c}</li>
                    ))}
                  </ul>
                </article>
              ))}
            </div>
          </section>

          <section>
            <Eyebrow>02 · reading the label</Eyebrow>
            <div className="bg-white border border-[#E5E5DF] rounded-2xl divide-y divide-[#E5E5DF]">
              {LABEL_TIPS.map((t) => (
                <div key={t.k} className="grid grid-cols-1 sm:grid-cols-[200px_1fr] gap-1 sm:gap-6 p-5">
                  <div className="font-heading font-bold text-[15px]">{t.k}</div>
                  <div className="text-[14px] text-[#444440] leading-relaxed">{t.v}</div>
                </div>
              ))}
            </div>
          </section>

          <section>
            <Eyebrow>03 · portion calculator</Eyebrow>
            <PortionCalculator />
          </section>

          <section>
            <Eyebrow>04 · switching foods</Eyebrow>
            <p className="text-[14px] text-[#444440] mb-4 max-w-2xl">
              A sudden switch often causes an upset stomach. Mix the new food in gradually over about a week. Slow down if you notice loose stools.
            </p>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {TRANSITION.map((t) => (
                <div key={t.days} className="bg-white border border-[#E5E5DF] rounded-xl p-4">
                  <div className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#888884]">{t.days}</div>
                  <div className="mt-3 h-2 w-full rounded-full overflow-hidden flex bg-[#E5E5DF]" aria-hidden="true">
                    <div className="bg-[#D0D0C8]" style={{ width: `${t.oldPct}%` }} />
                    <div className="bg-[#10B981]" style={{ width: `${100 - t.oldPct}%` }} />
                  </div>
                  <div className="mt-2 text-[12px] font-mono text-[#444440]">{100 - t.oldPct}% new · {t.oldPct}% old</div>
                </div>
              ))}
            </div>
          </section>

          <section>
            <Eyebrow>05 · common questions</Eyebrow>
            <div className="space-y-2">
              {FAQ.map((f, i) => {
                const open = openFaq === i;
                return (
                  <div key={f.q} className="bg-white border border-[#E5E5DF] rounded-xl">
                    <button
                      data-testid={`faq-${i}`}
                      onClick={() => setOpenFaq(open ? -1 : i)}
                      aria-expanded={open}
                      className="w-full flex items-center justify-between gap-4 px-5 py-4 text-left"
                    >
                      <span className="font-heading font-bold text-[15px]">{f.q}</span>
                      <span className="font-mono text-[#888884] text-lg leading-none">{open ? "−" : "+"}</span>
                    </button>
                    {open && (
                      <div className="px-5 pb-5 text-[14px] text-[#444440] leading-relaxed anim-msg-in">{f.a}</div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          <div className="px-4 py-3 bg-[#FFF8E1] border border-[#E8DCA8] rounded-lg text-[12px] text-[#6B5D2E] text-center leading-relaxed">
            <span className="font-heading font-bold uppercase tracking-wider">General guidance only.</span>{" "}
            Every dog is different. Talk to your veterinarian before changing the diet of a dog with health conditions, allergies or weight concerns.
          </div>
        </div>
      </main>
    </div>
  );
}
