import React, { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Eye, EyeOff, ArrowRight, Info } from "lucide-react";
import { GridViz, MetricPill } from "@/components/WatermarkUI";

const loginSchema = z.object({
  email: z
    .string()
    .trim()
    .min(1, "Enter your email address.")
    .email("That doesn't look like a valid email address."),
  password: z
    .string()
    .min(1, "Enter your password.")
    .min(8, "Passwords are at least 8 characters."),
  remember: z.boolean().optional(),
});

function buildDiamondGrid(size = 9) {
  const mid = Math.floor(size / 2);
  const rows = [];
  for (let r = 0; r < size; r++) {
    const row = [];
    for (let c = 0; c < size; c++) {
      row.push(Math.abs(r - mid) + Math.abs(c - mid) <= mid ? "1" : "0");
    }
    rows.push(row);
  }
  return rows;
}

function FieldError({ id, message }) {
  if (!message) return null;
  return (
    <p id={id} role="alert" className="text-[11px] font-mono text-[#B91C1C] mt-1.5">
      {message}
    </p>
  );
}

export default function LoginPage() {
  const [showPassword, setShowPassword] = useState(false);
  const [submitted, setSubmitted] = useState(null);
  const [forgotOpen, setForgotOpen] = useState(false);
  const grid = useMemo(() => buildDiamondGrid(9), []);

  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
  } = useForm({
    resolver: zodResolver(loginSchema),
    defaultValues: { email: "", password: "", remember: true },
    mode: "onTouched",
  });

  const onSubmit = async (values) => {
    await new Promise((r) => setTimeout(r, 450));
    setSubmitted(values.email);
  };

  const inputBase =
    "h-11 w-full bg-[#F5F5F0] border rounded-lg px-3.5 text-[14px] text-[#111] placeholder:text-[#A8A8A1] focus:outline-none focus:ring-2 focus:ring-[#111]/20 transition-colors";

  return (
    <div className="App app-grain min-h-screen flex flex-col">
      <header className="h-16 flex items-center justify-between px-5 sm:px-8 border-b border-[#E5E5DF] bg-[#F5F5F0]/75 backdrop-blur-md relative z-10">
        <Link to="/" className="flex items-baseline gap-3" data-testid="login-home-link">
          <div className="w-6 h-6 bg-[#111] flex items-center justify-center self-center" aria-hidden="true">
            <div className="w-2 h-2 bg-[#10B981]" />
          </div>
          <span className="font-heading font-bold text-[15px] tracking-tight">Watermark Studio</span>
        </Link>
        <Link
          to="/"
          className="text-[11px] font-mono text-[#666661] hover:text-[#111] transition-colors"
        >
          ← Back to studio
        </Link>
      </header>

      <main className="flex-1 grid grid-cols-1 lg:grid-cols-2 relative z-10">
        {/* Left: brand panel */}
        <section className="hidden lg:flex flex-col justify-between p-12 border-r border-[#E5E5DF]">
          <div className="inline-flex items-center gap-2 text-[10px] font-mono uppercase tracking-[0.28em] text-[#888884]">
            <span className="w-1.5 h-1.5 rounded-full bg-[#111]" /> Proof of origin, made visible
          </div>

          <div className="flex flex-col gap-8 max-w-md">
            <h1 className="font-heading font-bold text-[44px] leading-[1.04] tracking-[-0.02em] text-[#111]">
              Every output carries a <em style={{ fontStyle: "italic" }}>signature</em>.
            </h1>
            <p className="text-[15px] text-[#555550] leading-relaxed">
              Sign in to keep your secret keys, custom patterns and verification history in one place — and pick up exactly where you left off.
            </p>
            <div className="p-5 bg-white border border-[#E5E5DF] rounded-2xl shadow-[0_8px_32px_rgba(17,17,17,0.04)] flex items-center gap-6">
              <GridViz grid={grid} title="recovered pattern" variant="target" ariaLabel="Decorative diamond watermark pattern" />
              <div className="flex flex-col gap-2">
                <MetricPill label="z-score" value="7.84" sub="τ 4.0" />
                <MetricPill label="bit-acc" value="96.3%" sub="vs ~50%" />
                <MetricPill label="grid" value="9×9" sub="81 bits" />
              </div>
            </div>
          </div>

          <div className="text-[11px] font-mono text-[#888884]">
            Based on Bashir 2025 · visual pattern watermarking for LLMs
          </div>
        </section>

        {/* Right: form */}
        <section className="flex items-center justify-center px-5 py-14 sm:px-10">
          <div className="w-full max-w-[400px] anim-msg-in">
            <div className="mb-8">
              <span className="text-[10px] font-mono uppercase tracking-[0.22em] text-[#888884]">Welcome back</span>
              <h2 className="mt-2 font-heading font-bold text-[30px] leading-tight tracking-[-0.01em] text-[#111]">
                Sign in to Watermark Studio
              </h2>
              <p className="mt-2 text-[14px] text-[#666661] leading-relaxed">
                Use the email address linked to your workspace.
              </p>
            </div>

            {submitted ? (
              <div
                data-testid="login-notice"
                className="p-5 bg-white border border-[#E5E5DF] rounded-2xl flex flex-col gap-4 anim-msg-in"
              >
                <div className="flex items-start gap-3">
                  <Info className="w-4 h-4 mt-0.5 text-[#111] shrink-0" />
                  <div>
                    <div className="font-heading font-bold text-[15px] text-[#111]">Accounts are coming soon</div>
                    <p className="mt-1 text-[13px] text-[#555550] leading-relaxed">
                      Thanks, <span className="font-mono text-[12px] text-[#111]">{submitted}</span>. Sign-in isn't connected in this demo yet — the studio works fully without an account in the meantime.
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <Link
                    to="/"
                    className="h-10 px-5 rounded-full bg-[#111] text-white text-[12px] font-heading font-bold uppercase tracking-[0.14em] hover:bg-[#2a2a28] transition-colors inline-flex items-center gap-2"
                  >
                    Open the studio <ArrowRight className="w-3.5 h-3.5" />
                  </Link>
                  <button
                    type="button"
                    onClick={() => setSubmitted(null)}
                    className="text-[11px] font-mono text-[#888884] underline hover:text-[#111]"
                  >
                    use a different email
                  </button>
                </div>
              </div>
            ) : (
              <form
                onSubmit={handleSubmit(onSubmit)}
                noValidate
                data-testid="login-form"
                className="bg-white border border-[#E5E5DF] rounded-2xl shadow-[0_8px_32px_rgba(17,17,17,0.06)] p-6 flex flex-col gap-5"
              >
                <div>
                  <label htmlFor="login-email" className="block text-[10px] font-mono uppercase tracking-[0.18em] text-[#888884] mb-2">
                    Email
                  </label>
                  <input
                    id="login-email"
                    type="email"
                    autoComplete="email"
                    placeholder="you@lab.org"
                    data-testid="login-email"
                    aria-invalid={!!errors.email}
                    aria-describedby={errors.email ? "login-email-error" : undefined}
                    className={`${inputBase} ${errors.email ? "border-[#EF4444]/50" : "border-transparent"}`}
                    {...register("email")}
                  />
                  <FieldError id="login-email-error" message={errors.email?.message} />
                </div>

                <div>
                  <div className="flex items-baseline justify-between mb-2">
                    <label htmlFor="login-password" className="text-[10px] font-mono uppercase tracking-[0.18em] text-[#888884]">
                      Password
                    </label>
                    <button
                      type="button"
                      onClick={() => setForgotOpen((v) => !v)}
                      className="text-[11px] font-mono text-[#666661] hover:text-[#111] underline-offset-2 hover:underline"
                    >
                      Forgot password?
                    </button>
                  </div>
                  <div className="relative">
                    <input
                      id="login-password"
                      type={showPassword ? "text" : "password"}
                      autoComplete="current-password"
                      placeholder="At least 8 characters"
                      data-testid="login-password"
                      aria-invalid={!!errors.password}
                      aria-describedby={errors.password ? "login-password-error" : undefined}
                      className={`${inputBase} pr-11 ${errors.password ? "border-[#EF4444]/50" : "border-transparent"}`}
                      {...register("password")}
                    />
                    <button
                      type="button"
                      onClick={() => setShowPassword((v) => !v)}
                      aria-label={showPassword ? "Hide password" : "Show password"}
                      className="absolute right-1.5 top-1/2 -translate-y-1/2 h-8 w-8 rounded-md flex items-center justify-center text-[#666661] hover:text-[#111] hover:bg-[#E5E5DF]/60 transition-colors"
                    >
                      {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                  <FieldError id="login-password-error" message={errors.password?.message} />
                  {forgotOpen && (
                    <p className="mt-2 px-3 py-2 bg-[#F5F5F0] border border-[#E5E5DF] rounded-lg text-[12px] text-[#555550] leading-relaxed anim-msg-in">
                      Password resets are handled by your workspace admin. Ask them to send you a new invite link.
                    </p>
                  )}
                </div>

                <label className="flex items-center gap-2.5 text-[13px] text-[#444440] cursor-pointer select-none">
                  <input type="checkbox" className="w-4 h-4 accent-[#111]" {...register("remember")} />
                  Keep me signed in on this device
                </label>

                <button
                  type="submit"
                  disabled={isSubmitting}
                  data-testid="login-submit"
                  className="h-11 w-full rounded-full bg-[#111] text-white text-[12px] font-heading font-bold uppercase tracking-[0.14em] hover:bg-[#2a2a28] disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
                >
                  {isSubmitting ? (
                    <>
                      <span className="w-1.5 h-1.5 rounded-full bg-white pulse-dot" />
                      <span>Signing in</span>
                    </>
                  ) : (
                    <>
                      <span>Sign in</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </>
                  )}
                </button>

                <div className="rule-mono" />

                <p className="text-[12px] text-[#666661] text-center leading-relaxed">
                  New here?{" "}
                  <Link to="/" className="text-[#111] font-semibold hover:underline underline-offset-2">
                    Try the studio without an account
                  </Link>
                </p>
              </form>
            )}

            <p className="mt-6 text-[11px] font-mono text-[#888884] text-center leading-relaxed">
              Your secret keys never leave your workspace.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}
