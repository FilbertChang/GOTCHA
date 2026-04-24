import { useState, useEffect } from "react";
import axios from "axios";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, PieChart, Pie, Cell
} from "recharts";
import {
  Shield, AlertTriangle, CheckCircle, Activity,
  TrendingUp, Sun, Moon, Eye, Zap
} from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const DARK = {
  bg: "#0B1A2F", surface: "#162A45", card: "#1A3050", border: "#1E3D5C",
  accent: "#00C2FF", danger: "#FF3B3B", warn: "#F59E0B", safe: "#16A34A",
  text: "#FFFFFF", muted: "#B0C0D0", inputBg: "#0B1A2F", shadow: "rgba(0,194,255,0.08)",
};
const LIGHT = {
  bg: "#F8FAFC", surface: "#FFFFFF", card: "#FFFFFF", border: "#E2E8F0",
  accent: "#1E3A8A", danger: "#DC2626", warn: "#F59E0B", safe: "#16A34A",
  text: "#0F172A", muted: "#64748B", inputBg: "#F1F5F9", shadow: "rgba(30,58,138,0.08)",
};

const FRAUD_TYPES = [
  { key: "social_engineering", label: "Social Engineering", desc: "Penipuan via WA/Telegram", color: "#FF3B3B" },
  { key: "rekening_mule", label: "Rekening Mule", desc: "Rekening penampung ilegal", color: "#F59E0B" },
  { key: "qris_fraud_substitusi", label: "QRIS Substitusi", desc: "QR code palsu ditempel", color: "#A855F7" },
  { key: "qris_fraud_merchant_fiktif", label: "QRIS Merchant Fiktif", desc: "Merchant QRIS palsu", color: "#EC4899" },
  { key: "pinjol_ilegal", label: "Pinjol Ilegal", desc: "Platform pinjaman ilegal", color: "#F97316" },
];

const FRAUD_LABEL_MAP = Object.fromEntries(FRAUD_TYPES.map(f => [f.key, f.label]));

const SCENARIOS = [
  { label: "Social Engineering", fraud_key: "social_engineering", data: { transaction_id: "TXN-SE-001", platform_type: "e_wallet", transaction_type: "transfer", amount_idr: 24500000, interbank_transfer: true, merchant_category: null, is_merchant_blacklisted: false, sender_account_age_days: 365, device_changed_recently: true, sender_os: "Android", sender_province: "Jawa Barat", receiver_type: "personal", receiver_account_age_days: 1, receiver_id_match_blacklist: false, trx_count_last_1h: 2, trx_count_last_24h: 3, amount_vs_avg_ratio: 12.5, hour_of_day: 22, is_outside_normal_hours: true, time_since_last_trx_minutes: 3.2, is_emulator: false, amount_roundness: 0.0 } },
  { label: "Rekening Mule", fraud_key: "rekening_mule", data: { transaction_id: "TXN-RM-001", platform_type: "mobile_banking", transaction_type: "transfer", amount_idr: 18000000, interbank_transfer: true, merchant_category: null, is_merchant_blacklisted: false, sender_account_age_days: 200, device_changed_recently: false, sender_os: "Android", sender_province: "DKI Jakarta", receiver_type: "personal", receiver_account_age_days: 2, receiver_id_match_blacklist: true, trx_count_last_1h: 8, trx_count_last_24h: 25, amount_vs_avg_ratio: 9.3, hour_of_day: 14, is_outside_normal_hours: false, time_since_last_trx_minutes: 1.5, is_emulator: true, amount_roundness: 0.0 } },
  { label: "QRIS Fraud", fraud_key: "qris_fraud_substitusi", data: { transaction_id: "TXN-QR-001", platform_type: "e_wallet", transaction_type: "qris_payment", amount_idr: 150000, interbank_transfer: false, merchant_category: "F&B", is_merchant_blacklisted: false, sender_account_age_days: 500, device_changed_recently: false, sender_os: "iOS", sender_province: "Jawa Timur", receiver_type: "merchant_qris", receiver_account_age_days: 3, receiver_id_match_blacklist: true, trx_count_last_1h: 85, trx_count_last_24h: 150, amount_vs_avg_ratio: 1.2, hour_of_day: 11, is_outside_normal_hours: false, time_since_last_trx_minutes: 0.8, is_emulator: false, amount_roundness: 0.5 } },
  { label: "Pinjol Ilegal", fraud_key: "pinjol_ilegal", data: { transaction_id: "TXN-PJ-001", platform_type: "e_wallet", transaction_type: "pembayaran_tagihan", amount_idr: 150000, interbank_transfer: true, merchant_category: "game_topup", is_merchant_blacklisted: true, sender_account_age_days: 180, device_changed_recently: false, sender_os: "Android", sender_province: "Jawa Timur", receiver_type: "merchant_online", receiver_account_age_days: 90, receiver_id_match_blacklist: true, trx_count_last_1h: 8, trx_count_last_24h: 22, amount_vs_avg_ratio: 1.1, hour_of_day: 2, is_outside_normal_hours: true, time_since_last_trx_minutes: 4.5, is_emulator: false, amount_roundness: 0.5 } },
  { label: "Transaksi Normal", fraud_key: null, data: { transaction_id: "TXN-OK-001", platform_type: "e_wallet", transaction_type: "qris_payment", amount_idr: 45000, interbank_transfer: false, merchant_category: "F&B", is_merchant_blacklisted: false, sender_account_age_days: 720, device_changed_recently: false, sender_os: "Android", sender_province: "Jawa Barat", receiver_type: "merchant_qris", receiver_account_age_days: 365, receiver_id_match_blacklist: false, trx_count_last_1h: 1, trx_count_last_24h: 4, amount_vs_avg_ratio: 0.9, hour_of_day: 12, is_outside_normal_hours: false, time_since_last_trx_minutes: 45.0, is_emulator: false, amount_roundness: 0.75 } },
];

function RiskMeter({ score, T }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.8 ? T.danger : score >= 0.5 ? T.warn : T.safe;
  const label = score >= 0.8 ? "TINGGI" : score >= 0.5 ? "SEDANG" : "RENDAH";
  const r = 48, circ = 2 * Math.PI * r;
  const dash = circ - (pct / 100) * circ;
  return (
    <div style={{ position: "relative", width: 116, height: 116, flexShrink: 0 }}>
      <svg width={116} height={116} style={{ transform: "rotate(-90deg)" }}>
        <circle cx={58} cy={58} r={r} fill="none" stroke={T.border} strokeWidth={9} />
        <circle cx={58} cy={58} r={r} fill="none" stroke={color} strokeWidth={9}
          strokeDasharray={circ} strokeDashoffset={dash} strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 0.8s ease, stroke 0.3s ease" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
        <span style={{ fontSize: 22, fontWeight: 700, color, lineHeight: 1 }}>{pct}%</span>
        <span style={{ fontSize: 9, color: T.muted, letterSpacing: 2, marginTop: 2 }}>{label}</span>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color, sub, T }) {
  return (
    <div style={{ background: T.card, border: `1px solid ${T.border}`, borderTop: `3px solid ${color}`, borderRadius: 12, padding: "18px 20px", boxShadow: `0 4px 16px ${T.shadow}` }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
        <Icon size={14} color={color} />
        <span style={{ fontSize: 10, color: T.muted, letterSpacing: 1, fontWeight: 500 }}>{label}</span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color: T.text, lineHeight: 1 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: T.muted, marginTop: 6 }}>{sub}</div>}
    </div>
  );
}

function FraudTypePanel({ detectedKey, T }) {
  const hasResult = detectedKey !== undefined;
  return (
    <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 12, padding: "18px 20px", boxShadow: `0 4px 16px ${T.shadow}` }}>
      <div style={{ fontSize: 10, color: T.muted, letterSpacing: 1, fontWeight: 500, marginBottom: 14 }}>
        JENIS FRAUD TERDETEKSI
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
        {FRAUD_TYPES.map(ft => {
          const active = hasResult && detectedKey === ft.key;
          return (
            <div key={ft.key} style={{
              display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderRadius: 8,
              background: active ? `${ft.color}15` : T.surface,
              border: `1px solid ${active ? ft.color + "55" : T.border}`,
              transition: "all 0.3s ease",
              boxShadow: active ? `0 0 14px ${ft.color}20` : "none",
            }}>
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: active ? ft.color : T.border, boxShadow: active ? `0 0 7px ${ft.color}` : "none", flexShrink: 0, transition: "all 0.3s" }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 12, fontWeight: active ? 600 : 400, color: active ? ft.color : T.muted, transition: "color 0.3s" }}>{ft.label}</div>
                <div style={{ fontSize: 10, color: T.muted, marginTop: 1 }}>{ft.desc}</div>
              </div>
              {active && (
                <div style={{ fontSize: 9, padding: "2px 7px", borderRadius: 20, background: `${ft.color}20`, color: ft.color, border: `1px solid ${ft.color}44`, fontWeight: 600, letterSpacing: 1, flexShrink: 0 }}>
                  AKTIF
                </div>
              )}
            </div>
          );
        })}
        {/* Normal row */}
        {(() => {
          const active = hasResult && detectedKey === null;
          return (
            <div style={{
              display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderRadius: 8,
              background: active ? `${T.safe}15` : T.surface,
              border: `1px solid ${active ? T.safe + "55" : T.border}`,
              transition: "all 0.3s ease",
            }}>
              <div style={{ width: 7, height: 7, borderRadius: "50%", background: active ? T.safe : T.border, flexShrink: 0, transition: "all 0.3s" }} />
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12, fontWeight: active ? 600 : 400, color: active ? T.safe : T.muted, transition: "color 0.3s" }}>Transaksi Normal</div>
                <div style={{ fontSize: 10, color: T.muted, marginTop: 1 }}>Tidak ada indikasi fraud</div>
              </div>
              {active && (
                <div style={{ fontSize: 9, padding: "2px 7px", borderRadius: 20, background: `${T.safe}20`, color: T.safe, border: `1px solid ${T.safe}44`, fontWeight: 600, letterSpacing: 1, flexShrink: 0 }}>
                  AKTIF
                </div>
              )}
            </div>
          );
        })()}
      </div>
    </div>
  );
}

export default function App() {
  const [dark, setDark] = useState(true);
  const T = dark ? DARK : LIGHT;
  const [result, setResult] = useState(null);
  const [detectedKey, setDetectedKey] = useState(undefined);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [backendOk, setBackendOk] = useState(null);
  const [customInput, setCustomInput] = useState("");
  const [customError, setCustomError] = useState("");
  const [globalError, setGlobalError] = useState("");

  useEffect(() => {
    axios.get(`${API}/health`).then(() => setBackendOk(true)).catch(() => setBackendOk(false));
  }, []);

  const chartData = history.slice(-20).map((h, i) => ({ name: i + 1, risk: Math.round(h.risk_score * 100) }));
  const totalFraud = history.filter(h => h.is_fraud).length;
  const totalNormal = history.filter(h => !h.is_fraud).length;
  const avgRisk = history.length ? Math.round(history.reduce((a, b) => a + b.risk_score, 0) / history.length * 100) : 0;
  const pieData = [{ name: "Fraud", value: totalFraud || 1 }, { name: "Normal", value: totalNormal || 1 }];

  async function runPredict(data) {
    setLoading(true);
    setCustomError("");
    try {
      const res = await axios.post(`${API}/predict`, data);
      const r = { ...res.data, timestamp: new Date().toLocaleTimeString("id-ID") };
      setResult(r);
      setDetectedKey(r.is_fraud ? (r.fraud_type_predicted ?? null) : null);
      setHistory(prev => [...prev.slice(-49), r]);
    } catch {
      setCustomError("Gagal menghubungi backend. Pastikan server berjalan.");
      setGlobalError("Gagal menghubungi backend. Pastikan server berjalan.");
    } finally {
      setLoading(false);
    }
  }

  function handleCustomSubmit() {
    try { runPredict(JSON.parse(customInput)); }
    catch { setCustomError("JSON tidak valid. Periksa format input."); }
  }

  const actionColor = result
    ? result.recommended_action === "BLOCK" ? T.danger
      : result.recommended_action === "REVIEW" ? T.warn : T.safe
    : T.muted;

  const navBtn = (tab) => (
    <button key={tab} onClick={() => setActiveTab(tab)} style={{
      background: activeTab === tab ? `${T.accent}18` : "none",
      border: `1px solid ${activeTab === tab ? T.accent + "55" : "transparent"}`,
      borderRadius: 8, cursor: "pointer",
      color: activeTab === tab ? T.accent : T.muted,
      fontSize: 11, letterSpacing: 1, textTransform: "uppercase",
      padding: "6px 14px", fontWeight: activeTab === tab ? 600 : 400,
      transition: "all 0.2s",
    }}>{tab}</button>
  );

  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.text, fontFamily: "'Inter', -apple-system, sans-serif", transition: "background 0.3s, color 0.3s" }}>

      {/* Header */}
      <header style={{ background: T.surface, borderBottom: `1px solid ${T.border}`, padding: "0 32px", display: "flex", alignItems: "center", justifyContent: "space-between", height: 60, position: "sticky", top: 0, zIndex: 100, boxShadow: `0 2px 12px ${T.shadow}` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 34, height: 34, borderRadius: 8, background: `linear-gradient(135deg, ${T.accent}, ${dark ? "#0088cc" : "#1E3A8A"})`, display: "flex", alignItems: "center", justifyContent: "center", boxShadow: `0 0 12px ${T.accent}44` }}>
            <Shield size={18} color="#fff" />
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 700, color: T.accent, letterSpacing: 2 }}>G.O.T.C.H.A</div>
            <div style={{ fontSize: 9, color: T.muted, letterSpacing: 0.5 }}>Guard & Observe Transactions with Cognitive Hybrid AI</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {["dashboard", "simulate", "history"].map(navBtn)}
          <button onClick={() => setDark(d => !d)} style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 8, cursor: "pointer", padding: "6px 10px", display: "flex", alignItems: "center", gap: 6, color: T.muted, fontSize: 11, marginLeft: 8 }}>
            {dark ? <Sun size={13} /> : <Moon size={13} />} {dark ? "Light" : "Dark"}
          </button>
          <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "6px 12px", borderRadius: 8, background: T.card, border: `1px solid ${T.border}` }}>
            <div style={{ width: 7, height: 7, borderRadius: "50%", background: backendOk === null ? T.warn : backendOk ? T.safe : T.danger, boxShadow: backendOk ? `0 0 6px ${T.safe}` : "none" }} />
            <span style={{ fontSize: 11, color: T.muted }}>{backendOk === null ? "Connecting..." : backendOk ? "API Online" : "API Offline"}</span>
          </div>
        </div>
      </header>

      <main style={{ padding: "28px 32px", maxWidth: 1400, margin: "0 auto" }}>
        {globalError && (
          <div style={{ background: `${T.danger}15`, border: `1px solid ${T.danger}44`, borderRadius: 8, padding: "10px 16px", marginBottom: 16, fontSize: 12, color: T.danger, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span>⚠ {globalError}</span>
            <button onClick={() => setGlobalError("")} style={{ background: "none", border: "none", color: T.danger, cursor: "pointer", fontSize: 16, lineHeight: 1 }}>×</button>
          </div>
        )}

        {/* DASHBOARD */}
        {activeTab === "dashboard" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 }}>
              <StatCard icon={Activity} label="TOTAL TRANSAKSI" value={history.length} color={T.accent} sub="sesi ini" T={T} />
              <StatCard icon={AlertTriangle} label="FRAUD TERDETEKSI" value={totalFraud} color={T.danger} sub={`${history.length ? Math.round(totalFraud / history.length * 100) : 0}% dari total`} T={T} />
              <StatCard icon={CheckCircle} label="TRANSAKSI AMAN" value={totalNormal} color={T.safe} sub="lolos verifikasi" T={T} />
              <StatCard icon={TrendingUp} label="AVG RISK SCORE" value={`${avgRisk}%`} color={T.warn} sub="rata-rata sesi ini" T={T} />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 300px", gap: 16 }}>
              <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 12, padding: 20, boxShadow: `0 4px 16px ${T.shadow}` }}>
                <div style={{ fontSize: 10, color: T.muted, letterSpacing: 1, fontWeight: 500, marginBottom: 16 }}>RISK SCORE TREND — 20 TRANSAKSI TERAKHIR</div>
                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={200}>
                    <AreaChart data={chartData}>
                      <defs><linearGradient id="rg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={T.accent} stopOpacity={0.25} /><stop offset="95%" stopColor={T.accent} stopOpacity={0} /></linearGradient></defs>
                      <XAxis dataKey="name" stroke={T.muted} tick={{ fontSize: 10, fill: T.muted }} />
                      <YAxis domain={[0, 100]} stroke={T.muted} tick={{ fontSize: 10, fill: T.muted }} />
                      <Tooltip contentStyle={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 8, fontSize: 12, color: T.text }} formatter={v => [`${v}%`, "Risk"]} />
                      <Area type="monotone" dataKey="risk" stroke={T.accent} fill="url(#rg)" strokeWidth={2} dot={{ fill: T.accent, r: 3 }} />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ height: 200, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, color: T.muted }}>
                    <Eye size={32} color={T.border} />
                    <span style={{ fontSize: 13 }}>Belum ada data — jalankan simulasi dulu</span>
                  </div>
                )}
              </div>
              <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 12, padding: 20, boxShadow: `0 4px 16px ${T.shadow}` }}>
                <div style={{ fontSize: 10, color: T.muted, letterSpacing: 1, fontWeight: 500, marginBottom: 12 }}>DISTRIBUSI HASIL</div>
                {history.length === 0 ? (
                  <div style={{ height: 140, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 6, color: T.muted }}>
                    <Eye size={24} color={T.border} />
                    <span style={{ fontSize: 12 }}>Belum ada data</span>
                  </div>
                ) : (
                  <>
                    <ResponsiveContainer width="100%" height={140}>
                      <PieChart><Pie data={pieData} cx="50%" cy="50%" innerRadius={38} outerRadius={60} dataKey="value" strokeWidth={0}><Cell fill={T.danger} /><Cell fill={T.safe} /></Pie><Tooltip contentStyle={{ background: T.surface, border: `1px solid ${T.border}`, fontSize: 12, color: T.text }} /></PieChart>
                    </ResponsiveContainer>
                    <div style={{ display: "flex", justifyContent: "center", gap: 16, marginTop: 8 }}>
                      {[["Fraud", T.danger], ["Normal", T.safe]].map(([l, c]) => (
                        <div key={l} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: T.muted }}><div style={{ width: 8, height: 8, borderRadius: 2, background: c }} />{l}</div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
            {history.length > 0 && (
              <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 12, padding: 20, boxShadow: `0 4px 16px ${T.shadow}` }}>
                <div style={{ fontSize: 10, color: T.muted, letterSpacing: 1, fontWeight: 500, marginBottom: 14 }}>RIWAYAT TERKINI</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  {[...history].reverse().slice(0, 5).map((h, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 14px", borderRadius: 8, background: T.surface, border: `1px solid ${T.border}` }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                        <div style={{ width: 6, height: 6, borderRadius: "50%", background: h.is_fraud ? T.danger : T.safe }} />
                        <span style={{ fontSize: 12, color: T.text }}>{h.transaction_id}</span>
                        {h.fraud_type_predicted && <span style={{ fontSize: 11, color: T.muted }}>— {FRAUD_LABEL_MAP[h.fraud_type_predicted] || h.fraud_type_predicted}</span>}
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                        <span style={{ fontSize: 11, color: T.muted }}>{h.timestamp}</span>
                        <span style={{ fontSize: 13, fontWeight: 600, color: h.is_fraud ? T.danger : T.safe }}>{Math.round(h.risk_score * 100)}%</span>
                        <span style={{ fontSize: 11, padding: "2px 10px", borderRadius: 20, fontWeight: 500, background: h.recommended_action === "BLOCK" ? `${T.danger}18` : h.recommended_action === "REVIEW" ? `${T.warn}18` : `${T.safe}18`, color: h.recommended_action === "BLOCK" ? T.danger : h.recommended_action === "REVIEW" ? T.warn : T.safe, border: `1px solid ${h.recommended_action === "BLOCK" ? T.danger + "44" : h.recommended_action === "REVIEW" ? T.warn + "44" : T.safe + "44"}` }}>{h.recommended_action}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* SIMULATE */}
        {activeTab === "simulate" && (
          <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 24 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <FraudTypePanel detectedKey={detectedKey} T={T} />
              <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 12, padding: "16px 20px", boxShadow: `0 4px 16px ${T.shadow}` }}>
                <div style={{ fontSize: 10, color: T.muted, letterSpacing: 1, fontWeight: 500, marginBottom: 12 }}>QUICK TEST</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {SCENARIOS.map(s => (
                    <button key={s.label} onClick={() => runPredict(s.data)} disabled={loading} style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 8, padding: "7px 12px", cursor: loading ? "not-allowed" : "pointer", fontSize: 11, color: T.muted, transition: "all 0.2s", opacity: loading ? 0.5 : 1 }}
                      onMouseEnter={e => { if (!loading) { e.currentTarget.style.borderColor = T.accent; e.currentTarget.style.color = T.accent; } }}
                      onMouseLeave={e => { e.currentTarget.style.borderColor = T.border; e.currentTarget.style.color = T.muted; }}>
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>
              <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 12, padding: "16px 20px", boxShadow: `0 4px 16px ${T.shadow}` }}>
                <div style={{ fontSize: 10, color: T.muted, letterSpacing: 1, fontWeight: 500, marginBottom: 12 }}>CUSTOM INPUT (JSON)</div>
                <textarea value={customInput} onChange={e => setCustomInput(e.target.value)} placeholder='{"transaction_id": "TXN-001", ...}'
                  style={{ width: "100%", height: 160, background: T.inputBg, border: `1px solid ${T.border}`, borderRadius: 8, color: T.text, fontSize: 11, padding: 12, resize: "none", fontFamily: "monospace", outline: "none" }}
                  onFocus={e => e.target.style.borderColor = T.accent}
                  onBlur={e => e.target.style.borderColor = T.border} />
                {customError && <div style={{ fontSize: 11, color: T.danger, marginTop: 6 }}>{customError}</div>}
                <button onClick={handleCustomSubmit} disabled={loading || !customInput.trim()} style={{ marginTop: 10, width: "100%", padding: "11px", background: loading || !customInput.trim() ? T.border : T.accent, border: "none", borderRadius: 8, cursor: loading || !customInput.trim() ? "not-allowed" : "pointer", color: dark ? "#0B1A2F" : "#fff", fontSize: 12, fontWeight: 700, letterSpacing: 1, transition: "all 0.2s", boxShadow: !loading && customInput.trim() ? `0 0 16px ${T.accent}44` : "none" }}>
                  {loading ? "MENGANALISIS..." : "ANALISIS TRANSAKSI"}
                </button>
              </div>
            </div>

            <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 12, padding: 28, boxShadow: `0 4px 16px ${T.shadow}`, minHeight: 500 }}>
              {!result && !loading && (
                <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, color: T.muted }}>
                  <Zap size={44} color={T.border} />
                  <span style={{ fontSize: 14 }}>Pilih quick test atau masukkan JSON</span>
                </div>
              )}
              {loading && (
                <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16 }}>
                  <div style={{ width: 44, height: 44, border: `3px solid ${T.border}`, borderTop: `3px solid ${T.accent}`, borderRadius: "50%", animation: "spin 0.8s linear infinite", boxShadow: `0 0 12px ${T.accent}44` }} />
                  <span style={{ fontSize: 11, color: T.muted, letterSpacing: 2 }}>MENGANALISIS TRANSAKSI...</span>
                </div>
              )}
              {result && !loading && (
                <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
                    <RiskMeter score={result.risk_score} T={T} />
                    <div>
                      <div style={{ fontSize: 32, fontWeight: 800, color: actionColor, letterSpacing: 2, lineHeight: 1 }}>{result.recommended_action}</div>
                      <div style={{ fontSize: 12, color: T.muted, marginTop: 4, marginBottom: 12 }}>{result.transaction_id}</div>
                      {result.fraud_type_predicted && (
                        <div style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, padding: "5px 12px", borderRadius: 20, background: `${T.danger}15`, border: `1px solid ${T.danger}44`, color: T.danger, fontWeight: 500 }}>
                          <div style={{ width: 6, height: 6, borderRadius: "50%", background: T.danger, boxShadow: `0 0 5px ${T.danger}` }} />
                          {FRAUD_LABEL_MAP[result.fraud_type_predicted] || result.fraud_type_predicted}
                        </div>
                      )}
                      {!result.is_fraud && (
                        <div style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, padding: "5px 12px", borderRadius: 20, background: `${T.safe}15`, border: `1px solid ${T.safe}44`, color: T.safe, fontWeight: 500 }}>
                          <div style={{ width: 6, height: 6, borderRadius: "50%", background: T.safe }} /> Transaksi Normal
                        </div>
                      )}
                    </div>
                  </div>
                  <div style={{ height: 1, background: T.border }} />
                  <div style={{ background: T.surface, borderRadius: 10, padding: 16, border: `1px solid ${T.border}`, borderLeft: `3px solid ${T.accent}` }}>
                    <div style={{ fontSize: 10, color: T.accent, letterSpacing: 2, fontWeight: 600, marginBottom: 10 }}>AI EXPLANATION — GPT-4o</div>
                    <div style={{ fontSize: 13, color: T.text, lineHeight: 1.75 }}>{result.explanation}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: T.muted, letterSpacing: 2, fontWeight: 500, marginBottom: 10 }}>SINYAL TERDETEKSI</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {Object.entries(result.signals).map(([k, v]) => (
                        <div key={k} style={{ display: "inline-flex", alignItems: "center", gap: 5, padding: "3px 10px", borderRadius: 20, fontSize: 11, background: v ? `${T.danger}15` : `${T.muted}10`, border: `1px solid ${v ? T.danger + "44" : T.border}`, color: v ? T.danger : T.muted, fontWeight: v ? 500 : 400 }}>
                          <div style={{ width: 5, height: 5, borderRadius: "50%", background: v ? T.danger : T.muted, boxShadow: v ? `0 0 4px ${T.danger}` : "none" }} />
                          {k.replace(/_/g, " ")}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* HISTORY */}
        {activeTab === "history" && (
          <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 12, padding: 24, boxShadow: `0 4px 16px ${T.shadow}` }}>
            <div style={{ fontSize: 10, color: T.muted, letterSpacing: 1, fontWeight: 500, marginBottom: 16 }}>RIWAYAT LENGKAP — {history.length} TRANSAKSI</div>
            {history.length === 0 ? (
              <div style={{ textAlign: "center", color: T.muted, padding: 48, fontSize: 13 }}>Belum ada riwayat transaksi</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1.5fr 80px 100px", padding: "6px 14px", gap: 12, fontSize: 10, color: T.muted, letterSpacing: 1, fontWeight: 500 }}>
                  <span>TRANSACTION ID</span><span>WAKTU</span><span>JENIS</span><span>RISK</span><span>ACTION</span>
                </div>
                {[...history].reverse().map((h, i) => (
                  <div key={i} style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1.5fr 80px 100px", alignItems: "center", padding: "11px 14px", borderRadius: 8, background: T.surface, border: `1px solid ${T.border}`, gap: 12 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{ width: 6, height: 6, borderRadius: "50%", background: h.is_fraud ? T.danger : T.safe, flexShrink: 0 }} />
                      <span style={{ fontSize: 12, color: T.text }}>{h.transaction_id}</span>
                    </div>
                    <span style={{ fontSize: 12, color: T.muted }}>{h.timestamp}</span>
                    <span style={{ fontSize: 12, color: h.is_fraud ? T.danger : T.safe }}>{h.fraud_type_predicted ? (FRAUD_LABEL_MAP[h.fraud_type_predicted] || h.fraud_type_predicted) : "Normal"}</span>
                    <span style={{ fontSize: 13, fontWeight: 700, color: h.is_fraud ? T.danger : T.safe }}>{Math.round(h.risk_score * 100)}%</span>
                    <span style={{ fontSize: 11, padding: "3px 10px", borderRadius: 20, textAlign: "center", fontWeight: 500, background: h.recommended_action === "BLOCK" ? `${T.danger}15` : h.recommended_action === "REVIEW" ? `${T.warn}15` : `${T.safe}15`, color: h.recommended_action === "BLOCK" ? T.danger : h.recommended_action === "REVIEW" ? T.warn : T.safe, border: `1px solid ${h.recommended_action === "BLOCK" ? T.danger + "44" : h.recommended_action === "REVIEW" ? T.warn + "44" : T.safe + "44"}` }}>{h.recommended_action}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
      `}</style>
    </div>
  );
}
