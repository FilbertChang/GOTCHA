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
  bg: "#191c1f", surface: "#2a2d30", card: "#191c1f", border: "#3a4148",
  accent: "#494fdf", danger: "#e23b4a", warn: "#ec7e00", safe: "#00a87e",
  text: "#ffffff", muted: "#8d969e", inputBg: "#191c1f", shadow: "transparent",
  pillBg: "#ffffff", pillText: "#191c1f", pillHover: "rgba(255,255,255,0.85)"
};
const LIGHT = {
  bg: "#ffffff", surface: "#f4f4f4", card: "#ffffff", border: "#c9c9cd",
  accent: "#494fdf", danger: "#e23b4a", warn: "#ec7e00", safe: "#00a87e",
  text: "#191c1f", muted: "#505a63", inputBg: "#ffffff", shadow: "transparent",
  pillBg: "#191c1f", pillText: "#ffffff", pillHover: "rgba(25,28,31,0.85)"
};

const FRAUD_TYPES = [
  { key: "social_engineering", label: "Social Engineering", desc: "Penipuan via WA/Telegram", color: "#e23b4a" },
  { key: "rekening_mule", label: "Rekening Mule", desc: "Rekening penampung ilegal", color: "#ec7e00" },
  { key: "qris_fraud_substitusi", label: "QRIS Substitusi", desc: "QR code palsu ditempel", color: "#e61e49" },
  { key: "qris_fraud_merchant_fiktif", label: "QRIS Merchant Fiktif", desc: "Merchant QRIS palsu", color: "#494fdf" },
  { key: "pinjol_ilegal", label: "Pinjol Ilegal", desc: "Platform pinjaman ilegal", color: "#b09000" },
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
        <span style={{ fontFamily: '"Aeonik Pro", sans-serif', fontSize: 32, fontWeight: 500, color, lineHeight: 1, letterSpacing: "-0.32px" }}>{pct}%</span>
        <span style={{ fontSize: 10, color: T.muted, letterSpacing: "0.24px", marginTop: 4, textTransform: "uppercase" }}>{label}</span>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color, sub, T }) {
  return (
    <div style={{ background: T.card, border: `1px solid ${T.border}`, borderTop: `3px solid ${color}`, borderRadius: 20, padding: "24px 32px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16 }}>
        <Icon size={16} color={color} />
        <span style={{ fontSize: 12, color: T.muted, letterSpacing: "0.24px", fontWeight: 500, textTransform: "uppercase" }}>{label}</span>
      </div>
      <div style={{ fontSize: 40, fontWeight: 500, color: T.text, lineHeight: 1.2, fontFamily: '"Aeonik Pro", sans-serif', letterSpacing: "-0.4px" }}>{value}</div>
      {sub && <div style={{ fontSize: 13, color: T.muted, marginTop: 8, letterSpacing: "0.16px" }}>{sub}</div>}
    </div>
  );
}

function FraudTypePanel({ detectedKey, T }) {
  const hasResult = detectedKey !== undefined;
  return (
    <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 20, padding: "24px 32px" }}>
      <div style={{ fontSize: 12, color: T.muted, letterSpacing: "0.24px", fontWeight: 500, marginBottom: 20, textTransform: "uppercase" }}>
        JENIS FRAUD TERDETEKSI
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {FRAUD_TYPES.map(ft => {
          const active = hasResult && detectedKey === ft.key;
          return (
            <div key={ft.key} style={{
              display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", borderRadius: 12,
              background: active ? `${ft.color}15` : T.surface,
              border: `1px solid ${active ? ft.color : "transparent"}`,
              transition: "all 0.3s ease",
            }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: active ? ft.color : T.border, flexShrink: 0, transition: "all 0.3s" }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 14, fontWeight: active ? 600 : 400, color: active ? ft.color : T.muted, letterSpacing: "0.16px" }}>{ft.label}</div>
                <div style={{ fontSize: 12, color: T.muted, marginTop: 2, letterSpacing: "0.16px" }}>{ft.desc}</div>
              </div>
              {active && (
                <div style={{ fontSize: 10, padding: "4px 10px", borderRadius: 9999, background: ft.color, color: "#fff", fontWeight: 600, letterSpacing: "0.24px", flexShrink: 0 }}>
                  AKTIF
                </div>
              )}
            </div>
          );
        })}
        {(() => {
          const active = hasResult && detectedKey === null;
          return (
            <div style={{
              display: "flex", alignItems: "center", gap: 12, padding: "12px 16px", borderRadius: 12,
              background: active ? `${T.safe}15` : T.surface,
              border: `1px solid ${active ? T.safe : "transparent"}`,
              transition: "all 0.3s ease",
            }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: active ? T.safe : T.border, flexShrink: 0, transition: "all 0.3s" }} />
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 14, fontWeight: active ? 600 : 400, color: active ? T.safe : T.muted, letterSpacing: "0.16px" }}>Transaksi Normal</div>
                <div style={{ fontSize: 12, color: T.muted, marginTop: 2, letterSpacing: "0.16px" }}>Tidak ada indikasi fraud</div>
              </div>
              {active && (
                <div style={{ fontSize: 10, padding: "4px 10px", borderRadius: 9999, background: T.safe, color: "#fff", fontWeight: 600, letterSpacing: "0.24px", flexShrink: 0 }}>
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
  const [dark, setDark] = useState(false);
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
      background: activeTab === tab ? T.pillBg : "transparent",
      border: `2px solid ${activeTab === tab ? "transparent" : T.border}`,
      borderRadius: 9999, cursor: "pointer",
      color: activeTab === tab ? T.pillText : T.text,
      fontSize: 16, letterSpacing: "0.24px",
      padding: "10px 24px", fontWeight: 500, fontFamily: "Inter, sans-serif",
      transition: "all 0.2s",
    }}>{tab.charAt(0).toUpperCase() + tab.slice(1)}</button>
  );

  return (
    <div style={{ minHeight: "100vh", background: T.bg, color: T.text, fontFamily: "'Inter', -apple-system, sans-serif", transition: "background 0.3s, color 0.3s", letterSpacing: "0.16px" }}>

      {/* Header */}
      <header style={{ background: T.bg, borderBottom: `1px solid ${T.border}`, padding: "0 32px", display: "flex", alignItems: "center", justifyContent: "space-between", height: 80, position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 40, height: 40, borderRadius: 12, background: T.accent, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <Shield size={20} color="#fff" />
          </div>
          <div>
            <div style={{ fontSize: 24, fontWeight: 500, color: T.text, fontFamily: '"Aeonik Pro", sans-serif', letterSpacing: "-0.32px", lineHeight: 1.2 }}>G.O.T.C.H.A</div>
            <div style={{ fontSize: 12, color: T.muted, letterSpacing: "0.16px" }}>Guard & Observe Transactions</div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {["dashboard", "simulate", "history"].map(navBtn)}
          <div style={{ width: 1, height: 24, background: T.border, margin: "0 8px" }} />
          <button onClick={() => setDark(d => !d)} style={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 9999, cursor: "pointer", padding: "10px 20px", display: "flex", alignItems: "center", gap: 8, color: T.text, fontSize: 14, fontWeight: 500 }}>
            {dark ? <Sun size={16} /> : <Moon size={16} />} {dark ? "Light" : "Dark"}
          </button>
          <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 20px", borderRadius: 9999, background: T.surface, border: `1px solid ${T.border}` }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: backendOk === null ? T.warn : backendOk ? T.safe : T.danger }} />
            <span style={{ fontSize: 14, color: T.text, fontWeight: 500 }}>{backendOk === null ? "Connecting" : backendOk ? "Online" : "Offline"}</span>
          </div>
        </div>
      </header>

      <main style={{ padding: "48px 32px", maxWidth: 1400, margin: "0 auto" }}>
        {globalError && (
          <div style={{ background: `${T.danger}15`, border: `1px solid ${T.danger}`, borderRadius: 12, padding: "16px 20px", marginBottom: 24, fontSize: 14, color: T.danger, display: "flex", justifyContent: "space-between", alignItems: "center", fontWeight: 500 }}>
            <span>⚠ {globalError}</span>
            <button onClick={() => setGlobalError("")} style={{ background: "none", border: "none", color: T.danger, cursor: "pointer", fontSize: 24, lineHeight: 1 }}>×</button>
          </div>
        )}

        {/* DASHBOARD */}
        {activeTab === "dashboard" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 24 }}>
              <StatCard icon={Activity} label="Total Transaksi" value={history.length} color={T.accent} sub="Sesi ini" T={T} />
              <StatCard icon={AlertTriangle} label="Fraud Terdeteksi" value={totalFraud} color={T.danger} sub={`${history.length ? Math.round(totalFraud / history.length * 100) : 0}% dari total`} T={T} />
              <StatCard icon={CheckCircle} label="Transaksi Aman" value={totalNormal} color={T.safe} sub="Lolos verifikasi" T={T} />
              <StatCard icon={TrendingUp} label="Avg Risk Score" value={`${avgRisk}%`} color={T.warn} sub="Rata-rata sesi ini" T={T} />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 24 }}>
              <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 20, padding: 32 }}>
                <div style={{ fontSize: 12, color: T.muted, letterSpacing: "0.24px", fontWeight: 500, marginBottom: 24, textTransform: "uppercase" }}>Risk Score Trend — 20 Transaksi Terakhir</div>
                {chartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={240}>
                    <AreaChart data={chartData}>
                      <defs><linearGradient id="rg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={T.accent} stopOpacity={0.15} /><stop offset="95%" stopColor={T.accent} stopOpacity={0} /></linearGradient></defs>
                      <XAxis dataKey="name" stroke={T.muted} tick={{ fontSize: 12, fill: T.muted }} axisLine={false} tickLine={false} />
                      <YAxis domain={[0, 100]} stroke={T.muted} tick={{ fontSize: 12, fill: T.muted }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 12, fontSize: 14, color: T.text, padding: "12px 16px" }} formatter={v => [`${v}%`, "Risk"]} />
                      <Area type="monotone" dataKey="risk" stroke={T.accent} fill="url(#rg)" strokeWidth={3} dot={{ fill: T.bg, stroke: T.accent, strokeWidth: 2, r: 4 }} activeDot={{ r: 6 }} />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ height: 240, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, color: T.muted }}>
                    <Eye size={40} color={T.border} />
                    <span style={{ fontSize: 16 }}>Belum ada data — jalankan simulasi dulu</span>
                  </div>
                )}
              </div>
              <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 20, padding: 32 }}>
                <div style={{ fontSize: 12, color: T.muted, letterSpacing: "0.24px", fontWeight: 500, marginBottom: 24, textTransform: "uppercase" }}>Distribusi Hasil</div>
                {history.length === 0 ? (
                  <div style={{ height: 200, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 12, color: T.muted }}>
                    <Eye size={32} color={T.border} />
                    <span style={{ fontSize: 14 }}>Belum ada data</span>
                  </div>
                ) : (
                  <>
                    <ResponsiveContainer width="100%" height={180}>
                      <PieChart><Pie data={pieData} cx="50%" cy="50%" innerRadius={50} outerRadius={80} dataKey="value" stroke={T.card} strokeWidth={2}><Cell fill={T.danger} /><Cell fill={T.safe} /></Pie><Tooltip contentStyle={{ background: T.surface, border: `1px solid ${T.border}`, borderRadius: 12, fontSize: 14, color: T.text }} /></PieChart>
                    </ResponsiveContainer>
                    <div style={{ display: "flex", justifyContent: "center", gap: 24, marginTop: 16 }}>
                      {[["Fraud", T.danger], ["Normal", T.safe]].map(([l, c]) => (
                        <div key={l} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, color: T.text, fontWeight: 500 }}><div style={{ width: 12, height: 12, borderRadius: 4, background: c }} />{l}</div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            </div>
            {history.length > 0 && (
              <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 20, padding: 32 }}>
                <div style={{ fontSize: 12, color: T.muted, letterSpacing: "0.24px", fontWeight: 500, marginBottom: 24, textTransform: "uppercase" }}>Riwayat Terkini</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                  {[...history].reverse().slice(0, 5).map((h, i) => (
                    <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "16px 20px", borderRadius: 12, background: T.surface, border: `1px solid ${T.border}` }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
                        <div style={{ width: 10, height: 10, borderRadius: "50%", background: h.is_fraud ? T.danger : T.safe }} />
                        <span style={{ fontSize: 16, fontWeight: 500, color: T.text }}>{h.transaction_id}</span>
                        {h.fraud_type_predicted && <span style={{ fontSize: 14, color: T.muted }}>— {FRAUD_LABEL_MAP[h.fraud_type_predicted] || h.fraud_type_predicted}</span>}
                      </div>
                      <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
                        <span style={{ fontSize: 14, color: T.muted }}>{h.timestamp}</span>
                        <span style={{ fontSize: 16, fontWeight: 600, color: h.is_fraud ? T.danger : T.safe, fontFamily: '"Aeonik Pro", sans-serif', letterSpacing: "-0.16px" }}>{Math.round(h.risk_score * 100)}%</span>
                        <span style={{ fontSize: 12, padding: "6px 16px", borderRadius: 9999, fontWeight: 600, letterSpacing: "0.24px", background: h.recommended_action === "BLOCK" ? T.danger : h.recommended_action === "REVIEW" ? T.warn : T.safe, color: "#fff" }}>{h.recommended_action}</span>
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
          <div style={{ display: "grid", gridTemplateColumns: "420px 1fr", gap: 32 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
              <FraudTypePanel detectedKey={detectedKey} T={T} />
              <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 20, padding: 32 }}>
                <div style={{ fontSize: 12, color: T.muted, letterSpacing: "0.24px", fontWeight: 500, marginBottom: 20, textTransform: "uppercase" }}>Quick Test</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
                  {SCENARIOS.map(s => (
                    <button key={s.label} onClick={() => runPredict(s.data)} disabled={loading} style={{ background: "transparent", border: `2px solid ${T.border}`, borderRadius: 9999, padding: "10px 20px", cursor: loading ? "not-allowed" : "pointer", fontSize: 14, fontWeight: 500, color: T.text, transition: "all 0.2s", opacity: loading ? 0.5 : 1 }}
                      onMouseEnter={e => { if (!loading) { e.currentTarget.style.borderColor = T.text; } }}
                      onMouseLeave={e => { e.currentTarget.style.borderColor = T.border; }}>
                      {s.label}
                    </button>
                  ))}
                </div>
              </div>
              <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 20, padding: 32 }}>
                <div style={{ fontSize: 12, color: T.muted, letterSpacing: "0.24px", fontWeight: 500, marginBottom: 20, textTransform: "uppercase" }}>Custom Input (JSON)</div>
                <textarea value={customInput} onChange={e => setCustomInput(e.target.value)} placeholder='{"transaction_id": "TXN-001", ...}'
                  style={{ width: "100%", height: 200, background: T.inputBg, border: `1px solid ${T.border}`, borderRadius: 12, color: T.text, fontSize: 14, padding: 16, resize: "none", fontFamily: "monospace", outline: "none", lineHeight: 1.5 }}
                  onFocus={e => e.target.style.borderColor = T.text}
                  onBlur={e => e.target.style.borderColor = T.border} />
                {customError && <div style={{ fontSize: 14, color: T.danger, marginTop: 12 }}>{customError}</div>}
                <button onClick={handleCustomSubmit} disabled={loading || !customInput.trim()} style={{ marginTop: 24, width: "100%", padding: "14px 32px", background: loading || !customInput.trim() ? T.border : T.pillBg, border: "none", borderRadius: 9999, cursor: loading || !customInput.trim() ? "not-allowed" : "pointer", color: T.pillText, fontSize: 16, fontWeight: 600, letterSpacing: "0.24px", transition: "all 0.2s" }}
                  onMouseEnter={e => { if (!loading && customInput.trim()) e.currentTarget.style.background = T.pillHover; }}
                  onMouseLeave={e => { if (!loading && customInput.trim()) e.currentTarget.style.background = T.pillBg; }}>
                  {loading ? "MENGANALISIS..." : "ANALISIS TRANSAKSI"}
                </button>
              </div>
            </div>

            <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 20, padding: 48, minHeight: 600 }}>
              {!result && !loading && (
                <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, color: T.muted }}>
                  <Zap size={64} color={T.border} />
                  <span style={{ fontSize: 18, fontWeight: 500 }}>Pilih quick test atau masukkan JSON</span>
                </div>
              )}
              {loading && (
                <div style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 24 }}>
                  <div style={{ width: 64, height: 64, border: `4px solid ${T.border}`, borderTop: `4px solid ${T.text}`, borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
                  <span style={{ fontSize: 14, color: T.muted, letterSpacing: "0.24px", fontWeight: 500 }}>MENGANALISIS TRANSAKSI...</span>
                </div>
              )}
              {result && !loading && (
                <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
                    <RiskMeter score={result.risk_score} T={T} />
                    <div>
                      <div style={{ fontSize: 48, fontWeight: 500, color: actionColor, letterSpacing: "-0.48px", lineHeight: 1, fontFamily: '"Aeonik Pro", sans-serif' }}>{result.recommended_action}</div>
                      <div style={{ fontSize: 16, color: T.muted, marginTop: 8, marginBottom: 16, fontWeight: 500 }}>{result.transaction_id}</div>
                      {result.fraud_type_predicted && (
                        <div style={{ display: "inline-flex", alignItems: "center", gap: 8, fontSize: 14, padding: "8px 16px", borderRadius: 9999, background: T.danger, color: "#fff", fontWeight: 600, letterSpacing: "0.16px" }}>
                          {FRAUD_LABEL_MAP[result.fraud_type_predicted] || result.fraud_type_predicted}
                        </div>
                      )}
                      {!result.is_fraud && (
                        <div style={{ display: "inline-flex", alignItems: "center", gap: 8, fontSize: 14, padding: "8px 16px", borderRadius: 9999, background: T.safe, color: "#fff", fontWeight: 600, letterSpacing: "0.16px" }}>
                          Transaksi Normal
                        </div>
                      )}
                    </div>
                  </div>
                  <div style={{ height: 1, background: T.border }} />
                  <div style={{ background: T.surface, borderRadius: 16, padding: 24, border: `1px solid ${T.border}` }}>
                    <div style={{ fontSize: 12, color: T.text, letterSpacing: "0.24px", fontWeight: 600, marginBottom: 12, textTransform: "uppercase" }}>AI Explanation — GPT-4o</div>
                    <div style={{ fontSize: 16, color: T.text, lineHeight: 1.6, fontWeight: 400 }}>{result.explanation}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 12, color: T.muted, letterSpacing: "0.24px", fontWeight: 500, marginBottom: 16, textTransform: "uppercase" }}>Sinyal Terdeteksi</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
                      {Object.entries(result.signals).map(([k, v]) => (
                        <div key={k} style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "8px 16px", borderRadius: 9999, fontSize: 14, background: v ? `${T.danger}15` : T.surface, border: `1px solid ${v ? T.danger : T.border}`, color: v ? T.danger : T.muted, fontWeight: v ? 600 : 500 }}>
                          <div style={{ width: 8, height: 8, borderRadius: "50%", background: v ? T.danger : T.border }} />
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
          <div style={{ background: T.card, border: `1px solid ${T.border}`, borderRadius: 20, padding: 32 }}>
            <div style={{ fontSize: 12, color: T.muted, letterSpacing: "0.24px", fontWeight: 500, marginBottom: 24, textTransform: "uppercase" }}>Riwayat Lengkap — {history.length} Transaksi</div>
            {history.length === 0 ? (
              <div style={{ textAlign: "center", color: T.muted, padding: 64, fontSize: 16 }}>Belum ada riwayat transaksi</div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1.5fr 100px 120px", padding: "8px 20px", gap: 16, fontSize: 12, color: T.muted, letterSpacing: "0.24px", fontWeight: 600, textTransform: "uppercase" }}>
                  <span>Transaction ID</span><span>Waktu</span><span>Jenis</span><span>Risk</span><span>Action</span>
                </div>
                {[...history].reverse().map((h, i) => (
                  <div key={i} style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1.5fr 100px 120px", alignItems: "center", padding: "16px 20px", borderRadius: 12, background: T.surface, border: `1px solid ${T.border}`, gap: 16 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <div style={{ width: 10, height: 10, borderRadius: "50%", background: h.is_fraud ? T.danger : T.safe, flexShrink: 0 }} />
                      <span style={{ fontSize: 14, color: T.text, fontWeight: 500 }}>{h.transaction_id}</span>
                    </div>
                    <span style={{ fontSize: 14, color: T.muted }}>{h.timestamp}</span>
                    <span style={{ fontSize: 14, color: h.is_fraud ? T.danger : T.safe, fontWeight: 500 }}>{h.fraud_type_predicted ? (FRAUD_LABEL_MAP[h.fraud_type_predicted] || h.fraud_type_predicted) : "Normal"}</span>
                    <span style={{ fontSize: 16, fontWeight: 500, color: h.is_fraud ? T.danger : T.safe, fontFamily: '"Aeonik Pro", sans-serif', letterSpacing: "-0.16px" }}>{Math.round(h.risk_score * 100)}%</span>
                    <span style={{ fontSize: 12, padding: "6px 16px", borderRadius: 9999, textAlign: "center", fontWeight: 600, letterSpacing: "0.24px", background: h.recommended_action === "BLOCK" ? T.danger : h.recommended_action === "REVIEW" ? T.warn : T.safe, color: "#fff" }}>{h.recommended_action}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        /* Fallback for Aeonik Pro if not installed locally */
        @font-face {
          font-family: 'Aeonik Pro';
          src: local('Aeonik Pro'), local('AeonikPro-Medium');
          font-weight: 500;
          font-style: normal;
        }
      `}</style>
    </div>
  );
}
