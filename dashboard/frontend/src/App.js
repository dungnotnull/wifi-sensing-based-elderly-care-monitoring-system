import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';

const API_BASE = '';
const POLL_MS = 5000;
const TOKEN_KEY = 'eldercare_token';

// -- Vietnamese labels -------------------------------------------------------
const LABELS = {
  appTitle: 'ElderCare',
  appSubtitle: 'He thong giam sat nguoi cao tuoi qua WiFi',
  zoneStatus: 'Trang thai khu vuc',
  activity: { active: 'Dang hoat dong', still: 'Dung yen', inactivity: 'Khong hoat dong', unknown: 'Cho du lieu', starting: 'Dang khoi dong' },
  sleepStage: { awake: 'Thuc', light: 'Ngu nông', deep: 'Ngu sau', unknown: '--' },
  online: 'Truc tuyen',
  offline: 'Mat ket noi',
  respiration: 'Nhip tho',
  heartRate: 'Nhip tim',
  bpm: 'bmp',
  fallAlert: 'Phat hien ngã!',
  confidence: 'Do tin cay',
  vitalsChart: 'Nhip tho (24 gio)',
  sleepChart: 'Diem chat luong ngu (30 ngay)',
  alertLog: 'Lich su canh bao',
  systemHealth: 'Tinh trang he thong',
  dailySummary: 'Tom tat hang ngay',
  noData: 'Chua co du lieu',
  severity: { INFO: 'Thong tin', WARNING: 'Canh bao', EMERGENCY: 'Khan cap' },
  connectivity: 'Ket noi ESP32',
  latency: 'Do tre xu ly',
  diskUsage: 'Dung luong o dia',
  workers: 'Trang thai worker',
};

// -- Colors ------------------------------------------------------------------
const COLORS = {
  bg: '#f5f7fa',
  cardBg: '#ffffff',
  primary: '#1976d2',
  primaryLight: '#e3f2fd',
  danger: '#d32f2f',
  dangerLight: '#ffebee',
  warning: '#f57c00',
  warningLight: '#fff3e0',
  success: '#388e3c',
  successLight: '#e8f5e9',
  text: '#212121',
  textSecondary: '#616161',
  border: '#e0e0e0',
  chartLine: '#1976d2',
  chartGrid: '#e0e0e0',
  chartFill: 'rgba(25,118,210,0.08)',
};

// -- Auth helpers -------------------------------------------------------------
function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

function getAuthHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// -- Login screen -------------------------------------------------------------
function LoginScreen({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const resp = await axios.post(`${API_BASE}/api/login`, { username, password });
      setToken(resp.data.access_token);
      onLogin();
    } catch (err) {
      if (err.response && err.response.status === 401) {
        setError('Tên đăng nhập hoặc mật khẩu không đúng');
      } else {
        setError('Không thể kết nối đến máy chủ');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', minHeight: '100vh',
      background: COLORS.bg, fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      <div style={{
        background: COLORS.cardBg, borderRadius: 12, padding: '32px 40px',
        boxShadow: '0 2px 12px rgba(0,0,0,0.1)', width: 360, maxWidth: '90vw',
      }}>
        <h1 style={{ textAlign: 'center', color: COLORS.primary, fontSize: 24, marginBottom: 4 }}>
          {LABELS.appTitle}
        </h1>
        <p style={{ textAlign: 'center', color: COLORS.textSecondary, fontSize: 14, marginBottom: 24 }}>
          Đăng nhập để truy cập hệ thống giám sát
        </p>
        {error && (
          <div style={{
            background: COLORS.dangerLight, color: COLORS.danger, borderRadius: 6,
            padding: '10px 14px', marginBottom: 16, fontSize: 14,
          }}>
            {error}
          </div>
        )}
        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: 16 }}>
            <label style={{ display: 'block', fontSize: 14, fontWeight: 600, marginBottom: 4, color: COLORS.text }}>
              Tên đăng nhập
            </label>
            <input
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              autoFocus
              style={{
                width: '100%', padding: '10px 12px', fontSize: 16,
                borderRadius: 6, border: `1px solid ${COLORS.border}`,
                boxSizing: 'border-box',
              }}
            />
          </div>
          <div style={{ marginBottom: 20 }}>
            <label style={{ display: 'block', fontSize: 14, fontWeight: 600, marginBottom: 4, color: COLORS.text }}>
              Mật khẩu
            </label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              style={{
                width: '100%', padding: '10px 12px', fontSize: 16,
                borderRadius: 6, border: `1px solid ${COLORS.border}`,
                boxSizing: 'border-box',
              }}
            />
          </div>
          <button
            type="submit"
            disabled={loading}
            style={{
              width: '100%', padding: '12px', fontSize: 16, fontWeight: 700,
              color: '#fff', background: COLORS.primary, border: 'none',
              borderRadius: 6, cursor: loading ? 'default' : 'pointer',
              opacity: loading ? 0.7 : 1,
            }}
          >
            {loading ? 'Đang đăng nhập...' : 'Đăng nhập'}
          </button>
        </form>
      </div>
    </div>
  );
}

// -- Severity badge -----------------------------------------------------------
function SeverityBadge({ level }) {
  const cfg = {
    INFO: { bg: COLORS.primaryLight, color: COLORS.primary },
    WARNING: { bg: COLORS.warningLight, color: COLORS.warning },
    EMERGENCY: { bg: COLORS.dangerLight, color: COLORS.danger },
  }[level] || { bg: COLORS.border, color: COLORS.textSecondary };
  return (
    <span style={{
      display: 'inline-block', padding: '2px 10px', borderRadius: 12,
      fontSize: 14, fontWeight: 600, background: cfg.bg, color: cfg.color,
      minWidth: 80, textAlign: 'center',
    }}>
      {LABELS.severity[level] || level}
    </span>
  );
}

// -- Canvas-based line chart --------------------------------------------------
function LineChart({ data, width, height, label }) {
  const canvasRef = useRef(null);
  const dpr = window.devicePixelRatio || 1;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = width;
    const h = height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);
    const padL = 50, padR = 16, padT = 24, padB = 32;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    // Background
    ctx.fillStyle = COLORS.cardBg;
    ctx.fillRect(0, 0, w, h);

    if (!data || data.length === 0) {
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(LABELS.noData, w / 2, h / 2);
      return;
    }

    const values = data.map(d => d.value).filter(v => v != null);
    if (values.length === 0) {
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(LABELS.noData, w / 2, h / 2);
      return;
    }
    const minV = Math.min(...values);
    const maxV = Math.max(...values);
    const rangeV = maxV - minV || 1;

    // Grid lines
    ctx.strokeStyle = COLORS.chartGrid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText((maxV - (rangeV / 4) * i).toFixed(1), padL - 6, y + 4);
    }

    // Data line + fill
    const points = data.filter(d => d.value != null);
    ctx.beginPath();
    points.forEach((d, i) => {
      const x = padL + (i / Math.max(points.length - 1, 1)) * plotW;
      const y = padT + plotH - ((d.value - minV) / rangeV) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = COLORS.chartLine;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Fill area
    const lastX = padL + plotW;
    ctx.lineTo(lastX, padT + plotH);
    ctx.lineTo(padL, padT + plotH);
    ctx.closePath();
    ctx.fillStyle = COLORS.chartFill;
    ctx.fill();

    // Title
    ctx.fillStyle = COLORS.text;
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(label, padL, 16);
  }, [data, width, height, dpr, label]);

  return <canvas ref={canvasRef} style={{ display: 'block' }} />;
}

// -- Canvas-based bar chart ---------------------------------------------------
function BarChart({ data, width, height, label }) {
  const canvasRef = useRef(null);
  const dpr = window.devicePixelRatio || 1;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = width;
    const h = height;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);

    ctx.clearRect(0, 0, w, h);
    const padL = 40, padR = 16, padT = 24, padB = 40;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    ctx.fillStyle = COLORS.cardBg;
    ctx.fillRect(0, 0, w, h);

    if (!data || data.length === 0) {
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(LABELS.noData, w / 2, h / 2);
      return;
    }

    const maxV = 100; // sleep score is 0-100

    // Grid
    ctx.strokeStyle = COLORS.chartGrid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(String(maxV - 25 * i), padL - 6, y + 4);
    }

    // Bars
    const barW = Math.max(4, Math.min(24, plotW / data.length - 4));
    const gap = (plotW - barW * data.length) / (data.length + 1);
    data.forEach((d, i) => {
      const x = padL + gap + i * (barW + gap);
      const barH = (d.value / maxV) * plotH;
      const y = padT + plotH - barH;

      // Color based on score
      let color = COLORS.success;
      if (d.value < 50) color = COLORS.danger;
      else if (d.value < 70) color = COLORS.warning;

      ctx.fillStyle = color;
      ctx.fillRect(x, y, barW, barH);

      // Label
      ctx.save();
      ctx.translate(x + barW / 2, padT + plotH + 8);
      ctx.rotate(-0.6);
      ctx.fillStyle = COLORS.textSecondary;
      ctx.font = '11px sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(d.label, 0, 0);
      ctx.restore();
    });

    // Title
    ctx.fillStyle = COLORS.text;
    ctx.font = 'bold 16px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(label, padL, 16);
  }, [data, width, height, dpr, label]);

  return <canvas ref={canvasRef} style={{ display: 'block' }} />;
}

// -- Zone status card ---------------------------------------------------------
function ZoneCard({ zone }) {
  const stateLabel = LABELS.activity[zone.activity_state] || zone.activity_state;
  const sleepLabel = LABELS.sleepStage[zone.sleep_stage] || zone.sleep_stage;
  const isOnline = zone.online;

  return (
    <div style={{
      background: COLORS.cardBg, borderRadius: 10, padding: 16,
      border: zone.fall_detected ? `2px solid ${COLORS.danger}` : `1px solid ${COLORS.border}`,
      marginBottom: 12, position: 'relative', minHeight: 44,
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontSize: 18, fontWeight: 700, color: COLORS.text }}>{zone.name}</span>
        <span style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          fontSize: 14, fontWeight: 600,
          color: isOnline ? COLORS.success : COLORS.danger,
        }}>
          <span style={{
            width: 10, height: 10, borderRadius: '50%',
            background: isOnline ? COLORS.success : COLORS.danger,
            display: 'inline-block',
          }} />
          {isOnline ? LABELS.online : LABELS.offline}
        </span>
      </div>

      {/* Fall alert badge */}
      {zone.fall_detected && (
        <div style={{
          background: COLORS.dangerLight, color: COLORS.danger, borderRadius: 6,
          padding: '8px 12px', marginBottom: 8, fontWeight: 700, fontSize: 16,
          display: 'flex', alignItems: 'center', gap: 8,
        }}>
          {LABELS.fallAlert}
          <span style={{ fontSize: 14, fontWeight: 400 }}>
            ({(zone.fall_confidence * 100).toFixed(0)}% {LABELS.confidence})
          </span>
        </div>
      )}

      {/* Metrics row */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, fontSize: 16 }}>
        <div>
          <span style={{ color: COLORS.textSecondary }}>{LABELS.activity}:</span>{' '}
          <strong style={{ color: zone.activity_state === 'inactivity' ? COLORS.warning : COLORS.text }}>
            {stateLabel}
          </strong>
        </div>
        {zone.respiration_bpm != null && (
          <div>
            <span style={{ color: COLORS.textSecondary }}>{LABELS.respiration}:</span>{' '}
            <strong>{zone.respiration_bpm.toFixed(1)} {LABELS.bpm}</strong>
          </div>
        )}
        {zone.heart_rate_bpm != null && (
          <div>
            <span style={{ color: COLORS.textSecondary }}>{LABELS.heartRate}:</span>{' '}
            <strong>{zone.heart_rate_bpm.toFixed(1)} {LABELS.bpm}</strong>
          </div>
        )}
        <div>
          <span style={{ color: COLORS.textSecondary }}>Ngủ:</span>{' '}
          <strong>{sleepLabel}</strong>
          {zone.sleep_score != null && ` (${zone.sleep_score.toFixed(0)}/100)`}
        </div>
      </div>
    </div>
  );
}

// -- Alert log ----------------------------------------------------------------
function AlertLog({ alerts }) {
  if (!alerts || alerts.length === 0) {
    return (
      <div style={{ color: COLORS.textSecondary, padding: 12, fontSize: 16 }}>
        {LABELS.noData}
      </div>
    );
  }
  return (
    <div style={{ maxHeight: 360, overflowY: 'auto' }}>
      {alerts.map((a, i) => {
        const ts = a.timestamp ? new Date(a.timestamp * 1000).toLocaleString('vi-VN') : '--';
        return (
          <div key={a.id ?? i} style={{
            display: 'flex', alignItems: 'center', gap: 12,
            padding: '10px 0', borderBottom: `1px solid ${COLORS.border}`,
            fontSize: 16, minHeight: 44,
          }}>
            <SeverityBadge level={a.level} />
            <span style={{ color: COLORS.textSecondary, minWidth: 130, fontSize: 14 }}>{ts}</span>
            <span style={{ fontWeight: 600 }}>{a.zone_name || a.zone_id || '--'}</span>
            <span style={{ color: COLORS.text }}>{a.description || a.event_type || ''}</span>
          </div>
        );
      })}
    </div>
  );
}

// -- System health panel ------------------------------------------------------
function SystemHealthPanel({ health }) {
  const items = [
    { label: LABELS.connectivity, value: `${health.zones_online || 0}/${health.zones_total || 0} zones` },
    { label: LABELS.workers, value: health.status === 'healthy' ? 'Hoat dong' : 'Loi' },
  ];
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 16, fontSize: 16 }}>
      {items.map(item => (
        <div key={item.label} style={{
          background: COLORS.primaryLight, borderRadius: 8, padding: '12px 18px',
          minWidth: 140,
        }}>
          <div style={{ color: COLORS.textSecondary, fontSize: 14 }}>{item.label}</div>
          <div style={{ fontWeight: 700, marginTop: 4 }}>{item.value}</div>
        </div>
      ))}
    </div>
  );
}

// -- Daily summary panel ------------------------------------------------------
function DailySummaryPanel({ summary }) {
  if (!summary) return <div style={{ color: COLORS.textSecondary, fontSize: 16 }}>{LABELS.noData}</div>;
  return (
    <pre style={{
      whiteSpace: 'pre-wrap', fontFamily: 'inherit', fontSize: 16,
      lineHeight: 1.6, margin: 0, color: COLORS.text,
    }}>
      {summary}
    </pre>
  );
}

// -- Main App -----------------------------------------------------------------
function App() {
  const [authenticated, setAuthenticated] = useState(!!getToken());
  const [zones, setZones] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [health, setHealth] = useState({});
  const [dailySummary, setDailySummary] = useState('');
  const [vitalsData, setVitalsData] = useState({});
  const [sleepData, setSleepData] = useState({});
  const [selectedZone, setSelectedZone] = useState(null);
  const [chartWidth, setChartWidth] = useState(600);
  const chartContainerRef = useRef(null);

  const handleLogout = () => {
    clearToken();
    setAuthenticated(false);
  };

  if (!authenticated) {
    return <LoginScreen onLogin={() => setAuthenticated(true)} />;
  }

  return <Dashboard onLogout={handleLogout} />;
}

// -- Dashboard (wrapped by App) -----------------------------------------------
function Dashboard({ onLogout }) {
  const [zones, setZones] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [health, setHealth] = useState({});
  const [dailySummary, setDailySummary] = useState('');
  const [vitalsData, setVitalsData] = useState({});
  const [sleepData, setSleepData] = useState({});
  const [selectedZone, setSelectedZone] = useState(null);
  const [chartWidth, setChartWidth] = useState(600);
  const chartContainerRef = useRef(null);

  // Responsive chart width
  useEffect(() => {
    const updateWidth = () => {
      if (chartContainerRef.current) {
        setChartWidth(chartContainerRef.current.offsetWidth);
      }
    };
    updateWidth();
    window.addEventListener('resize', updateWidth);
    return () => window.removeEventListener('resize', updateWidth);
  }, []);

  // Polling
  useEffect(() => {
    const config = { headers: getAuthHeaders() };
    const fetchAll = () => {
      axios.get(`${API_BASE}/api/zones`, config).then(r => {
        setZones(r.data);
        if (!selectedZone && r.data.length > 0) {
          setSelectedZone(r.data[0].zone_id);
        }
      }).catch(err => {
        if (err.response && err.response.status === 401) onLogout();
      });
      axios.get(`${API_BASE}/api/alerts?limit=30`, config).then(r => setAlerts(r.data)).catch(() => {});
      axios.get(`${API_BASE}/api/health`, config).then(r => setHealth(r.data)).catch(() => {});
      axios.get(`${API_BASE}/api/daily-summary`, config).then(r => setDailySummary(r.data.summary || '')).catch(() => {});
    };
    fetchAll();
    const interval = setInterval(fetchAll, POLL_MS);
    return () => clearInterval(interval);
  }, [selectedZone, onLogout]);

  // Fetch vitals for selected zone
  useEffect(() => {
    if (!selectedZone) return;
    const config = { headers: getAuthHeaders() };
    axios.get(`${API_BASE}/api/vitals?zone_id=${selectedZone}&hours=24`, config).then(r => {
      setVitalsData(prev => ({
        ...prev,
        [selectedZone]: (r.data || []).filter(d => d.respiration_bpm != null).map(d => ({
          value: d.respiration_bpm, time: d.timestamp,
        })),
      }));
    }).catch(() => {});
    axios.get(`${API_BASE}/api/sleep?zone_id=${selectedZone}&days=30`, config).then(r => {
      setSleepData(prev => ({
        ...prev,
        [selectedZone]: (r.data || []).map(d => ({
          value: d.sleep_score, label: d.date,
        })),
      }));
    }).catch(() => {});
  }, [selectedZone, zones]);

  const handleZoneSelect = useCallback((zoneId) => {
    setSelectedZone(zoneId);
  }, []);

  const zoneOptions = zones.length > 0 ? zones : [{ zone_id: 'zone_default', name: 'He thong' }];

  return (
    <div style={{
      background: COLORS.bg, minHeight: '100vh', fontSize: 16, color: COLORS.text,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      {/* Header */}
      <header style={{
        background: COLORS.primary, color: '#fff', padding: '14px 20px',
        display: 'flex', alignItems: 'center', gap: 12, justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>{LABELS.appTitle}</h1>
          <span style={{ fontSize: 16, opacity: 0.85 }}>{LABELS.appSubtitle}</span>
        </div>
        <button
          onClick={onLogout}
          style={{
            background: 'rgba(255,255,255,0.2)', color: '#fff', border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: 6, padding: '6px 16px', fontSize: 14, fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          Đăng xuất
        </button>
      </header>

      <div style={{ maxWidth: 960, margin: '0 auto', padding: '16px 16px 40px' }}>
        {/* Zone Status Cards */}
        <section style={{ marginBottom: 24 }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 12 }}>{LABELS.zoneStatus}</h2>
          {zones.length === 0 && (
            <div style={{ color: COLORS.textSecondary, fontSize: 16, padding: 12 }}>
              {LABELS.noData}
            </div>
          )}
          {zones.map(z => (
            <ZoneCard key={z.zone_id} zone={z} />
          ))}
        </section>

        {/* Charts -- zone selector */}
        <section style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
            <label style={{ fontSize: 16, fontWeight: 600 }}>Khu vuc:</label>
            <select
              value={selectedZone || ''}
              onChange={e => handleZoneSelect(e.target.value)}
              style={{ fontSize: 16, padding: '8px 12px', borderRadius: 6, border: `1px solid ${COLORS.border}`, minWidth: 180 }}
            >
              {zoneOptions.map(z => (
                <option key={z.zone_id} value={z.zone_id}>{z.name}</option>
              ))}
            </select>
          </div>

          {/* Vitals trend */}
          <div ref={chartContainerRef} style={{ marginBottom: 20 }}>
            <LineChart
              data={vitalsData[selectedZone] || []}
              width={chartWidth}
              height={260}
              label={LABELS.vitalsChart}
            />
          </div>

          {/* Sleep quality */}
          <div>
            <BarChart
              data={sleepData[selectedZone] || []}
              width={chartWidth}
              height={280}
              label={LABELS.sleepChart}
            />
          </div>
        </section>

        {/* Alert Log */}
        <section style={{ marginBottom: 24 }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 12 }}>{LABELS.alertLog}</h2>
          <div style={{
            background: COLORS.cardBg, borderRadius: 10, padding: 16,
            border: `1px solid ${COLORS.border}`,
          }}>
            <AlertLog alerts={alerts} />
          </div>
        </section>

        {/* System Health */}
        <section style={{ marginBottom: 24 }}>
          <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 12 }}>{LABELS.systemHealth}</h2>
          <SystemHealthPanel health={health} />
        </section>

        {/* Daily Summary */}
        <section>
          <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 12 }}>{LABELS.dailySummary}</h2>
          <div style={{
            background: COLORS.cardBg, borderRadius: 10, padding: 16,
            border: `1px solid ${COLORS.border}`,
          }}>
            <DailySummaryPanel summary={dailySummary} />
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
