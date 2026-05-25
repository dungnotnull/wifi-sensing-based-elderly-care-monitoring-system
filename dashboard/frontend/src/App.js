import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_BASE || '';

function App() {
  const [zones, setZones] = useState([]);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    const interval = setInterval(() => {
      axios.get(`${API_BASE}/api/zones`)
        .then(res => setZones(res.data))
        .catch(err => console.error('Failed to fetch zones:', err));

      axios.get(`${API_BASE}/api/alerts?limit=10&unacknowledged_only=true`)
        .then(res => setAlerts(res.data))
        .catch(err => console.error('Failed to fetch alerts:', err));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getStateIcon = (state) => {
    switch (state) {
      case 'active': return '🏃';
      case 'still': return '😴';
      case 'inactivity': return '⚠️';
      default: return '❓';
    }
  };

  const getStateLabelVN = (state) => {
    switch (state) {
      case 'active': return 'Đang hoạt động';
      case 'still': return 'Đứng yên';
      case 'inactivity': return 'Không hoạt động';
      default: return 'Không xác định';
    }
  };

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: 20, fontFamily: 'sans-serif', fontSize: 16 }}>
      <header style={{ textAlign: 'center', marginBottom: 30 }}>
        <h1>🏥 ElderCare</h1>
        <p style={{ color: '#666' }}>Hệ thống giám sát người cao tuổi qua WiFi</p>
      </header>

      <section style={{ marginBottom: 30 }}>
        <h2>📍 Trạng thái các khu vực</h2>
        {zones.length === 0 && <p style={{ color: '#999' }}>Đang tải dữ liệu...</p>}
        {zones.map(zone => (
          <div
            key={zone.zone_id}
            style={{
              border: '1px solid #ddd',
              borderRadius: 8,
              padding: 15,
              marginBottom: 10,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div>
              <strong>{zone.name}</strong>
              <div style={{ color: zone.online ? 'green' : 'red', fontSize: 14 }}>
                {zone.online ? '✅ Trực tuyến' : '❌ Mất kết nối'}
              </div>
            </div>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: 24 }}>{getStateIcon(zone.activity_state)}</div>
              <div style={{ fontSize: 14 }}>{getStateLabelVN(zone.activity_state)}</div>
              {zone.respiration_bpm && (
                <div style={{ fontSize: 14, color: '#666' }}>
                  Nhịp thở: {zone.respiration_bpm} lần/phút
                </div>
              )}
            </div>
          </div>
        ))}
      </section>

      <section>
        <h2>🔔 Cảnh báo gần đây</h2>
        {alerts.length === 0 && <p style={{ color: '#999' }}>Không có cảnh báo nào</p>}
        {alerts.map(alert => (
          <div
            key={alert.id}
            style={{
              border: '1px solid #ddd',
              borderRadius: 8,
              padding: 12,
              marginBottom: 8,
              borderLeft: `4px solid ${
                alert.level === 'EMERGENCY' ? '#e53935' :
                alert.level === 'WARNING' ? '#fb8c00' : '#43a047'
              }`,
            }}
          >
            <div style={{ fontSize: 14, color: '#666' }}>
              {new Date(alert.timestamp * 1000).toLocaleTimeString('vi-VN')} — {alert.zone_name}
            </div>
            <div>{alert.description}</div>
          </div>
        ))}
      </section>
    </div>
  );
}

export default App;
