import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = '';

function App() {
  const [zones, setZones] = useState([]);

  useEffect(() => {
    const interval = setInterval(() => {
      axios.get(`${API_BASE}/api/zones`).then(r => setZones(r.data)).catch(() => {});
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const stateLabel = (s) => ({ active: 'Đang hoạt động', still: 'Đứng yên', inactivity: 'Không hoạt động' }[s] || '?');

  return (
    <div style={{maxWidth:800,margin:'0 auto',padding:20,fontSize:16}}>
      <h1>🏥 ElderCare</h1>
      <p>Hệ thống giám sát người cao tuổi qua WiFi</p>
      {zones.map(z => (
        <div key={z.zone_id} style={{border:'1px solid #ddd',borderRadius:8,padding:12,marginBottom:10}}>
          <strong>{z.name}</strong>
          <span style={{color:z.online?'green':'red',marginLeft:10}}>{z.online?'Trực tuyến':'Mất kết nối'}</span>
          <div>{stateLabel(z.activity_state)} {z.respiration_bpm ? `| Nhịp thở: ${z.respiration_bpm}` : ''}</div>
        </div>
      ))}
    </div>
  );
}
export default App;
