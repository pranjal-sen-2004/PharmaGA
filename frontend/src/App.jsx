import React, { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Activity, ShieldAlert, Zap, AlertTriangle } from 'lucide-react';
import './App.css';

function App() {
  const [config, setConfig] = useState({ age: 55, bmi: 24.5, hla_risk: 0.15 });
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedRegimen, setSelectedRegimen] = useState(null);

  const handleRunGA = async () => {
    setLoading(true);
    setSelectedRegimen(null);
    try {
      const response = await fetch("http://localhost:8000/api/run-ga", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(config),
      });
      const data = await response.json();
      setResults(data.pareto_front);
    } catch (error) {
      console.error("Failed to fetch GA results:", error);
    } finally {
      setLoading(false);
    }
  };

  const handlePointClick = (data) => {
    setSelectedRegimen(data);
  };

  // Custom tooltip for the ScatterChart
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="custom-tooltip">
          <p><strong>{data.id}</strong></p>
          <p>Efficacy: {data.efficacy}</p>
          <p>Toxicity: {data.toxicity}</p>
          <p className="click-hint">Click to view details</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="dashboard-container">
      <header className="header">
        <h1>PharmaGA Personalised Drug Selection</h1>
        <p>Niching Multi-Objective Genetic Algorithm Explorer</p>
      </header>

      <div className="control-panel">
        <div className="input-group">
          <label>Patient Age <span>{config.age}</span></label>
          <input type="range" min="30" max="80" value={config.age} 
            onChange={e => setConfig({...config, age: parseInt(e.target.value)})} />
        </div>
        <div className="input-group">
          <label>BMI <span>{config.bmi.toFixed(1)}</span></label>
          <input type="range" min="18" max="42" step="0.1" value={config.bmi} 
            onChange={e => setConfig({...config, bmi: parseFloat(e.target.value)})} />
        </div>
        <div className="input-group">
          <label>HLA Risk Factor <span>{config.hla_risk.toFixed(2)}</span></label>
          <input type="range" min="0" max="1" step="0.05" value={config.hla_risk} 
            onChange={e => setConfig({...config, hla_risk: parseFloat(e.target.value)})} />
        </div>
        
        <button className="run-button" onClick={handleRunGA} disabled={loading}>
          {loading ? "Running Genetic Algorithm..." : "Generate Optimal Regimens"}
        </button>
      </div>

      <div className="main-content">
        <div className="chart-section">
          <h2>Pareto Front (Efficacy vs. Toxicity)</h2>
          {results.length === 0 && !loading ? (
             <div className="empty-state">Adjust parameters and click Generate to see the Pareto front.</div>
          ) : (
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={400}>
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                  <XAxis type="number" dataKey="efficacy" name="Efficacy" domain={[0, 1]} label={{ value: 'Efficacy (Higher is Better)', position: 'insideBottom', offset: -10 }} />
                  <YAxis type="number" dataKey="toxicity" name="Toxicity" domain={[0, 1]} label={{ value: 'Toxicity (Lower is Better)', angle: -90, position: 'insideLeft' }} />
                  <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
                  <Scatter name="Regimens" data={results} onClick={handlePointClick}>
                    {results.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={selectedRegimen?.id === entry.id ? "#ef4444" : "#3b82f6"} 
                            r={selectedRegimen?.id === entry.id ? 8 : 6} style={{ cursor: 'pointer', transition: 'all 0.3s' }} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {selectedRegimen && (
          <div className="details-section">
            <h2>{selectedRegimen.id} Details</h2>
            
            <div className="composition-card">
              <h3>Drug Composition</h3>
              <ul>
                {selectedRegimen.drugs.map((d, i) => (
                  <li key={i}><strong>{d.name}</strong> — {d.dose} Dose Unit(s)</li>
                ))}
              </ul>
            </div>

            <div className="metrics-grid">
              <div className="metric-box success">
                <div className="metric-header">
                  <Activity size={18} />
                  <span>Efficacy</span>
                </div>
                <strong>{(selectedRegimen.efficacy * 100).toFixed(1)}%</strong>
              </div>
              <div className="metric-box warning">
                <div className="metric-header">
                  <ShieldAlert size={18} />
                  <span>Toxicity</span>
                </div>
                <strong>{(selectedRegimen.toxicity * 100).toFixed(1)}%</strong>
              </div>
              <div className="metric-box info">
                <div className="metric-header">
                  <Zap size={18} />
                  <span>PK Mismatch</span>
                </div>
                <strong>{(selectedRegimen.pk_mismatch * 100).toFixed(1)}%</strong>
              </div>
              <div className="metric-box danger">
                <div className="metric-header">
                  <AlertTriangle size={18} />
                  <span>DDI Risk</span>
                </div>
                <strong>{(selectedRegimen.ddi * 100).toFixed(1)}%</strong>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;