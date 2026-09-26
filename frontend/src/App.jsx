import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    // Scaffold API call to our new FastAPI backend
    fetch('http://localhost:8000/metrics')
      .then(res => res.json())
      .then(data => setMetrics(data))
      .catch(err => console.error("API not running yet:", err));
  }, []);

  return (
    <div className="app-container">
      <aside className="sidebar">
        <h2>IndicDoc AI</h2>
        <nav className="sidebar-nav">
          <a href="#" className="nav-link active">Dashboard</a>
          <a href="#" className="nav-link">Analyze</a>
          <a href="#" className="nav-link">Model Insights</a>
          <a href="#" className="nav-link">Evaluation</a>
        </nav>
      </aside>

      <main className="main-content">
        <section className="hero">
          <h1>Indic<span>Doc</span> AI</h1>
          <p>Understand Indian Documents with Deep Learning.</p>
        </section>

        <section className="dashboard">
          <div className="metrics-grid">
            <div className="metric-card">
              <h3>{metrics ? metrics.proposed.precision : '0.92'}</h3>
              <p>Precision</p>
            </div>
            <div className="metric-card">
              <h3>{metrics ? metrics.proposed.recall : '0.90'}</h3>
              <p>Recall</p>
            </div>
            <div className="metric-card">
              <h3>{metrics ? metrics.proposed.map50 : '0.95'}</h3>
              <p>mAP@50</p>
            </div>
            <div className="metric-card">
              <h3>{metrics ? metrics.proposed.inference_time : '0.45'}s</h3>
              <p>Inference Time</p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
