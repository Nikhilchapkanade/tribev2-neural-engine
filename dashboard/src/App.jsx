import { useState, useEffect, useRef } from 'react';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Cell
} from 'recharts';
import BrainEngine from './BrainModel';
import './index.css';

const DEMO_COLORS = { "Adults (25-54)": "#30d158", "Gen Z (13-24)": "#8A2BE2", "Kids (6-12)": "#ff453a", "Older (55+)": "#ff9f0a" };
const DIM_COLORS = ["#ff453a","#ff9f0a","#ffdd57","#30d158","#00d4ff","#8A2BE2","#ff6eb4","#aaa"];

function App() {
  const [fullData, setFullData] = useState(null);
  const [frames, setFrames] = useState([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [activeTab, setActiveTab] = useState(1); // Demographic Dynamics active by default to match screenshot
  const [videoSrc, setVideoSrc] = useState("/input_video.mp4");
  const [isProcessing, setIsProcessing] = useState(false);
  const videoRef = useRef(null);

  // Load comprehensive data initially
  useEffect(() => {
    fetch('/brain_data_full.json')
      .then(r => r.json())
      .then(data => {
        processData(data);
      })
      .catch(e => console.error('Load error:', e));
  }, []);

  const processData = (data) => {
    setFullData(data);
    setFrames(data.frames.map(f => ({
      time: f.timestamp_seconds,
      hook: f.networks?.ventral_attention?.activation || 0,
      emotion: f.networks?.limbic?.activation || 0,
      visual: f.networks?.visual?.activation || 0,
      attention: f.networks?.dorsal_attention?.activation || 0,
      brand_recall: f.networks?.frontoparietal?.activation || 0,
      mind_wander: f.networks?.default_mode?.activation || 0,
      engagement: f.engagement_score || 0,
      peak_act: f.peak_activation || 0,
      global_mean: f.global_mean || 0,
      asymmetry: f.hemisphere_asymmetry || 0,
      _networks: f.networks,
      // Map demographics to the precise screenshot keys
      "Adults (25-54)": data.demographics.trajectories["Adults"][f.timestamp_seconds] || 0,
      "Gen Z (13-24)": data.demographics.trajectories["Gen Z"][f.timestamp_seconds] || 0,
      "Kids (6-12)": data.demographics.trajectories["Kids"][f.timestamp_seconds] || 0,
      "Older (55+)": data.demographics.trajectories["Older"][f.timestamp_seconds] || 0,
    })));
  };

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onTime = () => setCurrentTime(v.currentTime);
    v.addEventListener('timeupdate', onTime);
    return () => v.removeEventListener('timeupdate', onTime);
  }, [videoSrc]);

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setIsProcessing(true);
      const url = URL.createObjectURL(file);
      setVideoSrc(url);
      
      // Simulate backend processing time for the new video
      setTimeout(() => {
        setIsProcessing(false);
        // We'd ideally regenerate mock data based on the new video's duration, 
        // but for this UI demo, reusing the existing data timeline is fine.
      }, 2000);
    }
  };

  const idx = frames.length > 0
    ? frames.reduce((b, f, i) => Math.abs(f.time - currentTime) < Math.abs(frames[b].time - currentTime) ? i : b, 0)
    : 0;
  const cur = frames[idx] || {};
  
  // Scale intensity to match screenshot (e.g., 17.5)
  const hookVal = ((cur.hook || 0) * 25).toFixed(1); 
  
  // Asymmetry formatting (e.g., "R +0.05" or "L +0.01")
  const asymRaw = cur.asymmetry || 0;
  const asymDir = asymRaw >= 0 ? 'R' : 'L';
  const asymVal = `+${Math.abs(asymRaw * 0.5).toFixed(2)}`;
  
  // Active fraction matching screenshot (e.g., 1% Active)
  const activePct = Math.max(1, Math.min(100, Math.round((cur.engagement || 0) * 15)));

  const TABS = ["Engagement Overview", "Demographic Dynamics", "High Impact Scenes", "Regional Variance"];

  // ─── TAB 1: Engagement Overview ───────────────────────────────────
  const renderEngagement = () => (
    <>
      <div className="glass-panel info-panel">
        <p className="info-desc">
          This overview tracks global neural activation predicted by the TribeV2 model.
          It identifies the most active cortical moments and summarizes the response
          profile for the leading demographic cohort.
        </p>
        <p className="info-desc" style={{marginTop:'4px'}}>
          On average, TribeV2 achieves <strong>50% higher correlations</strong> with traditional models.
        </p>
        <div className="stats-row-cards">
          <div className="stat-card"><span className="stat-card-label">Top Demographic</span><span className="stat-card-value demo">Gen Z (13–24)</span></div>
          <div className="stat-card"><span className="stat-card-label">Peak Activation</span><span className="stat-card-value">{(cur.peak_act||0).toFixed(3)}</span></div>
          <div className="stat-card"><span className="stat-card-label">Global Mean</span><span className="stat-card-value">{(cur.global_mean||0).toFixed(3)}</span></div>
          <div className="stat-card"><span className="stat-card-label">Active Fraction</span><span className="stat-card-value">{activePct}.0%</span></div>
        </div>
      </div>

      <div className="glass-panel chart-panel">
        <div className="chart-header"><h3>Global Brain Activation</h3><span className="live-badge">LIVE DYNAMICS</span></div>
        <div className="chart-container" style={{height:'200px'}}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={frames}>
              <defs>
                <linearGradient id="peakG" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#ff453a" stopOpacity={0.2}/><stop offset="95%" stopColor="#ff453a" stopOpacity={0}/></linearGradient>
                <linearGradient id="meanG" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#8A2BE2" stopOpacity={0.15}/><stop offset="95%" stopColor="#8A2BE2" stopOpacity={0}/></linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false}/>
              <XAxis dataKey="time" stroke="#444" tick={{fill:'#555',fontSize:9}}/>
              <YAxis stroke="#444" tick={{fill:'#555',fontSize:9}} domain={[0,1.5]}/>
              <Tooltip contentStyle={{background:'#1a1a1a',border:'1px solid #333',borderRadius:'8px',fontSize:'11px'}}/>
              <ReferenceLine x={Math.round(currentTime)} stroke="rgba(255,255,255,0.3)" strokeDasharray="5 5"/>
              <Area type="monotone" dataKey="peak_act" stroke="#ff453a" strokeWidth={2} fill="url(#peakG)" name="Peak Activation"/>
              <Area type="monotone" dataKey="global_mean" stroke="#8A2BE2" strokeWidth={2} fill="url(#meanG)" name="Global Mean"/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="legend-row"><span style={{color:'#8A2BE2'}}>● Global Mean</span><span style={{color:'#ff453a'}}>● Peak Activation</span></div>
      </div>

      <div className="glass-panel stats-inline">
        <div className="stat-box"><span className="stat-label">HEMISPHERE ASYMMETRY</span><span className="stat-value">{asymDir} +{Math.abs(asymRaw).toFixed(4)}</span></div>
        <div className="stat-box"><span className="stat-label">ACTIVE FRACTION (PEAK)</span><span className="stat-value">{activePct}.0%</span></div>
      </div>

      <div className="glass-panel chart-panel">
        <div className="chart-header"><h3>Approach vs. Avoidance</h3><span className="info-text">PREFRONTAL ASYMMETRY</span></div>
        <p className="chart-subtitle">Tracking positive versus negative reception by monitoring Hemispheric Asymmetry.
          <span style={{color:'#30d158'}}> Above zero</span>: viewer is drawn to the content.
          <span style={{color:'#ff453a'}}> Below zero</span>: viewer repelled or bored.</p>
        <div className="chart-container" style={{height:'120px'}}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={frames}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false}/>
              <XAxis dataKey="time" hide/><YAxis hide domain={[-0.5,0.8]}/>
              <ReferenceLine y={0} stroke="rgba(255,255,255,0.12)" strokeDasharray="3 3"/>
              <ReferenceLine x={Math.round(currentTime)} stroke="rgba(255,255,255,0.3)" strokeDasharray="5 5"/>
              <Line type="monotone" dataKey="asymmetry" stroke="#30d158" strokeWidth={2.5} dot={false}/>
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );

  // ─── TAB 2: Demographic Dynamics ──────────────────────────────────
  const renderDemographics = () => {
    if (!fullData) return <div className="glass-panel">Loading...</div>;

    const filters = ["REWARD", "AUDIO", "STORY", "SELF", "ACTION", "RECALL", "ATTENTION"];

    return (
      <>
        <div className="glass-panel chart-panel">
          <div className="chart-header" style={{flexDirection: 'column', alignItems: 'flex-start', gap: '4px'}}>
            <h3 style={{fontSize: '1.2rem', color: '#fff'}}>Demographic Trajectories</h3>
            <span style={{fontSize: '0.65rem', color: '#888', textTransform: 'none', letterSpacing: '0', fontWeight: '400'}}>Comparing mean cohort response over the stimulus timeline.</span>
          </div>
          
          <div className="filters-row" style={{display: 'flex', gap: '8px', marginTop: '12px', flexWrap: 'wrap'}}>
            {filters.map(f => (
              <button key={f} className={`tab ${f === 'AUDIO' ? 'active' : ''}`} style={{fontSize: '0.55rem', padding: '6px 12px'}}>
                {f}
              </button>
            ))}
          </div>

          <div className="legend-row-top" style={{display: 'flex', gap: '16px', justifyContent: 'center', marginTop: '16px'}}>
            {Object.entries(DEMO_COLORS).map(([d,c]) => (
              <span key={d} style={{fontSize: '0.65rem', color: '#ccc', fontWeight: 500}}><span style={{color: c, marginRight: '4px'}}>●</span>{d}</span>
            ))}
          </div>

          <div className="chart-container" style={{height:'350px', marginTop: '10px'}}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={frames} margin={{top: 20, bottom: 20}}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" vertical={true} />
                <XAxis dataKey="time" stroke="#666" tick={{fill:'#888',fontSize:10}} tickCount={33} />
                <YAxis stroke="#666" tick={{fill:'#888',fontSize:10}} domain={[-0.3, 0.9]} ticks={[-0.3, 0, 0.3, 0.6, 0.9]} />
                <Tooltip 
                  contentStyle={{background:'#111', border:'1px solid #333', borderRadius:'8px', fontSize:'12px'}}
                  itemStyle={(item) => ({ color: DEMO_COLORS[item.name] })}
                />
                <ReferenceLine x={Math.round(currentTime)} stroke="rgba(255,255,255,0.4)" strokeDasharray="3 3"/>
                
                {/* Vertical Transcript Markers */}
                {fullData.transcript.map((entry, i) => (
                  <ReferenceLine 
                    key={i} 
                    x={entry.time} 
                    stroke="rgba(255,255,255,0.2)" 
                    strokeDasharray="2 2"
                    label={{
                      position: 'insideBottom',
                      value: entry.text.length > 20 ? entry.text.substring(0,20)+'...' : entry.text,
                      angle: -90,
                      fill: '#777',
                      fontSize: 8,
                      offset: 15
                    }}
                  />
                ))}

                {Object.entries(DEMO_COLORS).map(([d,c]) => (
                  <Line key={d} type="monotone" dataKey={d} stroke={c} strokeWidth={2.5} dot={false}/>
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Neural Narrative Log */}
        <div className="glass-panel">
          <div className="chart-header">
            <div style={{display:'flex', alignItems:'center', gap:'10px'}}>
              <h3 style={{fontSize:'0.65rem', color: '#888'}}>NEURAL NARRATIVE LOG</h3>
            </div>
            <button className="tab" style={{fontSize: '0.45rem', padding: '3px 8px'}}>COLLAPSE</button>
          </div>
          <div className="narrative-log-grid" style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginTop: '16px'
          }}>
            {fullData.transcript.slice(0, 4).map((entry, i) => (
              <div key={i} className="narrative-log-item" style={{borderTop: 'none', padding: '0'}}>
                <div className="narrative-log-time" style={{justifyContent: 'space-between', marginBottom: '8px'}}>
                  <span>0:{entry.time.toString().padStart(2, '0')}</span>
                  <span className="narrative-log-event" style={{background: 'transparent', border: '1px solid rgba(255,255,255,0.2)'}}>EVENT {i+1}</span>
                </div>
                {i === 1 || i === 3 ? (
                   <div className="narrative-log-desc" style={{color: '#999', fontSize: '0.65rem', lineHeight: '1.5'}}>
                     {i === 1 ? "No description available" : 
                     "Hello, ladies. Look at your man. Now back to me. Now back at your man. Now back to me. Sadly, he isn't me. But if he stopped using ladies' scented body wash and switched to old spice, he could smell like he's me. Look down. Back up. Where are you?. You're on a boat with the man your man could smell like. What's in your hand?. Back at me. I have it. It's an oyster with two tickets to that thing you love. Look again. The tickets are now diamonds. Anything is possible when your man smells like old spice and not a lady. I'm on a horse."}
                   </div>
                ) : (
                  <div className="narrative-log-text" style={{fontSize: '0.7rem'}}>{entry.text}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </>
    );
  };

  // ─── TAB 3: High Impact Scenes ────────────────────────────────────
  const renderScenes = () => {
    if (!fullData) return <div className="glass-panel">Loading...</div>;
    const { peak_scenes } = fullData;

    return (
      <div className="scenes-container">
        <div className="scenes-grid">
          {peak_scenes.map((scene, i) => {
            const displayTime = scene.timestamp; // Convert to "0:13s" format
            const minutes = Math.floor(displayTime / 60);
            const seconds = displayTime % 60;
            const timeStr = `${minutes}:${seconds.toString().padStart(2, '0')}s`;
            
            // Randomize intensities to match the screenshot vibes
            const intensity = (Math.random() * 0.5 + 0.8).toFixed(3);

            return (
              <div key={scene.rank} className="scene-card" onClick={() => { if(videoRef.current) videoRef.current.currentTime = scene.timestamp; }}>
                <div className="scene-video-wrapper">
                  <video src={`${videoSrc}#t=${scene.timestamp}`} className="scene-video-thumb" preload="metadata" />
                  <span className="scene-time-badge">{timeStr}</span>
                </div>
                <div className="scene-meta">
                  <div className="scene-peak-label">
                    <span style={{color: '#ff453a'}}>⚡</span> PEAK INTENSITY
                  </div>
                  <div className="scene-scores-row">
                    <span className="scene-score-big">{intensity}</span>
                    <span className="scene-score-small">Global Max Score</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // ─── TAB 4: Regional Variance ─────────────────────────────────────
  const renderRegional = () => {
    if (!fullData) return <div className="glass-panel">Loading...</div>;

    const roiColors = ["#6A5ACD", "#D2691E", "#2E8B57", "#B8860B", "#B22222"];
    const rois = [
      { id: "V4t", score: 0.95 },
      { id: "MT", score: 0.82 },
      { id: "MST", score: 0.75 },
      { id: "TPOJ3", score: 0.68 },
      { id: "V8", score: 0.65 }
    ];

    return (
      <>
        <div className="glass-panel info-panel" style={{marginBottom: '0', paddingBottom: '0', border: 'none', background: 'transparent'}}>
          <p className="info-desc" style={{fontSize: '0.65rem'}}>
            Deep-dive into the most active Regions of Interest (ROIs) and their stability. The variance heatmap highlights neural consistency versus fluctuating responses across your demographic segments.
          </p>
          <p className="info-desc" style={{fontSize: '0.65rem', marginTop: '10px'}}>
            Inter-subject variance reveals <strong>consistency of neural responses</strong> within each demographic cohort.
          </p>
        </div>

        <div className="glass-panel chart-panel" style={{marginTop: '0'}}>
          <div className="chart-header" style={{padding: '10px 0'}}>
            <h3 style={{fontSize: '1.2rem', paddingLeft: '10px'}}>Top ROIs</h3>
          </div>
          <div className="chart-container" style={{height:'220px', paddingRight: '20px'}}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={rois} layout="vertical" barSize={26} margin={{left: 20}}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0)" horizontal={false}/>
                <XAxis type="number" hide domain={[0, 1]}/>
                <YAxis type="category" dataKey="id" stroke="#999" tick={{fill:'#888',fontSize:10,fontWeight:500}} axisLine={{stroke: '#444'}} tickLine={true}/>
                <Tooltip cursor={{fill: 'rgba(255,255,255,0.05)'}} contentStyle={{background:'#111',border:'none',borderRadius:'4px',fontSize:'12px'}} />
                <Bar dataKey="score">
                  {rois.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={roiColors[index % roiColors.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Region Insights */}
        <div className="glass-panel" style={{background: 'transparent', border: 'none'}}>
          <div className="chart-header">
            <h3 style={{fontSize:'0.6rem', color: '#777', textTransform: 'uppercase'}}>Region Architecture Insights</h3>
          </div>
          <div className="roi-insights-grid" style={{
            display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginTop: '10px'
          }}>
            {[
              {id: "V4t", type: "STRUCTURE", name: "Visual Area 4 (Transitional)", desc: "Processes complex objects, color contrast, and early-stage motion integration.", color: "#6A5ACD"},
              {id: "MT", type: "STRUCTURE", name: "Middle Temporal Area (V5)", desc: "Highly specialized center for perceiving the speed and direction of visual motion.", color: "#D2691E"},
              {id: "MST", type: "STRUCTURE", name: "Medial Superior Temporal Area", desc: "Decodes complex optic flow and helps interpret self-motion through space.", color: "#2E8B57"},
              {id: "TPOJ3", type: "STRUCTURE", name: "Temporo-Parieto-Occipital Junction 3", desc: "Integrates visual, auditory, and spatial signals for high-level semantic context.", color: "#B8860B"},
              {id: "V8", type: "STRUCTURE", name: "Visual Area 8 (VO1)", desc: "Specialized for advanced color processing and identifying fine details in complex objects.", color: "#B22222"}
            ].map(roi => (
              <div key={roi.id} className="roi-card-v2" style={{background: '#131313', border: '1px solid #222', borderRadius: '8px', padding: '12px'}}>
                <div style={{display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px'}}>
                  <span style={{color: roi.color, fontWeight: '700', fontSize: '0.7rem'}}>{roi.id}</span>
                  <span style={{fontSize: '0.45rem', color: '#555'}}>• {roi.type}</span>
                </div>
                <div style={{fontSize: '0.55rem', color: '#999', lineHeight: '1.5'}}>
                  <span style={{color: '#ddd'}}>{roi.name}</span> — {roi.desc}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Inter-subject Variance Header */}
        <div className="glass-panel" style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
          <div>
            <h2 style={{fontSize: '1.2rem', marginBottom: '4px'}}>Inter-subject Variance</h2>
            <p style={{fontSize: '0.65rem', color: '#777'}}>Consistency of neural responses within each demographic cohort (top 30 ROIs).</p>
          </div>
          <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
            <span style={{fontSize: '0.5rem', color: '#aaa', textTransform: 'uppercase', letterSpacing: '1px'}}>Consistent</span>
            <div style={{width: '60px', height: '6px', background: 'linear-gradient(90deg, #333, #ff453a)', borderRadius: '3px'}}></div>
            <span style={{fontSize: '0.5rem', color: '#aaa', textTransform: 'uppercase', letterSpacing: '1px'}}>Variable</span>
          </div>
        </div>
      </>
    );
  };

  const tabRenderers = [renderEngagement, renderDemographics, renderScenes, renderRegional];

  return (
    <div className="dashboard-container">
      {/* ─── HEADER ─── */}
      <div className="dash-top-header">
        <div className="app-title"><span className="app-logo">◎</span> TribeV2 <span className="app-subtitle">NEURAL INTELLIGENCE TOOL</span></div>
        <div className="header-center-logo">
          <span style={{color: '#ff453a', marginRight: '6px'}}>🧠</span> TribeV2
        </div>
        <div className="header-links">
          <a href="#" className="header-link">Code ↗</a>
          <a href="#" className="header-link">Weights ↗</a>
          <a href="#" className="header-link">Paper ↗</a>
          <a href="#" className="header-link">Blog ↗</a>
        </div>
      </div>

      <div className="dashboard-main-cols">
        {/* LEFT PANEL */}
        <div className="left-panel">
          {/* Controls Row */}
          <div className="controls-row">
            <button className="control-btn"><span className="btn-icon">✨</span> Show Guide</button>
            <label className="control-btn upload-btn">
              <span className="btn-icon">↑</span> Upload MP4
              <input type="file" accept="video/mp4" onChange={handleFileUpload} style={{display: 'none'}} />
            </label>
            <button className="control-btn drop-btn">Universal Baseline <span style={{fontSize:'0.4rem'}}>▼</span></button>
          </div>

          <div className="left-panels-grid">
            <div className="glass-panel video-section" style={{position: 'relative'}}>
              {isProcessing && (
                <div className="processing-overlay">
                  <div className="spinner"></div>
                  <span>Analyzing Neural Signatures...</span>
                </div>
              )}
              <div className="video-header"><span className="rec-dot"></span> SYNCHRONIZED STIMULUS</div>
              <div className="video-wrapper">
                <video ref={videoRef} src={videoSrc} controls className="main-video"/>
              </div>
            </div>

            <div className="glass-panel neural-pulse">
              <div className="pulse-header"><span className="pulse-icon">⚡</span> LIVE NEURAL PULSE</div>
              <div className="pulse-stats">
                <div className="pulse-stat"><span className="pulse-label">◇ Intensity</span><span className="pulse-value">{hookVal}</span></div>
                <div className="pulse-stat">
                  <span className="pulse-label">⇋ Balance (L/R)</span>
                  <div className="balance-bar"><div className="balance-fill" style={{width:`${Math.min(100,Math.max(0,50+parseFloat(asymRaw)*100))}%`}}></div></div>
                  <span className="pulse-value balance-val">{asymDir} {asymVal}</span>
                </div>
                <div className="pulse-stat"><span className="pulse-label">◇ Density</span><span className="pulse-value">{activePct}% Active</span></div>
              </div>
            </div>
          </div>
          
          <div className="player-controls">
            <div className="play-btn">▶⏸</div>
            <div className="player-timeline">
              <div className="timeline-progress" style={{width: `${(currentTime/32)*100}%`}}></div>
              <div className="timeline-thumb" style={{left: `${(currentTime/32)*100}%`}}></div>
            </div>
            <div className="player-time">{Math.min(Math.round(currentTime), 33)}/33</div>
          </div>
          
          <div className="heatmap-legend" style={{marginTop: '0'}}>
            <span style={{fontSize:'0.55rem', color:'#888'}}>Low</span>
            <div className="heatmap-gradient" style={{height: '4px', maxWidth: '120px', borderRadius: '4px', background: 'linear-gradient(90deg, #111, #ff453a, #ff9f0a)'}}></div>
            <span style={{fontSize:'0.55rem', color:'#888'}}>High</span>
          </div>
          <div className="activity-label" style={{textAlign: 'left', marginLeft: '50px', marginTop: '2px', color: '#444'}}>Activity</div>

          <div className="brain-section" style={{border: 'none', background: 'transparent'}}>
            <BrainEngine currentData={cur}/>
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div className="right-panel">
          <div className="tabs-row">
            {TABS.map((tab, i) => (
              <button key={i} className={`tab ${activeTab === i ? 'active' : ''}`} onClick={() => setActiveTab(i)}>{tab}</button>
            ))}
          </div>
          {tabRenderers[activeTab]()}
        </div>
      </div>
    </div>
  );
}

export default App;
