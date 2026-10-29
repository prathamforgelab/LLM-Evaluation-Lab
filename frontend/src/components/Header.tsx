import React from 'react';
import {
  Activity,
  BarChart3,
  Database,
  FileText,
  GitCompare,
  PlusCircle,
  Wifi,
  WifiOff,
} from 'lucide-react';

interface HeaderProps {
  activeTab: 'experiments' | 'new-eval' | 'compare' | 'datasets' | 'reports';
  setActiveTab: (tab: 'experiments' | 'new-eval' | 'compare' | 'datasets' | 'reports') => void;
  apiConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab, apiConnected }) => {
  return (
    <header className="glass-panel" style={{ margin: '16px 24px', padding: '14px 24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
        {/* Brand */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div
            style={{
              width: 38,
              height: 38,
              borderRadius: 'var(--radius-md)',
              background: 'linear-gradient(135deg, #06b6d4, #3b82f6)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              boxShadow: '0 0 15px rgba(6, 182, 212, 0.4)',
            }}
          >
            <BarChart3 size={22} color="#fff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <h1 style={{ fontSize: '1.15rem', fontWeight: 700, letterSpacing: '-0.02em', color: '#fff' }}>
                LLM Evaluation Lab
              </h1>
              <span className="badge" style={{ background: 'rgba(6, 182, 212, 0.1)', color: '#22d3ee', border: '1px solid rgba(6, 182, 212, 0.25)', fontSize: '0.65rem' }}>
                v0.1.0
              </span>
            </div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Scientific AI Benchmarking & Regression Detection
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: 6, background: 'rgba(0, 0, 0, 0.3)', padding: 4, borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
          <button
            onClick={() => setActiveTab('experiments')}
            className="btn btn-sm"
            style={{
              background: activeTab === 'experiments' ? 'var(--bg-card)' : 'transparent',
              color: activeTab === 'experiments' ? 'var(--text-primary)' : 'var(--text-muted)',
              border: activeTab === 'experiments' ? '1px solid var(--border)' : '1px solid transparent',
              boxShadow: activeTab === 'experiments' ? 'var(--shadow-sm)' : 'none',
            }}
          >
            <Activity size={15} />
            Experiments
          </button>

          <button
            onClick={() => setActiveTab('new-eval')}
            className="btn btn-sm"
            style={{
              background: activeTab === 'new-eval' ? 'var(--bg-card)' : 'transparent',
              color: activeTab === 'new-eval' ? 'var(--text-primary)' : 'var(--text-muted)',
              border: activeTab === 'new-eval' ? '1px solid var(--border)' : '1px solid transparent',
              boxShadow: activeTab === 'new-eval' ? 'var(--shadow-sm)' : 'none',
            }}
          >
            <PlusCircle size={15} />
            New Run
          </button>

          <button
            onClick={() => setActiveTab('compare')}
            className="btn btn-sm"
            style={{
              background: activeTab === 'compare' ? 'var(--bg-card)' : 'transparent',
              color: activeTab === 'compare' ? 'var(--text-primary)' : 'var(--text-muted)',
              border: activeTab === 'compare' ? '1px solid var(--border)' : '1px solid transparent',
              boxShadow: activeTab === 'compare' ? 'var(--shadow-sm)' : 'none',
            }}
          >
            <GitCompare size={15} />
            Statistical Compare
          </button>

          <button
            onClick={() => setActiveTab('datasets')}
            className="btn btn-sm"
            style={{
              background: activeTab === 'datasets' ? 'var(--bg-card)' : 'transparent',
              color: activeTab === 'datasets' ? 'var(--text-primary)' : 'var(--text-muted)',
              border: activeTab === 'datasets' ? '1px solid var(--border)' : '1px solid transparent',
              boxShadow: activeTab === 'datasets' ? 'var(--shadow-sm)' : 'none',
            }}
          >
            <Database size={15} />
            Datasets
          </button>

          <button
            onClick={() => setActiveTab('reports')}
            className="btn btn-sm"
            style={{
              background: activeTab === 'reports' ? 'var(--bg-card)' : 'transparent',
              color: activeTab === 'reports' ? 'var(--text-primary)' : 'var(--text-muted)',
              border: activeTab === 'reports' ? '1px solid var(--border)' : '1px solid transparent',
              boxShadow: activeTab === 'reports' ? 'var(--shadow-sm)' : 'none',
            }}
          >
            <FileText size={15} />
            Reports
          </button>
        </nav>

        {/* Backend Status indicator */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div
            title={apiConnected ? 'Connected to FastAPI backend (http://127.0.0.1:8000)' : 'Backend offline — Running in local interactive demo mode'}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              fontSize: '0.75rem',
              padding: '4px 10px',
              borderRadius: 'var(--radius-full)',
              background: apiConnected ? 'rgba(16, 185, 129, 0.1)' : 'rgba(245, 158, 11, 0.1)',
              color: apiConnected ? '#34d399' : '#fbbf24',
              border: `1px solid ${apiConnected ? 'rgba(16, 185, 129, 0.25)' : 'rgba(245, 158, 11, 0.25)'}`,
            }}
          >
            {apiConnected ? <Wifi size={13} /> : <WifiOff size={13} />}
            <span>{apiConnected ? 'Backend Live' : 'Demo / Standalone'}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
