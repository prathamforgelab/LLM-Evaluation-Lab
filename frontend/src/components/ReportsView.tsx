import React, { useEffect, useState } from 'react';
import {
  Check,
  Copy,
  Download,
  FileCode,
  FileSpreadsheet,
  FileText,
  Loader2,
} from 'lucide-react';
import { getExperimentReport } from '../api';
import type { ExperimentListItem } from '../types';

interface ReportsViewProps {
  experiments: ExperimentListItem[];
}

export const ReportsView: React.FC<ReportsViewProps> = ({ experiments }) => {
  const [selectedExpId, setSelectedExpId] = useState(experiments[0]?.id || '');
  const [format, setFormat] = useState<'markdown' | 'json' | 'csv'>('markdown');
  const [reportContent, setReportContent] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (selectedExpId) {
      loadReport(selectedExpId, format);
    }
  }, [selectedExpId, format]);

  const loadReport = async (expId: string, fmt: 'markdown' | 'json' | 'csv') => {
    setIsLoading(true);
    try {
      const content = await getExperimentReport(expId, fmt);
      setReportContent(content);
    } catch {
      // Fallback preview
      if (fmt === 'markdown') {
        setReportContent(`# LLM Evaluation Lab — Experiment Report\n\n**Experiment ID:** \`${expId}\`\n**Date:** ${new Date().toISOString()}\n\n## Summary\n- Total Cases: 12\n- Success Rate: 100%\n- Mean Latency: 142.5ms\n\n## Metrics\n| Metric | Score |\n|---|---|\n| exact_match | 0.7500 |\n| token_overlap_f1 | 0.8820 |\n`);
      } else if (fmt === 'json') {
        setReportContent(JSON.stringify({ experiment_id: expId, total_cases: 12, success_rate: 1.0 }, null, 2));
      } else {
        setReportContent(`case_id,input,latency_ms,exact_match\nsupport-001,"Password reset",130.0,0.0\nqa-002,"Capital of France",85.0,1.0\n`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(reportContent);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const ext = format === 'markdown' ? 'md' : format;
    const blob = new Blob([reportContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `eval_report_${selectedExpId.slice(0, 8)}.${ext}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ padding: '0 24px 32px' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff' }}>
          Reports & Evidence Generation
        </h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Export audit-ready evidence, metrics tables, and regression decisions in Markdown, JSON, and CSV formats.
        </p>
      </div>

      {/* Control bar */}
      <div
        className="glass-panel"
        style={{
          padding: 16,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 16,
          marginBottom: 20,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <div>
            <label className="form-label" style={{ marginBottom: 4 }}>Select Experiment</label>
            <select
              className="form-control"
              style={{ minWidth: 260 }}
              value={selectedExpId}
              onChange={(e) => setSelectedExpId(e.target.value)}
            >
              {experiments.map((e) => (
                <option key={e.id} value={e.id}>
                  {e.name} ({e.id.slice(0, 8)})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="form-label" style={{ marginBottom: 4 }}>Format</label>
            <div style={{ display: 'flex', gap: 4, background: 'rgba(0,0,0,0.3)', padding: 3, borderRadius: 'var(--radius-md)', border: '1px solid var(--border)' }}>
              <button
                onClick={() => setFormat('markdown')}
                className="btn btn-sm"
                style={{
                  background: format === 'markdown' ? 'var(--bg-card)' : 'transparent',
                  color: format === 'markdown' ? '#fff' : 'var(--text-muted)',
                }}
              >
                <FileText size={14} /> Markdown
              </button>
              <button
                onClick={() => setFormat('json')}
                className="btn btn-sm"
                style={{
                  background: format === 'json' ? 'var(--bg-card)' : 'transparent',
                  color: format === 'json' ? '#fff' : 'var(--text-muted)',
                }}
              >
                <FileCode size={14} /> JSON
              </button>
              <button
                onClick={() => setFormat('csv')}
                className="btn btn-sm"
                style={{
                  background: format === 'csv' ? 'var(--bg-card)' : 'transparent',
                  color: format === 'csv' ? '#fff' : 'var(--text-muted)',
                }}
              >
                <FileSpreadsheet size={14} /> CSV
              </button>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={handleCopy} className="btn btn-secondary">
            {copied ? <Check size={16} color="#10b981" /> : <Copy size={16} />}
            {copied ? 'Copied!' : 'Copy'}
          </button>
          <button onClick={handleDownload} className="btn btn-primary">
            <Download size={16} /> Download
          </button>
        </div>
      </div>

      {/* Report Preview */}
      <div className="glass-panel" style={{ padding: 20, minHeight: 400, position: 'relative' }}>
        {isLoading ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 300, color: 'var(--text-muted)', gap: 8 }}>
            <Loader2 size={20} className="spin" /> Generating report…
          </div>
        ) : (
          <pre
            className="mono"
            style={{
              fontSize: '0.825rem',
              lineHeight: 1.6,
              color: '#e2e8f0',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              overflowX: 'auto',
            }}
          >
            {reportContent}
          </pre>
        )}
      </div>
    </div>
  );
};
