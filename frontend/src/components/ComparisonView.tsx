import React, { useState } from 'react';
import {
  AlertTriangle,
  CheckCircle,
  GitCompare,
  HelpCircle,
  Info,
  Loader2,
  TrendingDown,
  TrendingUp,
} from 'lucide-react';
import { compareExperiments } from '../api';
import type { ComparisonReport, ExperimentListItem, RegressionDecision } from '../types';

interface ComparisonViewProps {
  experiments: ExperimentListItem[];
  initialBaselineId?: string;
  initialCandidateId?: string;
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({
  experiments,
  initialBaselineId,
  initialCandidateId,
}) => {
  const [baselineId, setBaselineId] = useState(
    initialBaselineId || (experiments.length > 1 ? experiments[1].id : experiments[0]?.id || '')
  );
  const [candidateId, setCandidateId] = useState(
    initialCandidateId || (experiments.length > 0 ? experiments[0].id : '')
  );

  const [isLoading, setIsLoading] = useState(false);
  const [report, setReport] = useState<ComparisonReport | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleCompare = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!baselineId || !candidateId) return;
    if (baselineId === candidateId) {
      setErrorMsg('Baseline and candidate must be distinct experiment runs.');
      return;
    }

    setIsLoading(true);
    setErrorMsg(null);
    try {
      const data = await compareExperiments(baselineId, candidateId);
      setReport(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'Comparison failed');
    } finally {
      setIsLoading(false);
    }
  };

  const getDecisionBadge = (decision: RegressionDecision) => {
    switch (decision) {
      case 'IMPROVED':
        return (
          <span className="badge badge-improved">
            <CheckCircle size={12} /> IMPROVED
          </span>
        );
      case 'REGRESSED':
        return (
          <span className="badge badge-regressed">
            <AlertTriangle size={12} /> REGRESSED
          </span>
        );
      case 'INCONCLUSIVE':
        return (
          <span className="badge badge-inconclusive">
            <HelpCircle size={12} /> INCONCLUSIVE
          </span>
        );
      case 'INSUFFICIENT_DATA':
        return (
          <span className="badge badge-insufficient">
            <Info size={12} /> INSUFFICIENT DATA
          </span>
        );
    }
  };

  return (
    <div style={{ padding: '0 24px 32px' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff' }}>
          Statistical Regression Detection
        </h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Paired bootstrap resampling to measure empirical performance delta with 95% confidence intervals.
        </p>
      </div>

      {/* Comparison Selector Form */}
      <form onSubmit={handleCompare} className="glass-panel" style={{ padding: 20, marginBottom: 24 }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr auto', gap: 16, alignItems: 'flex-end' }}>
          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label">Baseline Experiment (Control)</label>
            <select
              className="form-control"
              value={baselineId}
              onChange={(e) => setBaselineId(e.target.value)}
              required
            >
              <option value="">Select baseline…</option>
              {experiments.map((exp) => (
                <option key={exp.id} value={exp.id}>
                  {exp.name} ({exp.id.slice(0, 8)})
                </option>
              ))}
            </select>
          </div>

          <div className="form-group" style={{ margin: 0 }}>
            <label className="form-label">Candidate Experiment (Treatment)</label>
            <select
              className="form-control"
              value={candidateId}
              onChange={(e) => setCandidateId(e.target.value)}
              required
            >
              <option value="">Select candidate…</option>
              {experiments.map((exp) => (
                <option key={exp.id} value={exp.id}>
                  {exp.name} ({exp.id.slice(0, 8)})
                </option>
              ))}
            </select>
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            style={{ height: 42 }}
            disabled={isLoading || !baselineId || !candidateId}
          >
            {isLoading ? (
              <>
                <Loader2 size={16} /> Computing Resampling…
              </>
            ) : (
              <>
                <GitCompare size={16} /> Run Comparison
              </>
            )}
          </button>
        </div>

        {errorMsg && (
          <div
            style={{
              padding: '10px 14px',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(244, 63, 94, 0.1)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              color: '#fb7185',
              fontSize: '0.825rem',
              marginTop: 14,
            }}
          >
            {errorMsg}
          </div>
        )}
      </form>

      {/* Comparison Results */}
      {report && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Decision Banner */}
          <div
            className="glass-panel"
            style={{
              padding: 24,
              borderLeftWidth: 6,
              borderLeftColor:
                report.overall_decision === 'IMPROVED'
                  ? '#10b981'
                  : report.overall_decision === 'REGRESSED'
                  ? '#f43f5e'
                  : '#f59e0b',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                    Overall Decision:
                  </span>
                  {getDecisionBadge(report.overall_decision)}
                </div>
                <h3 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#fff', marginTop: 6 }}>
                  {report.overall_decision === 'IMPROVED' && 'Candidate demonstrates statistically significant gains.'}
                  {report.overall_decision === 'REGRESSED' && 'Candidate shows statistically significant regressions.'}
                  {report.overall_decision === 'INCONCLUSIVE' && 'Difference does not clear 95% confidence bounds (crosses zero).'}
                  {report.overall_decision === 'INSUFFICIENT_DATA' && 'Insufficient paired samples to draw conclusion.'}
                </h3>
              </div>

              <div style={{ display: 'flex', gap: 20, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                <div>
                  Bootstrap Resamples: <strong style={{ color: '#fff' }}>{report.bootstrap_n}</strong>
                </div>
                <div>
                  Confidence: <strong style={{ color: '#fff' }}>{(report.confidence_level * 100).toFixed(0)}%</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Metric Comparison Table */}
          <div className="table-container glass-panel">
            <table className="eval-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Baseline</th>
                  <th>Candidate</th>
                  <th>Absolute Diff</th>
                  <th>Relative %</th>
                  <th style={{ minWidth: 160 }}>95% Bootstrap CI</th>
                  <th style={{ textAlign: 'center' }}>Decision</th>
                </tr>
              </thead>
              <tbody>
                {report.metric_comparisons.map((m) => {
                  const isPositive = (m.absolute_change || 0) > 0;
                  const isNegative = (m.absolute_change || 0) < 0;
                  const diffColor = isPositive ? '#34d399' : isNegative ? '#fb7185' : 'var(--text-muted)';

                  return (
                    <tr key={m.metric_name}>
                      <td style={{ fontWeight: 600, color: '#fff' }}>{m.metric_name}</td>
                      <td>{m.baseline_mean !== null && m.baseline_mean !== undefined ? m.baseline_mean.toFixed(4) : '—'}</td>
                      <td>{m.candidate_mean !== null && m.candidate_mean !== undefined ? m.candidate_mean.toFixed(4) : '—'}</td>
                      <td style={{ color: diffColor, fontWeight: 600 }}>
                        {m.absolute_change !== null && m.absolute_change !== undefined ? (
                          <>
                            {isPositive ? '+' : ''}
                            {m.absolute_change.toFixed(4)}
                          </>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td style={{ color: diffColor }}>
                        {m.relative_change_pct !== null && m.relative_change_pct !== undefined ? (
                          <>
                            {isPositive ? '+' : ''}
                            {m.relative_change_pct.toFixed(2)}%
                          </>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td>
                        {m.ci ? (
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                            <span className="mono" style={{ fontSize: '0.75rem' }}>
                              [{m.ci.lower.toFixed(4)}, {m.ci.upper.toFixed(4)}]
                            </span>
                            {/* Visual mini bar */}
                            <div className="ci-viz">
                              <div className="ci-zero-line" />
                              <div
                                className="ci-range-bar"
                                style={{
                                  left: `${Math.max(5, Math.min(90, 50 + m.ci.lower * 50))}%`,
                                  width: `${Math.max(6, Math.min(90, (m.ci.upper - m.ci.lower) * 50))}%`,
                                  background: m.ci.lower > 0 ? '#10b981' : m.ci.upper < 0 ? '#f43f5e' : '#f59e0b',
                                }}
                              />
                            </div>
                          </div>
                        ) : (
                          'N/A'
                        )}
                      </td>
                      <td style={{ textAlign: 'center' }}>{getDecisionBadge(m.decision)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* Cases Regressed / Improved Drill-down */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {/* Cases Regressed */}
            <div className="glass-panel" style={{ padding: 18 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#f43f5e', marginBottom: 12 }}>
                <TrendingDown size={18} />
                <h4 style={{ fontSize: '0.95rem', fontWeight: 600 }}>
                  Regressed Test Cases ({report.cases_regressed.length})
                </h4>
              </div>
              {report.cases_regressed.length === 0 ? (
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  No individual cases regressed in this comparison.
                </p>
              ) : (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {report.cases_regressed.map((cid) => (
                    <span
                      key={cid}
                      className="mono"
                      style={{
                        padding: '3px 8px',
                        background: 'rgba(244, 63, 94, 0.12)',
                        border: '1px solid rgba(244, 63, 94, 0.25)',
                        borderRadius: 4,
                        fontSize: '0.75rem',
                        color: '#fca5a5',
                      }}
                    >
                      {cid}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Cases Improved */}
            <div className="glass-panel" style={{ padding: 18 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#10b981', marginBottom: 12 }}>
                <TrendingUp size={18} />
                <h4 style={{ fontSize: '0.95rem', fontWeight: 600 }}>
                  Improved Test Cases ({report.cases_improved.length})
                </h4>
              </div>
              {report.cases_improved.length === 0 ? (
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  No individual cases showed distinct metric improvements.
                </p>
              ) : (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {report.cases_improved.map((cid) => (
                    <span
                      key={cid}
                      className="mono"
                      style={{
                        padding: '3px 8px',
                        background: 'rgba(16, 185, 129, 0.12)',
                        border: '1px solid rgba(16, 185, 129, 0.25)',
                        borderRadius: 4,
                        fontSize: '0.75rem',
                        color: '#86efac',
                      }}
                    >
                      {cid}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
