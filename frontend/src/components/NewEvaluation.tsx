import React, { useState } from 'react';
import {
  Loader2,
  Play,
} from 'lucide-react';
import { runExperiment } from '../api';

interface NewEvaluationProps {
  onEvaluationCompleted: () => void;
  onCancel: () => void;
}

export const NewEvaluation: React.FC<NewEvaluationProps> = ({
  onEvaluationCompleted,
  onCancel,
}) => {
  const [name, setName] = useState('Experiment ' + new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
  const [datasetPath, setDatasetPath] = useState('examples/sample_dataset.jsonl');
  const [provider, setProvider] = useState<'mock' | 'openai_compatible'>('mock');
  const [model, setModel] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('You are an expert, precise AI assistant. Answer the user prompt directly and accurately.');
  const [userTemplate, setUserTemplate] = useState('{input}');
  const [exactMatch, setExactMatch] = useState(true);
  const [tokenF1, setTokenF1] = useState(true);
  const [jsonValidity, setJsonValidity] = useState(false);

  const [isRunning, setIsRunning] = useState(false);
  const [progressText, setProgressText] = useState('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleLaunch = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsRunning(true);
    setErrorMsg(null);
    setProgressText('Submitting configuration to runner…');

    try {
      setProgressText('Evaluating dataset with selected providers and metrics…');
      await runExperiment({
        name,
        dataset_path: datasetPath,
        provider_name: provider,
        model,
        system_prompt: systemPrompt,
        user_template: userTemplate,
        exact_match: exactMatch,
        token_f1: tokenF1,
        json_validity: jsonValidity,
      });

      setProgressText('Experiment complete! Indexing results…');
      setTimeout(() => {
        setIsRunning(false);
        onEvaluationCompleted();
      }, 600);
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to execute experiment run');
      setIsRunning(false);
    }
  };

  return (
    <div style={{ padding: '0 24px 32px', maxWidth: 840, margin: '0 auto' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff' }}>
          Configure New Evaluation Run
        </h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Run an evaluation suite against your dataset to record empirical metrics and detect performance changes.
        </p>
      </div>

      <form onSubmit={handleLaunch} className="glass-panel" style={{ padding: 24 }}>
        {errorMsg && (
          <div
            style={{
              padding: '12px 16px',
              borderRadius: 'var(--radius-md)',
              background: 'rgba(244, 63, 94, 0.1)',
              border: '1px solid rgba(244, 63, 94, 0.3)',
              color: '#fb7185',
              fontSize: '0.85rem',
              marginBottom: 20,
            }}
          >
            {errorMsg}
          </div>
        )}

        {/* Experiment Identity */}
        <div className="form-group">
          <label className="form-label">Experiment Name</label>
          <input
            type="text"
            className="form-control"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            disabled={isRunning}
          />
        </div>

        {/* Dataset Selection */}
        <div className="form-group">
          <label className="form-label">Dataset File Path (JSONL)</label>
          <input
            type="text"
            className="form-control"
            value={datasetPath}
            onChange={(e) => setDatasetPath(e.target.value)}
            required
            disabled={isRunning}
          />
          <span style={{ fontSize: '0.725rem', color: 'var(--text-muted)', marginTop: 4 }}>
            Relative path to JSONL file (e.g. <code className="mono">examples/sample_dataset.jsonl</code>)
          </span>
        </div>

        {/* Provider and Model */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <div className="form-group">
            <label className="form-label">Provider</label>
            <select
              className="form-control"
              value={provider}
              onChange={(e) => setProvider(e.target.value as any)}
              disabled={isRunning}
            >
              <option value="mock">Offline Mock Provider (Deterministic & Fast)</option>
              <option value="openai_compatible">OpenAI-Compatible Endpoint</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Model Identifier</label>
            <input
              type="text"
              className="form-control"
              placeholder={provider === 'mock' ? 'mock-default' : 'gpt-4o-mini'}
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isRunning}
            />
          </div>
        </div>

        {/* Prompt Templates */}
        <div className="form-group">
          <label className="form-label">System Prompt</label>
          <textarea
            className="form-control"
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            disabled={isRunning}
            rows={2}
          />
        </div>

        <div className="form-group">
          <label className="form-label">User Prompt Template</label>
          <input
            type="text"
            className="form-control"
            value={userTemplate}
            onChange={(e) => setUserTemplate(e.target.value)}
            disabled={isRunning}
          />
          <span style={{ fontSize: '0.725rem', color: 'var(--text-muted)', marginTop: 4 }}>
            Template placeholder <code className="mono">&#123;input&#125;</code> will be substituted for each test case.
          </span>
        </div>

        {/* Evaluators Selection */}
        <div className="form-group">
          <label className="form-label">Active Evaluators</label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 10 }}>
            <div
              className={`checkbox-card ${exactMatch ? 'active' : ''}`}
              onClick={() => !isRunning && setExactMatch(!exactMatch)}
            >
              <input
                type="checkbox"
                checked={exactMatch}
                onChange={() => {}}
                style={{ accentColor: '#06b6d4' }}
              />
              <div>
                <div style={{ fontSize: '0.825rem', fontWeight: 600 }}>Exact Match</div>
                <div style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>Exact string equality</div>
              </div>
            </div>

            <div
              className={`checkbox-card ${tokenF1 ? 'active' : ''}`}
              onClick={() => !isRunning && setTokenF1(!tokenF1)}
            >
              <input
                type="checkbox"
                checked={tokenF1}
                onChange={() => {}}
                style={{ accentColor: '#06b6d4' }}
              />
              <div>
                <div style={{ fontSize: '0.825rem', fontWeight: 600 }}>Token F1 Score</div>
                <div style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>Precision & recall overlap</div>
              </div>
            </div>

            <div
              className={`checkbox-card ${jsonValidity ? 'active' : ''}`}
              onClick={() => !isRunning && setJsonValidity(!jsonValidity)}
            >
              <input
                type="checkbox"
                checked={jsonValidity}
                onChange={() => {}}
                style={{ accentColor: '#06b6d4' }}
              />
              <div>
                <div style={{ fontSize: '0.825rem', fontWeight: 600 }}>JSON Validity</div>
                <div style={{ fontSize: '0.725rem', color: 'var(--text-muted)' }}>Syntactic JSON parsing</div>
              </div>
            </div>
          </div>
        </div>

        {/* Run progress bar */}
        {isRunning && (
          <div style={{ marginTop: 20, marginBottom: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.85rem', color: 'var(--primary)', marginBottom: 6 }}>
              <Loader2 size={16} className="spin" />
              <span>{progressText}</span>
            </div>
            <div style={{ height: 6, background: 'rgba(255, 255, 255, 0.08)', borderRadius: 3, overflow: 'hidden' }}>
              <div
                style={{
                  height: '100%',
                  width: '75%',
                  background: 'linear-gradient(90deg, #06b6d4, #3b82f6)',
                  borderRadius: 3,
                  animation: 'pulse 1.5s infinite',
                }}
              />
            </div>
          </div>
        )}

        {/* Form Actions */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 12, marginTop: 24, borderTop: '1px solid var(--border)', paddingTop: 16 }}>
          <button
            type="button"
            onClick={onCancel}
            className="btn btn-secondary"
            disabled={isRunning}
          >
            Cancel
          </button>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={isRunning}
          >
            {isRunning ? (
              <>
                <Loader2 size={16} /> Running Experiment…
              </>
            ) : (
              <>
                <Play size={16} /> Run Evaluation
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};
