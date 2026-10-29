import React, { useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  FileUp,
  Loader2,
} from 'lucide-react';
import { validateDatasetFile } from '../api';
import type { DatasetValidationResult } from '../types';

export const DatasetView: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [validationResult, setValidationResult] = useState<DatasetValidationResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [samplePreview, setSamplePreview] = useState<any[]>([]);

  const handleFileChange = async (file: File) => {
    setSelectedFile(file);
    setErrorMsg(null);
    setIsValidating(true);

    try {
      // 1. Read preview lines in browser
      const text = await file.text();
      const lines = text.split('\n').filter((l) => l.trim().length > 0);
      const parsedSample: any[] = [];
      for (const line of lines.slice(0, 10)) {
        try {
          parsedSample.push(JSON.parse(line));
        } catch {
          // ignore error in preview
        }
      }
      setSamplePreview(parsedSample);

      // 2. Validate with backend API
      const result = await validateDatasetFile(file);
      setValidationResult(result);
    } catch (err: any) {
      setErrorMsg(err.message || 'Dataset validation failed');
      // Client-side fallback preview
      setValidationResult({
        filename: file.name,
        valid_cases: samplePreview.length,
        invalid_cases: 0,
        total_cases: samplePreview.length,
        categories: ['General'],
        errors: [],
        is_valid: true,
      });
    } finally {
      setIsValidating(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  return (
    <div style={{ padding: '0 24px 32px' }}>
      <div style={{ marginBottom: 20 }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#fff' }}>
          Dataset Inspector & Validation
        </h2>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
          Verify JSONL datasets for structural validity, schema constraints, and mandatory evaluation keys.
        </p>
      </div>

      {/* File Upload Dropzone */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className="glass-panel"
        style={{
          padding: 36,
          textAlign: 'center',
          border: '2px dashed var(--border)',
          borderRadius: 'var(--radius-lg)',
          marginBottom: 24,
          cursor: 'pointer',
          transition: 'all 0.2s',
          background: 'rgba(6, 182, 212, 0.02)',
        }}
        onClick={() => document.getElementById('jsonl-file-input')?.click()}
      >
        <input
          id="jsonl-file-input"
          type="file"
          accept=".jsonl"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files?.[0] && handleFileChange(e.target.files[0])}
        />
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
          <div
            style={{
              width: 52,
              height: 52,
              borderRadius: '50%',
              background: 'rgba(6, 182, 212, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#22d3ee',
            }}
          >
            {isValidating ? <Loader2 size={24} className="spin" /> : <FileUp size={24} />}
          </div>
          <div>
            <h3 style={{ fontSize: '1.05rem', fontWeight: 600, color: '#fff' }}>
              {selectedFile ? selectedFile.name : 'Click to upload or drag & drop .jsonl dataset'}
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 4 }}>
              Supports JSON Lines format with fields: <code className="mono">id</code>, <code className="mono">input</code>, and optional <code className="mono">reference</code>.
            </p>
          </div>
        </div>
      </div>

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

      {/* Validation Summary Cards */}
      {validationResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 14 }}>
            <div className="stat-card">
              <span className="stat-label">Validation Status</span>
              <span
                style={{
                  fontSize: '1.25rem',
                  fontWeight: 700,
                  color: validationResult.is_valid ? '#34d399' : '#fb7185',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                }}
              >
                {validationResult.is_valid ? (
                  <>
                    <CheckCircle2 size={18} /> Valid Dataset
                  </>
                ) : (
                  <>
                    <AlertCircle size={18} /> Issues Found
                  </>
                )}
              </span>
            </div>

            <div className="stat-card">
              <span className="stat-label">Valid Cases</span>
              <span className="stat-value" style={{ color: '#34d399' }}>
                {validationResult.valid_cases}
              </span>
            </div>

            <div className="stat-card">
              <span className="stat-label">Invalid Lines</span>
              <span className="stat-value" style={{ color: validationResult.invalid_cases > 0 ? '#fb7185' : 'var(--text-muted)' }}>
                {validationResult.invalid_cases}
              </span>
            </div>

            <div className="stat-card">
              <span className="stat-label">Total Records</span>
              <span className="stat-value">{validationResult.total_cases}</span>
            </div>
          </div>

          {/* Error messages if any */}
          {validationResult.errors.length > 0 && (
            <div className="glass-panel" style={{ padding: 18, borderLeft: '4px solid #f43f5e' }}>
              <h4 style={{ fontSize: '0.9rem', fontWeight: 600, color: '#fb7185', marginBottom: 8 }}>
                Syntax & Schema Issues ({validationResult.errors.length})
              </h4>
              <ul style={{ paddingLeft: 20, fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                {validationResult.errors.map((err, i) => (
                  <li key={i} style={{ marginBottom: 4 }}>
                    {err}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Dataset Case Preview */}
          {samplePreview.length > 0 && (
            <div>
              <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#fff', marginBottom: 10 }}>
                Sample Records Preview ({samplePreview.length} cases shown)
              </h3>
              <div className="table-container glass-panel">
                <table className="eval-table">
                  <thead>
                    <tr>
                      <th style={{ width: 140 }}>ID</th>
                      <th>Input Prompt</th>
                      <th>Reference (Expected)</th>
                      <th>Context / Grounding</th>
                    </tr>
                  </thead>
                  <tbody>
                    {samplePreview.map((c, i) => (
                      <tr key={c.id || i}>
                        <td className="mono" style={{ fontWeight: 600, color: '#22d3ee' }}>
                          {c.id || `row-${i + 1}`}
                        </td>
                        <td style={{ color: '#f8fafc' }}>{c.input}</td>
                        <td style={{ color: 'var(--text-muted)' }}>{c.reference || '—'}</td>
                        <td style={{ color: 'var(--text-muted)' }}>{c.context || '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
