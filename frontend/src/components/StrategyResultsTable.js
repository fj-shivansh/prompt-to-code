import React, { useState } from 'react';

const StrategyResultsTable = ({ results, onUseStrategy }) => {
  const [expandedRow, setExpandedRow] = useState(null);

  if (!results || results.length === 0) {
    return (
      <div className="no-results">
        <p>No optimization results yet. Start an optimization to see results here.</p>
      </div>
    );
  }

  // Find best strategy
  const successfulResults = results.filter(r => r.status === 'success' && r.nav_metrics);
  const bestRatio = successfulResults.length > 0
    ? Math.max(...successfulResults.map(r => r.nav_metrics?.ratio || 0))
    : 0;

  const toggleRow = (iteration) => {
    setExpandedRow(expandedRow === iteration ? null : iteration);
  };

  const truncateText = (text, maxLength = 100) => {
    if (!text) return 'N/A';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  };

  const getStatusBadge = (status) => {
    const badges = {
      success: <span className="status-badge success">✓ Success</span>,
      failed: <span className="status-badge failed">✗ Failed</span>,
      error: <span className="status-badge error">⚠ Error</span>
    };
    return badges[status] || <span className="status-badge unknown">?</span>;
  };

  return (
    <div className="strategy-results-section">
      <div className="results-header">
        <h3>📊 Optimization Results</h3>
        <p className="results-summary">
          Total: {results.length} |
          Successful: <span className="success-count">{successfulResults.length}</span> |
          Failed: <span className="failed-count">{results.length - successfulResults.length}</span>
          {bestRatio > 0 && <> | Best Ratio: <span className="best-ratio">{bestRatio.toFixed(2)}</span></>}
        </p>
      </div>

      <div className="results-table-container">
        <table className="strategy-results-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Main Prompt</th>
              <th>Condition</th>
              <th>Ratio</th>
              <th>Annual Return</th>
              <th>Max Drawdown</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {results.map((result) => {
              const isBest = result.status === 'success' &&
                             result.nav_metrics?.ratio === bestRatio;
              const isExpanded = expandedRow === result.iteration;

              return (
                <React.Fragment key={result.iteration}>
                  <tr
                    className={`result-row ${isBest ? 'best-strategy' : ''} ${isExpanded ? 'expanded' : ''}`}
                    onClick={() => toggleRow(result.iteration)}
                  >
                    <td className="iteration-cell">
                      {result.iteration}
                      {isBest && <span className="best-badge">🏆</span>}
                    </td>
                    <td className="prompt-cell">
                      {truncateText(result.main_prompt, 80)}
                    </td>
                    <td className="condition-cell">
                      {truncateText(result.condition_prompt, 60)}
                    </td>
                    <td className="metric-cell">
                      {result.nav_metrics?.ratio?.toFixed(2) || 'N/A'}
                    </td>
                    <td className="metric-cell">
                      {result.nav_metrics?.annual_return?.toFixed(2) || 'N/A'}%
                    </td>
                    <td className="metric-cell">
                      {result.nav_metrics?.max_drawdown?.toFixed(2) || 'N/A'}%
                    </td>
                    <td className="status-cell">
                      {getStatusBadge(result.status)}
                    </td>
                    <td className="actions-cell">
                      {result.status === 'success' && (
                        <button
                          className="use-strategy-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            onUseStrategy(result);
                          }}
                          title="Use this strategy"
                        >
                          📋 Use
                        </button>
                      )}
                    </td>
                  </tr>

                  {isExpanded && (
                    <tr className="expanded-row">
                      <td colSpan="8">
                        <div className="expanded-content">
                          <div className="expanded-section">
                            <h4>📝 Main Prompt:</h4>
                            <pre className="prompt-text">{result.main_prompt || 'N/A'}</pre>
                          </div>

                          <div className="expanded-section">
                            <h4>🎯 Condition Prompt:</h4>
                            <pre className="prompt-text">{result.condition_prompt || 'N/A'}</pre>
                          </div>

                          {result.status === 'success' && result.nav_metrics && (
                            <div className="expanded-section">
                              <h4>📈 Detailed Metrics:</h4>
                              <div className="metrics-grid">
                                <div className="metric-item">
                                  <span className="metric-label">Initial Amount:</span>
                                  <span className="metric-value">
                                    ${result.nav_metrics.initial_amount?.toLocaleString()}
                                  </span>
                                </div>
                                <div className="metric-item">
                                  <span className="metric-label">Final NAV:</span>
                                  <span className="metric-value">
                                    ${result.nav_metrics.final_nav?.toLocaleString()}
                                  </span>
                                </div>
                                <div className="metric-item">
                                  <span className="metric-label">Total Return:</span>
                                  <span className="metric-value">
                                    {result.nav_metrics.total_return_pct?.toFixed(2)}%
                                  </span>
                                </div>
                                <div className="metric-item">
                                  <span className="metric-label">Total Signals:</span>
                                  <span className="metric-value">
                                    {result.nav_metrics.total_signals}
                                  </span>
                                </div>
                              </div>
                            </div>
                          )}

                          {result.status === 'failed' && result.error_message && (
                            <div className="expanded-section error-section">
                              <h4>❌ Error Message:</h4>
                              <pre className="error-text">{result.error_message}</pre>
                            </div>
                          )}
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default StrategyResultsTable;
