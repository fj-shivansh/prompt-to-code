import React, { useState, useEffect, useRef } from 'react';

const OptimizationPanel = ({
  onStartOptimization,
  isRunning,
  currentStatus,
  onStopOptimization
}) => {
  const [iterations, setIterations] = useState(2);
  const [error, setError] = useState(null);
  const iterationsInputRef = useRef(null);

  const handleStart = () => {
    // Validate iterations
    if (iterations < 1 || iterations > 50) {
      setError('Iterations must be between 1 and 50');
      return;
    }

    setError(null);
    onStartOptimization(iterations);
  };

  const handleStop = () => {
    if (window.confirm('Are you sure you want to stop the optimization?')) {
      onStopOptimization();
    }
  };

  // Calculate progress percentage
  const progressPct = currentStatus?.total_iterations && iterations
    ? Math.min((currentStatus.total_iterations / iterations) * 100, 100)
    : 0;

  return (
    <div className="optimization-panel">
      <div className="optimization-header">
        <h3>🤖 Automated Strategy Optimization</h3>
        <p className="optimization-description">
          Let AI generate, test, and improve trading strategies automatically.
          The system will create multiple strategies, evaluate their performance,
          and learn from results to generate better strategies.
        </p>
      </div>

      <div className="optimization-controls">
        <div className="control-group">
          <label htmlFor="iterations-input">
            Number of Iterations:
            <input
              id="iterations-input"
              ref={iterationsInputRef}
              type="number"
              min="1"
              max="50"
              value={iterations}
              onChange={(e) => setIterations(parseInt(e.target.value) || 1)}
              disabled={isRunning}
            />
          </label>
          <span className="control-hint">
            Each iteration generates a new strategy, tests it, and calculates NAV metrics
          </span>
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        <div className="action-buttons">
          {!isRunning ? (
            <button
              className="start-optimization-btn"
              onClick={handleStart}
              disabled={isRunning}
            >
              🚀 Start Optimization
            </button>
          ) : (
            <button
              className="stop-optimization-btn"
              onClick={handleStop}
            >
              ⏹ Stop Optimization
            </button>
          )}
        </div>
      </div>

      {isRunning && (
        <div className="optimization-progress">
          <div className="progress-header">
            <h4>🔄 Optimization in Progress...</h4>
            {currentStatus && (
              <span className="progress-text">
                Iteration {currentStatus.total_iterations || 0} of {iterations}
              </span>
            )}
          </div>

          <div className="progress-bar-container">
            <div
              className="progress-bar-fill"
              style={{ width: `${progressPct}%` }}
            >
              <span className="progress-percentage">{Math.round(progressPct)}%</span>
            </div>
          </div>

          {currentStatus && (
            <div className="live-stats">
              <div className="stat-card">
                <span className="stat-label">Current Iteration</span>
                <span className="stat-value">{currentStatus.total_iterations || 0}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Successful</span>
                <span className="stat-value success">{currentStatus.successful_iterations || 0}</span>
              </div>
              <div className="stat-card">
                <span className="stat-label">Failed</span>
                <span className="stat-value error">
                  {(currentStatus.total_iterations || 0) - (currentStatus.successful_iterations || 0)}
                </span>
              </div>
              <div className="stat-card highlight">
                <span className="stat-label">Best Ratio</span>
                <span className="stat-value">{currentStatus.best_ratio?.toFixed(2) || 'N/A'}</span>
              </div>
            </div>
          )}

          {!currentStatus && (
            <div className="live-status-message">
              <p>⏳ Starting optimization... Waiting for first update...</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default OptimizationPanel;
