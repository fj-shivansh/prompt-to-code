import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:5000/api';
// const API_BASE_URL = 'https://f92mgm70-5000.inc1.devtunnels.ms/api';

function App() {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [databaseData, setDatabaseData] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loadingDatabase, setLoadingDatabase] = useState(false);
  const [showFullOutput, setShowFullOutput] = useState(false);
  const [sortConfig, setSortConfig] = useState({ key: 'Date', direction: 'desc' });
  const [tickerFilter, setTickerFilter] = useState('');
  const [availableTickers, setAvailableTickers] = useState([]);
  
  // CSV output data states
  const [csvData, setCsvData] = useState([]);
  const [csvColumns, setCsvColumns] = useState([]);
  const [csvCurrentPage, setCsvCurrentPage] = useState(1);
  const [csvTotalPages, setCsvTotalPages] = useState(1);
  const [csvSortConfig, setCsvSortConfig] = useState({ key: '', direction: 'desc' });
  const [loadingCsv, setLoadingCsv] = useState(false);
  const [showPromptDetails, setShowPromptDetails] = useState(false);
  const [statusMessages, setStatusMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [refinedPrompt, setRefinedPrompt] = useState('');
  const [isRefining, setIsRefining] = useState(false);
  const [showRefinedPrompt, setShowRefinedPrompt] = useState(false);
  const [refinementError, setRefinementError] = useState('');
  const [showProcessingStatus, setShowProcessingStatus] = useState(true);

  useEffect(() => {
    fetchTickers();
    fetchDatabaseData(1);
  }, []);

  useEffect(() => {
    fetchDatabaseData(1);
  }, [sortConfig, tickerFilter]);

  useEffect(() => {
    if (result && result.success) {
      // Add longer delay to ensure CSV file is written to disk and processed
      setTimeout(() => {
        fetchCsvData(1);
      }, 1500); // Increased from 500ms to 1500ms
    }
  }, [result]);

  useEffect(() => {
    if (csvData.length > 0) {
      fetchCsvData(1);
    }
  }, [csvSortConfig]);

  const fetchTickers = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/tickers`);
      const data = await response.json();
      setAvailableTickers(data.tickers || []);
    } catch (error) {
      console.error('Error fetching tickers:', error);
    }
  };

  const fetchDatabaseData = async (page) => {
    setLoadingDatabase(true);
    try {
      const sortOrder = sortConfig.direction === 'asc' ? 'ASC' : 'DESC';
      const params = new URLSearchParams({
        page: page.toString(),
        per_page: '50',
        sort_by: sortConfig.key,
        sort_order: sortOrder,
      });
      
      if (tickerFilter) {
        params.append('ticker', tickerFilter);
      }
      
      const response = await fetch(`${API_BASE_URL}/database_data?${params}`);
      const data = await response.json();
      setDatabaseData(data.data);
      setCurrentPage(data.page);
      setTotalPages(data.total_pages);
    } catch (error) {
      console.error('Error fetching database data:', error);
    }
    setLoadingDatabase(false);
  };

  const handleRefinePrompt = async () => {
    if (!prompt.trim()) return;

    setIsRefining(true);
    setRefinementError('');
    try {
      const response = await fetch(`${API_BASE_URL}/refine_prompt`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      const data = await response.json();
      
      if (data.success) {
        setRefinedPrompt(data.refined_prompt);
        setShowRefinedPrompt(true);
        setRefinementError('');
      } else {
        setRefinementError(data.error);
      }
    } catch (error) {
      setRefinementError(`Failed to refine prompt: ${error.message}`);
    }
    setIsRefining(false);
  };

  const useRefinedPrompt = () => {
    const finalPrompt = refinedPrompt;
    setPrompt(finalPrompt);
    setShowRefinedPrompt(false);
    setRefinedPrompt('');
    
    // Trigger form submission programmatically
    setTimeout(() => {
      document.querySelector('form').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    }, 100);
  };

  const dismissRefinedPrompt = () => {
    setShowRefinedPrompt(false);
    setRefinedPrompt('');
    setRefinementError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setResult(null);
    setStatusMessages([]);
    setIsStreaming(true);
    setShowFullOutput(false);
    setShowPromptDetails(false);
    setShowRefinedPrompt(false);

    try {
      // Fallback to regular fetch for EventSource POST limitation
      const response = await fetch(`${API_BASE_URL}/process_prompt_stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              setStatusMessages(prev => [...prev, data]);
              
              if (data.type === 'final_result') {
                setResult(data.data);
                setIsStreaming(false);
              } else if (data.type === 'final_error') {
                setResult({ error: data.message });
                setIsStreaming(false);
              }
            } catch (e) {
              console.warn('Failed to parse SSE data:', line);
            }
          }
        }
      }
    } catch (error) {
      setResult({
        error: `Failed to process prompt: ${error.message}`
      });
      setIsStreaming(false);
    }

    setLoading(false);
  };

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      fetchDatabaseData(newPage);
    }
  };

  const truncateOutput = (text, maxLines = 10) => {
    if (!text || typeof text !== 'string') return text;
    const lines = text.split('\n');
    if (lines.length <= maxLines) return text;
    return lines.slice(0, maxLines).join('\n') + '\n\n... (output truncated - click "Show More" to see full output)';
  };

  const shouldShowToggle = (text) => {
    if (!text || typeof text !== 'string') return false;
    return text.split('\n').length > 10;
  };

  const downloadOutput = (content, filename) => {
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const fetchCsvData = async (page, retryCount = 0) => {
    setLoadingCsv(true);
    try {
      const sortOrder = csvSortConfig.direction === 'asc' ? 'ASC' : 'DESC';
      const params = new URLSearchParams({
        page: page.toString(),
        per_page: '50',
        sort_order: sortOrder,
      });
      
      if (csvSortConfig.key) {
        params.append('sort_by', csvSortConfig.key);
      }
      
      const response = await fetch(`${API_BASE_URL}/csv_data?${params}`);
      const data = await response.json();
      
      if (data.error) {
        console.error('CSV data error:', data.error);
        // If CSV file not found, retry with exponential backoff
        if (data.error.includes('No CSV output file found') && retryCount < 5) {
          const delay = Math.min(1000 * Math.pow(2, retryCount), 5000); // Exponential backoff, max 5s
          console.log(`Retrying CSV fetch in ${delay}ms... (attempt ${retryCount + 1}/5)`);
          setTimeout(() => {
            fetchCsvData(page, retryCount + 1);
          }, delay);
          return;
        }
        setCsvData([]);
        setCsvColumns([]);
      } else {
        // Successful fetch - always set the data even if empty
        console.log('CSV fetch successful:', { 
          dataLength: data.data ? data.data.length : 'no data array',
          columns: data.columns ? data.columns.length : 'no columns',
          totalPages: data.total_pages 
        });
        
        setCsvData(data.data || []);
        setCsvColumns(data.columns || []);
        setCsvCurrentPage(data.page || 1);
        setCsvTotalPages(data.total_pages || 1);
      }
    } catch (error) {
      console.error('Error fetching CSV data:', error);
      // Retry on network errors with exponential backoff
      if (retryCount < 4) {
        const delay = Math.min(1000 * Math.pow(2, retryCount), 4000);
        console.log(`Retrying CSV fetch due to network error in ${delay}ms... (attempt ${retryCount + 1}/4)`);
        setTimeout(() => {
          fetchCsvData(page, retryCount + 1);
        }, delay);
        return;
      }
      setCsvData([]);
      setCsvColumns([]);
    }
    setLoadingCsv(false);
  };

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  const handleCsvSort = (key) => {
    let direction = 'asc';
    if (csvSortConfig.key === key && csvSortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setCsvSortConfig({ key, direction });
  };

  const handleCsvPageChange = (newPage) => {
    if (newPage >= 1 && newPage <= csvTotalPages) {
      fetchCsvData(newPage);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Prompt-to-Code Testing System</h1>
      </header>

      <div className="main-container">
        <div className="left-panel">
          <div className="prompt-section">
            <form onSubmit={handleSubmit}>
              <div className="input-group">
                <label htmlFor="prompt">Enter your prompt:</label>
                <textarea
                  id="prompt"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="e.g., Calculate the average adjusted close price for AAPL"
                  rows="4"
                  disabled={loading}
                />
              </div>
              <button 
                type="button"
                onClick={handleRefinePrompt}
                disabled={isRefining || loading || !prompt.trim()}
                className="refine-btn"
              >
                {isRefining ? 'Refining...' : '✨ Refine Prompt'}
              </button>
            </form>
          </div>

          {/* Refinement Error */}
          {refinementError && (
            <div className="refinement-error">
              <h3>Refinement Error</h3>
              <p>{refinementError}</p>
              <button 
                onClick={() => setRefinementError('')}
                className="dismiss-error-btn"
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Refined Prompt Preview */}
          {showRefinedPrompt && (
            <div className="refined-prompt-section">
              <h3>Refined Prompt</h3>
              <p className="refinement-description">
                Your prompt has been enhanced. You can edit it further or use it as-is:
              </p>
              <div className="refined-prompt-editor">
                <textarea
                  value={refinedPrompt}
                  onChange={(e) => setRefinedPrompt(e.target.value)}
                  rows="6"
                  className="refined-textarea"
                  placeholder="Edit the refined prompt as needed..."
                />
              </div>
              <div className="refined-prompt-actions">
                <button 
                  onClick={useRefinedPrompt}
                  className="use-refined-btn"
                  disabled={!refinedPrompt.trim()}
                >
                  Generate & Execute Code
                </button>
                <button 
                  onClick={() => {
                    dismissRefinedPrompt();
                    document.querySelector('form').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
                  }}
                  className="use-original-btn"
                >
                  Use Original Instead
                </button>
              </div>
            </div>
          )}

          {/* Real-time Status Updates */}
          {(isStreaming || statusMessages.length > 0) && (
            <div className="status-section">
              <div 
                className="status-header"
                onClick={() => setShowProcessingStatus(!showProcessingStatus)}
              >
                <h3>Processing Status</h3>
                <button className="collapse-btn">
                  {showProcessingStatus ? '▲' : '▼'}
                </button>
              </div>
              
              {showProcessingStatus && (
                <div className="status-content">
              
              {/* Progress Bar */}
              {statusMessages.some(msg => msg.progress) && (
                <div className="progress-container">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{
                        width: `${(statusMessages.filter(msg => msg.progress).slice(-1)[0]?.progress?.step || 0) / 4 * 100}%`
                      }}
                    ></div>
                  </div>
                  <span className="progress-text">
                    Step {statusMessages.filter(msg => msg.progress).slice(-1)[0]?.progress?.step || 0} of 4
                  </span>
                </div>
              )}
              
              {/* Current Status */}
              <div className="current-status">
                {statusMessages.slice(-1).map((msg, index) => (
                  <div key={`current-${index}`} className={`current-message ${msg.type}`}>
                    {msg.type === 'prompt_refined' ? (
                      <div>
                        <strong>✨ Prompt Enhanced</strong>
                        <div className="refined-prompt-preview">{msg.refined?.substring(0, 150)}...</div>
                      </div>
                    ) : (
                      <div className="status-content">
                        <span className="status-icon">
                          {msg.type === 'init' && '🚀'}
                          {msg.type === 'attempt_start' && '🔄'}
                          {msg.type === 'refining' && '✨'}
                          {msg.type === 'generating' && '⚡'}
                          {msg.type === 'execution_complete' && '📋'}
                          {msg.type === 'retry' && '🔁'}
                          {msg.type === 'retry_success' && '✅'}
                          {msg.type === 'retry_failed' && '❌'}
                          {msg.type === 'attempt_success' && '🎉'}
                          {msg.type === 'attempt_failed' && '⚠️'}
                          {msg.type === 'restarting' && '🔄'}
                          {msg.type === 'final_error' && '💥'}
                        </span>
                        <span>{msg.message}</span>
                      </div>
                    )}
                  </div>
                ))}
                {isStreaming && (
                  <div className="streaming-indicator">
                    <span className="pulse-dot"></span>
                    <span>Streaming...</span>
                  </div>
                )}
              </div>
              
              {/* Attempt Tracking */}
              {statusMessages.some(msg => msg.attempt) && (
                <div className="attempt-tracking">
                  <div className="attempt-header">
                    <span>Complete Attempts Progress:</span>
                  </div>
                  <div className="attempt-grid">
                    {[1, 2, 3].map(attemptNum => {
                      const attemptMsgs = statusMessages.filter(msg => msg.attempt === attemptNum);
                      const isActive = attemptMsgs.length > 0;
                      const isSuccess = attemptMsgs.some(msg => msg.type === 'attempt_success');
                      const isFailed = attemptMsgs.some(msg => msg.type === 'attempt_failed');
                      const retryCount = attemptMsgs.filter(msg => msg.type === 'retry').length;
                      
                      return (
                        <div key={attemptNum} className={`attempt-card ${isActive ? 'active' : ''} ${isSuccess ? 'success' : ''} ${isFailed ? 'failed' : ''}`}>
                          <div className="attempt-number">Attempt {attemptNum}</div>
                          <div className="retry-count">
                            {isActive && `${retryCount}/5 retries`}
                            {isSuccess && '✅ Success'}
                            {isFailed && '❌ Failed'}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
              
              {/* Detailed Log (Collapsible) */}
              <details className="detailed-log">
                <summary>View Detailed Log ({statusMessages.length} events)</summary>
                <div className="status-messages">
                  {statusMessages.map((msg, index) => (
                    <div key={index} className={`status-message ${msg.type}`}>
                      <span className="timestamp">[{new Date().toLocaleTimeString()}]</span>
                      {msg.type === 'prompt_refined' ? (
                        <div>
                          <strong>Prompt Refined:</strong>
                          <div className="refined-prompt">{msg.refined}</div>
                        </div>
                      ) : (
                        <span>{msg.message}</span>
                      )}
                    </div>
                  ))}
                </div>
              </details>
                </div>
              )}
            </div>
          )}

          {result && (
            <div className="results-section">
              <h2>Results</h2>
              
              {result.error ? (
                <div className="error">
                  <h3>Error</h3>
                  <p>{result.error}</p>
                </div>
              ) : (
                <div className="success">
                  {/* Show prompt refinement details if available */}
                  {result.prompt_was_refined && (
                    <div className="section">
                      <div 
                        className="section-header-expandable"
                        onClick={() => setShowPromptDetails(!showPromptDetails)}
                      >
                        <h3>Prompt Refinement</h3>
                        <button className="expand-btn">
                          {showPromptDetails ? '▲' : '▼'}
                        </button>
                      </div>
                      
                      {showPromptDetails && (
                        <div className="prompt-details">
                          <div className="prompt-item">
                            <h4>Original Prompt:</h4>
                            <div className="prompt-content original">
                              {result.original_prompt}
                            </div>
                          </div>
                          <div className="prompt-item">
                            <h4>Refined Prompt:</h4>
                            <div className="prompt-content refined">
                              {result.refined_prompt}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                  
                  <div className="result-summary">
                    <div className="metric">
                      <span className="label">Success:</span>
                      <span className="value">{result.success ? 'Yes' : 'No'}</span>
                    </div>
                    <div className="metric">
                      <span className="label">Execution Time:</span>
                      <span className="value">{result.execution_time?.toFixed(4)}s</span>
                    </div>
                    <div className="metric">
                      <span className="label">Success Rate:</span>
                      <span className="value">{(result.success_rate * 100).toFixed(1)}%</span>
                    </div>
                    {result.prompt_was_refined && (
                      <div className="metric">
                        <span className="label">Prompt Refined:</span>
                        <span className="value">Yes</span>
                      </div>
                    )}
                    <div className="metric">
                      <span className="label">Execution Retries:</span>
                      <span className="value">{result.execution_retry_attempts - 1}/{result.max_execution_retries - 1}</span>
                    </div>
                    {result.had_complete_restarts && (
                      <div className="metric">
                        <span className="label">Complete Restarts:</span>
                        <span className="value">{result.complete_restart_attempts - 1}</span>
                      </div>
                    )}
                  </div>

                  {(result.result || csvData.length > 0) && (
                    <div className="section">
                      <div className="section-header">
                        <h3>Output</h3>
                        <div className="header-buttons">
                          <button 
                            className="download-btn"
                            onClick={() => window.open(`${API_BASE_URL}/download_csv`, '_blank')}
                            title="Download CSV"
                          >
                            📥 Download CSV
                          </button>
                        </div>
                      </div>
                      
                      {loadingCsv ? (
                        <div className="loading">Loading CSV data... (may take a few seconds for file processing)</div>
                      ) : csvData.length > 0 ? (
                        <div className="csv-output">
                          <div className="table-container">
                            <table className="data-table">
                              <thead>
                                <tr>
                                  {csvColumns.map((column) => (
                                    <th 
                                      key={column}
                                      onClick={() => handleCsvSort(column)}
                                      className="sortable"
                                      title="Click to sort"
                                    >
                                      {column} {csvSortConfig.key === column && (csvSortConfig.direction === 'asc' ? '↑' : '↓')}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {csvData.map((row, index) => (
                                  <tr key={index}>
                                    {csvColumns.map((column) => (
                                      <td key={column}>
                                        {typeof row[column] === 'number' && !Number.isInteger(row[column]) 
                                          ? row[column].toFixed(4) 
                                          : row[column]}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          
                          <div className="pagination">
                            <button 
                              onClick={() => handleCsvPageChange(csvCurrentPage - 1)}
                              disabled={csvCurrentPage === 1}
                            >
                              Previous
                            </button>
                            
                            <span className="page-info">
                              Page {csvCurrentPage} of {csvTotalPages}
                            </span>
                            
                            <button 
                              onClick={() => handleCsvPageChange(csvCurrentPage + 1)}
                              disabled={csvCurrentPage === csvTotalPages}
                            >
                              Next
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div className="csv-error">
                          <div className="error-message">
                            <p>CSV data not available or failed to load after multiple attempts.</p>
                            <p>This can happen due to timing issues during code execution.</p>
                            <button 
                              className="refetch-btn"
                              onClick={() => fetchCsvData(1, 0)}
                              disabled={loadingCsv}
                            >
                              {loadingCsv ? '🔄 Retrying...' : '🔄 Try Again'}
                            </button>
                          </div>
                          {result.result && (
                            <div className="text-output">
                              <pre className="code-block">
                                {result.result}
                              </pre>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="section">
                    <div className="section-header">
                      <h3>Generated Code</h3>
                      <button 
                        className="download-btn"
                        onClick={() => downloadOutput(result.code, 'generated_code.py')}
                        title="Download code"
                      >
                        📥 Download
                      </button>
                    </div>
                    <pre className="code-block">{result.code}</pre>
                  </div>

                  <div className="section">
                    <h3>Explanation</h3>
                    <p>{result.explanation}</p>
                  </div>

                  {result.requirements && result.requirements.length > 0 && (
                    <div className="section">
                      <h3>Requirements</h3>
                      <ul>
                        {result.requirements.map((req, index) => (
                          <li key={index}>{req}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {result.analytics && (
                    <div className="section">
                      <h3>Analytics</h3>
                      <div className="analytics-grid">
                        <div className="analytics-item">
                          <span>Total Tests:</span>
                          <span>{result.analytics.summary.total_tests}</span>
                        </div>
                        <div className="analytics-item">
                          <span>Code Length:</span>
                          <span>{result.analytics.generation_info.code_length} chars</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="right-panel">
          <div className="database-section">
            <h2>Database Data</h2>
            
            <div className="filter-section">
              <select
                value={tickerFilter}
                onChange={(e) => setTickerFilter(e.target.value)}
                className="filter-input"
              >
                <option value="">All Tickers</option>
                {availableTickers.map((ticker) => (
                  <option key={ticker} value={ticker}>
                    {ticker}
                  </option>
                ))}
              </select>
            </div>
            
            {loadingDatabase ? (
              <div className="loading">Loading database data...</div>
            ) : (
              <>
                <div className="table-container">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th 
                          onClick={() => handleSort('Date')}
                          className="sortable"
                          title="Click to sort"
                        >
                          Date {sortConfig.key === 'Date' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                        </th>
                        <th 
                          onClick={() => handleSort('Ticker')}
                          className="sortable"
                          title="Click to sort"
                        >
                          Ticker {sortConfig.key === 'Ticker' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                        </th>
                        <th 
                          onClick={() => handleSort('Adj_Close')}
                          className="sortable"
                          title="Click to sort"
                        >
                          Adj Close {sortConfig.key === 'Adj_Close' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {databaseData.map((row, index) => (
                        <tr key={index}>
                          <td>{row.Date}</td>
                          <td>{row.Ticker}</td>
                          <td>${row.Adj_Close?.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="pagination">
                  <button 
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={currentPage === 1}
                  >
                    Previous
                  </button>
                  
                  <span className="page-info">
                    Page {currentPage} of {totalPages}
                  </span>
                  
                  <button 
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={currentPage === totalPages}
                  >
                    Next
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;