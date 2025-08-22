import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = 'https://f92mgm70-5000.inc1.devtunnels.ms/api';

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

  useEffect(() => {
    fetchTickers();
    fetchDatabaseData(1);
  }, []);

  useEffect(() => {
    fetchDatabaseData(1);
  }, [sortConfig, tickerFilter]);

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

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/process_prompt`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
      });

      const data = await response.json();
      setResult(data);
      setShowFullOutput(false); // Reset to truncated view for new results
    } catch (error) {
      setResult({
        error: `Failed to process prompt: ${error.message}`
      });
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

  const handleSort = (key) => {
    let direction = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
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
              <button type="submit" disabled={loading || !prompt.trim()}>
                {loading ? 'Processing...' : 'Generate & Execute Code'}
              </button>
            </form>
          </div>

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
                  </div>

                  {result.result && (
                    <div className="section">
                      <div className="section-header">
                        <h3>Output</h3>
                        <div className="header-buttons">
                          {shouldShowToggle(result.result) && (
                            <button 
                              className="toggle-btn"
                              onClick={() => setShowFullOutput(!showFullOutput)}
                            >
                              {showFullOutput ? 'Show Less' : 'Show More'}
                            </button>
                          )}
                          <button 
                            className="download-btn"
                            onClick={() => downloadOutput(result.result, 'output.txt')}
                            title="Download output"
                          >
                            📥 Download
                          </button>
                        </div>
                      </div>
                      <pre className="code-block">
                        {showFullOutput ? result.result : truncateOutput(result.result)}
                      </pre>
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
