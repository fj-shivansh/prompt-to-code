import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';

// Components
import PersistentCsvFetch from './components/PersistentCsvFetch';
import PromptSection from './components/PromptSection';
import RefinedPrompt from './components/RefinedPrompt';
import ConditionSection from './components/ConditionSection';
import StatusSection from './components/StatusSection';
import DataTable from './components/DataTable';
import NavChart from './components/NavChart';

// Hooks
import useCsvData from './hooks/useCsvData';
import useConditionCsvData from './hooks/useConditionCsvData';

// Utils
import { downloadOutput, downloadNavCsv } from './utils/downloadUtils';
import { TAB_NAMES, config } from './utils/constants';

// Styles
import './App.css';
import './styles/components.css';
import './styles/layout.css';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

function App() {
  // Basic state
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState(TAB_NAMES.RESULTS);

  // Database data state
  const [databaseData, setDatabaseData] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loadingDatabase, setLoadingDatabase] = useState(false);
  const [sortConfig, setSortConfig] = useState({ key: 'Date', direction: 'desc' });
  const [tickerFilter, setTickerFilter] = useState('');
  const [availableTickers, setAvailableTickers] = useState([]);
  
  // Refinement state
  const [showRefinedPrompt, setShowRefinedPrompt] = useState(false);
  const [refinedPrompt, setRefinedPrompt] = useState('');
  const [isRefining, setIsRefining] = useState(false);
  const [refinementError, setRefinementError] = useState('');

  // Processing state
  const [statusMessages, setStatusMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [showProcessingStatus, setShowProcessingStatus] = useState(true);
  const [abortController, setAbortController] = useState(null);
  const [canStop, setCanStop] = useState(false);

  // Condition processing state
  const [conditionPrompt, setConditionPrompt] = useState('');
  const [conditionLoading, setConditionLoading] = useState(false);
  const [conditionResult, setConditionResult] = useState(null);
  const [showConditionForm, setShowConditionForm] = useState(true);
  
  // NAV state
  const [navData, setNavData] = useState([]);
  const [navMetrics, setNavMetrics] = useState(null);
  const [navLoading, setNavLoading] = useState(false);
  const [navError, setNavError] = useState(null);
  const [navSettings, setNavSettings] = useState(config.nav.defaultSettings);

  // Toast state
  const [toastMessage, setToastMessage] = useState('');
  const [showToast, setShowToast] = useState(false);

  // Custom hooks
  const {
    csvData,
    csvColumns,
    csvCurrentPage,
    csvTotalPages,
    csvSortConfig,
    loadingCsv,
    fetchCsvData,
    handleCsvSort,
    handleCsvPageChange,
    setCsvSortConfig
  } = useCsvData();

  const {
    conditionCsvData,
    conditionCsvColumns,
    conditionCsvCurrentPage,
    conditionCsvTotalPages,
    conditionCsvSortConfig,
    loadingConditionCsv,
    fetchConditionCsvData,
    handleConditionCsvSort,
    handleConditionCsvPageChange,
    setConditionCsvSortConfig
  } = useConditionCsvData();

  // Effects
  useEffect(() => {
    fetchTickers();
    fetchDatabaseData(1);
  }, []);

  useEffect(() => {
    fetchDatabaseData(1);
  }, [sortConfig, tickerFilter]);

  useEffect(() => {
    if (result && result.success) {
      setTimeout(() => {
        fetchCsvData(1);
      }, 1500);
    }
  }, [result]);

  useEffect(() => {
    if (csvData.length > 0) {
      fetchCsvData(1);
    }
  }, [csvSortConfig]);

  useEffect(() => {
    if (conditionCsvData.length > 0) {
      fetchConditionCsvData(1);
    }
  }, [conditionCsvSortConfig]);

  // Utility Functions
  const showToastNotification = (message) => {
    setToastMessage(message);
    setShowToast(true);
    setTimeout(() => {
      setShowToast(false);
    }, 3000);
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      showToastNotification('Code copied to clipboard!');
    } catch (err) {
      // Fallback for older browsers
      const textArea = document.createElement('textarea');
      textArea.value = text;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      showToastNotification('Code copied to clipboard!');
    }
  };

  // API Functions
  const fetchTickers = async () => {
    try {
      const response = await fetch(`${config.api.baseUrl}/tickers`);
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
        per_page: config.pagination.perPage.toString(),
        sort_by: sortConfig.key,
        sort_order: sortOrder,
      });
      
      if (tickerFilter) {
        params.append('ticker', tickerFilter);
      }
      
      const response = await fetch(`${config.api.baseUrl}/database_data?${params}`);
      const data = await response.json();
      setDatabaseData(data.data);
      setCurrentPage(data.page);
      setTotalPages(data.total_pages);
    } catch (error) {
      console.error('Error fetching database data:', error);
    }
    setLoadingDatabase(false);
  };

  const fetchMainCsvDirectly = async () => {
    try {
      await fetchCsvData(1);
      if (!result) {
        setResult({
          success: true,
          direct_csv_load: true,
          message: 'CSV data loaded directly'
        });
      }
      setActiveTab(TAB_NAMES.RESULTS);
    } catch (error) {
      console.error('Error fetching main CSV:', error);
    }
  };

  const fetchConditionCsvDirectly = async () => {
    try {
      await fetchConditionCsvData(1);
      if (!conditionResult) {
        setConditionResult({
          success: true,
          direct_csv_load: true,
          message: 'Condition CSV data loaded directly'
        });
      }
      setActiveTab(TAB_NAMES.CONDITION);
    } catch (error) {
      console.error('Error fetching condition CSV:', error);
    }
  };

  const handleRefinePrompt = async () => {
    if (!prompt.trim()) return;

    setIsRefining(true);
    setRefinementError('');
    try {
      const response = await fetch(`${config.api.baseUrl}/refine_prompt`, {
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
    
    setTimeout(() => {
      document.querySelector('form').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
    }, 100);
  };

  const dismissRefinedPrompt = () => {
    setShowRefinedPrompt(false);
    setRefinedPrompt('');
    setRefinementError('');
  };

  const handleStop = async () => {
    if (abortController) {
      console.log('Stopping processing...');
      abortController.abort();
      
      try {
        await fetch(`${config.api.baseUrl}/stop_processing`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
        });
      } catch (error) {
        console.warn('Failed to send stop request to backend:', error);
      }
      
      setCanStop(false);
      setIsStreaming(false);
      setLoading(false);
      setStatusMessages(prev => [...prev, {
        type: 'user_stopped',
        message: 'Processing stopped by user',
        timestamp: new Date().toISOString()
      }]);
      setResult({ error: 'Processing was stopped by user' });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setResult(null);
    setStatusMessages([]);
    setIsStreaming(true);
    setShowRefinedPrompt(false);
    setShowProcessingStatus(true);
    setCanStop(true);

    try {
      const controller = new AbortController();
      setAbortController(controller);
      
      const response = await fetch(`${config.api.baseUrl}/process_prompt_stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
        signal: controller.signal,
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let shouldBreak = false;
      while (true && !shouldBreak) {
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
                setCanStop(false);
                setTimeout(() => {
                  setShowProcessingStatus(false);
                }, 2000);
              } else if (data.type === 'final_error') {
                setResult({ error: data.message });
                setIsStreaming(false);
                setCanStop(false);
                setTimeout(() => {
                  setShowProcessingStatus(false);
                }, 3000);
              } else if (data.type === 'connection_close') {
                // Force close SSE connection
                console.log('SSE connection close signal received');
                shouldBreak = true;
                setTimeout(() => {
                  if (controller && !controller.signal.aborted) {
                    controller.abort();
                  }
                }, 100);
              }
            } catch (e) {
              console.warn('Failed to parse SSE data:', line);
            }
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request was aborted');
      } else {
        setResult({
          error: `Failed to process prompt: ${error.message}`
        });
        setIsStreaming(false);
        setCanStop(false);
      }
    } finally {
      setAbortController(null);
    }

    setLoading(false);
  };

  const handleConditionSubmit = async (e) => {
    e.preventDefault();
    if (!conditionPrompt.trim()) return;

    setConditionLoading(true);
    setConditionResult(null);

    try {
      const response = await fetch(`${config.api.baseUrl}/process_condition`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ condition: conditionPrompt }),
      });

      const data = await response.json();
      
      if (response.ok) {
        setConditionResult(data);
        setShowConditionForm(false);
        setActiveTab(TAB_NAMES.CONDITION);
        setTimeout(() => {
          fetchConditionCsvData(1);
        }, 1000);
      } else {
        setConditionResult({ error: data.error });
      }
    } catch (error) {
      setConditionResult({
        error: `Failed to process condition: ${error.message}`
      });
    }

    setConditionLoading(false);
  };

  // Helper function to format large numbers
  const formatNumber = (num, isPercentage = false, isCurrency = false) => {
    if (num === undefined || num === null) return '0';
    
    const absNum = Math.abs(num);
    const sign = num < 0 ? '-' : '';
    
    let formattedValue = '';
    
    if (absNum >= 1000000000000) { // Trillions
      formattedValue = `${(absNum / 1000000000000).toFixed(1)}T`;
    } else if (absNum >= 1000000000) { // Billions
      formattedValue = `${(absNum / 1000000000).toFixed(1)}B`;
    } else if (absNum >= 1000000) { // Millions
      formattedValue = `${(absNum / 1000000).toFixed(1)}M`;
    } else if (absNum >= 1000) { // Thousands
      formattedValue = `${(absNum / 1000).toFixed(1)}K`;
    } else {
      formattedValue = absNum.toFixed(1);
    }
    
    return `${sign}${isCurrency ? '$' : ''}${formattedValue}${isPercentage ? '%' : ''}`;
  };

  const calculateNav = async () => {
    setNavLoading(true);
    setNavError(null);

    try {
      const response = await fetch(`${config.api.baseUrl}/calculate_nav`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          initial_amount: navSettings.initialAmount,
          amount_to_invest: navSettings.amountToInvest
        })
      });

      const data = await response.json();
      
      if (response.ok) {
        setNavData(data.nav_data);
        setNavMetrics(data.metrics);
        setActiveTab(TAB_NAMES.NAV);
      } else {
        setNavError(data.error);
      }
    } catch (error) {
      setNavError(`Failed to calculate NAV: ${error.message}`);
    } finally {
      setNavLoading(false);
    }
  };

  const handlePageChange = (newPage) => {
    if (newPage >= 1 && newPage <= totalPages) {
      fetchDatabaseData(newPage);
    }
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
          <PersistentCsvFetch
            fetchMainCsvDirectly={fetchMainCsvDirectly}
            fetchConditionCsvDirectly={fetchConditionCsvDirectly}
            loadingCsv={loadingCsv}
            loadingConditionCsv={loadingConditionCsv}
          />

          <PromptSection
            prompt={prompt}
            setPrompt={setPrompt}
            handleSubmit={handleSubmit}
            handleRefinePrompt={handleRefinePrompt}
            handleStop={handleStop}
            loading={loading}
            isRefining={isRefining}
            canStop={canStop}
          />

          <RefinedPrompt
            showRefinedPrompt={showRefinedPrompt}
            refinedPrompt={refinedPrompt}
            setRefinedPrompt={setRefinedPrompt}
            useRefinedPrompt={useRefinedPrompt}
            dismissRefinedPrompt={dismissRefinedPrompt}
            refinementError={refinementError}
            setRefinementError={setRefinementError}
          />

          <ConditionSection
            result={result}
            showConditionForm={showConditionForm}
            setShowConditionForm={setShowConditionForm}
            conditionPrompt={conditionPrompt}
            setConditionPrompt={setConditionPrompt}
            handleConditionSubmit={handleConditionSubmit}
            conditionLoading={conditionLoading}
          />

          <StatusSection
            isStreaming={isStreaming}
            statusMessages={statusMessages}
            showProcessingStatus={showProcessingStatus}
            setShowProcessingStatus={setShowProcessingStatus}
          />

          {result && (
            <div className="results-section">
              <div className="results-tabs">
                <button 
                  className={`tab-btn ${activeTab === TAB_NAMES.RESULTS ? 'active' : ''}`}
                  onClick={() => setActiveTab(TAB_NAMES.RESULTS)}
                >
                  📊 Main Results
                </button>
                {conditionResult && (
                  <button 
                    className={`tab-btn ${activeTab === TAB_NAMES.CONDITION ? 'active' : ''}`}
                    onClick={() => setActiveTab(TAB_NAMES.CONDITION)}
                  >
                    🔍 Condition Results
                  </button>
                )}
                {conditionResult && (
                  <button 
                    className={`tab-btn ${activeTab === TAB_NAMES.NAV ? 'active' : ''}`}
                    onClick={() => setActiveTab(TAB_NAMES.NAV)}
                  >
                    📈 NAV Analysis
                  </button>
                )}
              </div>
              
              {activeTab === TAB_NAMES.RESULTS && (
                <div className="tab-content">
                  {result.error ? (
                    <div className="error">
                      <h3>Error</h3>
                      <p>{result.error}</p>
                    </div>
                  ) : (
                    <div className="success">
                      {result.direct_csv_load && (
                        <div className="direct-load-notice">
                          <h3>📊 CSV Data Loaded Directly</h3>
                          <p>Displaying existing CSV data without running a new prompt.</p>
                        </div>
                      )}

                      <DataTable
                        data={csvData}
                        columns={csvColumns}
                        sortConfig={csvSortConfig}
                        onSort={handleCsvSort}
                        currentPage={csvCurrentPage}
                        totalPages={csvTotalPages}
                        onPageChange={handleCsvPageChange}
                        loading={loadingCsv}
                        title="Output"
                        downloadUrl={`${config.api.baseUrl}/download_csv`}
                        onRefetch={() => fetchCsvData(1, 0)}
                      />

                      {!result.direct_csv_load && result.code && (
                        <div className="code-section">
                          <div className="code-header">
                            <h3>Generated Code</h3>
                            <div className="code-actions">
                              <button
                                className="copy-btn"
                                onClick={() => copyToClipboard(result.code)}
                                title="Copy code to clipboard"
                              >
                                📋 Copy
                              </button>
                              <button
                                className="download-btn"
                                onClick={() => downloadOutput(result.code, 'generated_code.py')}
                                title="Download generated code"
                              >
                                📥 Download Code
                              </button>
                            </div>
                          </div>
                          <pre className="code-block">{result.code}</pre>
                        </div>
                      )}

                      {!result.direct_csv_load && result.explanation && (
                        <div className="explanation-section">
                          <h3>Code Explanation</h3>
                          <p>{result.explanation}</p>
                        </div>
                      )}

                      {result.analytics && (
                        <div className="analytics-section">
                          <h3>Process Analytics</h3>
                          <div className="analytics-grid">
                            <div className="metric-item">
                              <span className="metric-label">Total Tests:</span>
                              <span>{result.analytics.summary.total_tests}</span>
                            </div>
                            <div className="metric-item">
                              <span className="metric-label">Code Length:</span>
                              <span>{result.analytics.generation_info.code_length} chars</span>
                            </div>
                            {result.analytics.generation_info.tokens && (
                              <>
                                <div className="metric-item">
                                  <span className="metric-label">Input Tokens:</span>
                                  <span>{result.analytics.generation_info.tokens.input_tokens}</span>
                                </div>
                                <div className="metric-item">
                                  <span className="metric-label">Output Tokens:</span>
                                  <span>{result.analytics.generation_info.tokens.output_tokens}</span>
                                </div>
                                <div className="metric-item">
                                  <span className="metric-label">Total Tokens:</span>
                                  <span>{result.analytics.generation_info.tokens.total_tokens}</span>
                                </div>
                              </>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
              
              {activeTab === TAB_NAMES.CONDITION && (
                <div className="tab-content condition-tab-content">
                  {conditionResult && conditionResult.error ? (
                    <div className="error">
                      <h3>Condition Error</h3>
                      <p>{conditionResult.error}</p>
                    </div>
                  ) : conditionResult ? (
                    <div className="success">
                      {conditionResult.direct_csv_load && (
                        <div className="direct-load-notice">
                          <h3>🔍 Condition CSV Data Loaded Directly</h3>
                          <p>Displaying existing condition CSV data without running a new condition.</p>
                        </div>
                      )}

                      <DataTable
                        data={conditionCsvData}
                        columns={conditionCsvColumns}
                        sortConfig={conditionCsvSortConfig}
                        onSort={handleConditionCsvSort}
                        currentPage={conditionCsvCurrentPage}
                        totalPages={conditionCsvTotalPages}
                        onPageChange={handleConditionCsvPageChange}
                        loading={loadingConditionCsv}
                        title="Condition Output"
                        downloadUrl={`${config.api.baseUrl}/download_condition_csv`}
                        onRefetch={() => fetchConditionCsvData(1, 0)}
                      />

                      {!conditionResult.direct_csv_load && conditionResult.code && (
                        <div className="code-section">
                          <div className="code-header">
                            <h3>Generated Condition Code</h3>
                            <div className="code-actions">
                              <button
                                className="copy-btn"
                                onClick={() => copyToClipboard(conditionResult.code)}
                                title="Copy code to clipboard"
                              >
                                📋 Copy
                              </button>
                              <button
                                className="download-btn"
                                onClick={() => downloadOutput(conditionResult.code, 'condition_generated_code.py')}
                                title="Download condition code"
                              >
                                📥 Download Code
                              </button>
                            </div>
                          </div>
                          <pre className="code-block">{conditionResult.code}</pre>
                        </div>
                      )}

                      {!conditionResult.direct_csv_load && conditionResult.explanation && (
                        <div className="explanation-section">
                          <h3>Condition Code Explanation</h3>
                          <p>{conditionResult.explanation}</p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="no-condition-data">
                      <p>No condition results yet. Use the "Load Latest Condition CSV" button above or process a condition first.</p>
                    </div>
                  )}
                </div>
              )}
              
              {activeTab === TAB_NAMES.NAV && (
                <div className="tab-content nav-tab-content">
                  <div className="nav-section">
                    <div className="nav-header">
                      <h3>📈 Portfolio NAV Analysis</h3>
                      <div className="nav-controls">
                        <div className="nav-settings">
                          <label>
                            Initial Amount ($):
                            <input
                              type="number"
                              value={navSettings.initialAmount}
                              onChange={(e) => setNavSettings({
                                ...navSettings,
                                initialAmount: parseFloat(e.target.value) || config.nav.defaultSettings.initialAmount
                              })}
                              min={config.nav.limits.minAmount}
                              max={config.nav.limits.maxAmount}
                              step="1000"
                            />
                          </label>
                          <label>
                            Investment Multiplier:
                            <input
                              type="number"
                              value={navSettings.amountToInvest}
                              onChange={(e) => setNavSettings({
                                ...navSettings,
                                amountToInvest: parseFloat(e.target.value) || config.nav.defaultSettings.amountToInvest
                              })}
                              min={config.nav.limits.minInvestment}
                              max={config.nav.limits.maxInvestment}
                              step="0.1"
                            />
                          </label>
                          <button
                            className="nav-calculate-btn"
                            onClick={calculateNav}
                            disabled={navLoading}
                          >
                            {navLoading ? 'Calculating...' : '🔄 Calculate NAV'}
                          </button>
                        </div>
                      </div>
                    </div>
                    
                    {navError && (
                      <div className="error">
                        <h4>NAV Calculation Error</h4>
                        <p>{navError}</p>
                      </div>
                    )}
                    
                    {navMetrics && (
                      <div className="nav-results">
                        <div className="nav-metrics-table">
                          <div className="metrics-summary">
                            <h4>📊 Performance Metrics</h4>
                            <div className="metrics-grid">
                              <div className="metric-row">
                                <div className="metric-section">
                                  <h5>Investment Summary</h5>
                                  <table className="metrics-table">
                                    <tbody>
                                      <tr>
                                        <td className="metric-label">Initial Amount</td>
                                        <td className="metric-value">{formatNumber(navMetrics.initial_amount, false, true)}</td>
                                      </tr>
                                      <tr>
                                        <td className="metric-label">Final NAV</td>
                                        <td className="metric-value">{formatNumber(navMetrics.final_nav, false, true)}</td>
                                      </tr>
                                    </tbody>
                                  </table>
                                </div>
                                <div className="metric-section">
                                  <h5>Returns Analysis</h5>
                                  <table className="metrics-table">
                                    <tbody>
                                      <tr>
                                        <td className="metric-label">Annual Return</td>
                                        <td className={`metric-value ${navMetrics.annual_return >= 0 ? 'positive' : 'negative'}`}>
                                          {formatNumber(navMetrics.annual_return, true)}
                                        </td>
                                      </tr>
                                      <tr>
                                        <td className="metric-label">Max Drawdown</td>
                                        <td className="metric-value negative">
                                          {formatNumber(-Math.abs(navMetrics.max_drawdown), true)}
                                        </td>
                                      </tr>
                                       <tr>
                                        <td className="metric-label">Ratio</td>
                                        <td className="metric-value">
                                          {formatNumber(navMetrics.ratio)}
                                        </td>
                                      </tr>
                                    </tbody>
                                  </table>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                        
                        <NavChart 
                          navData={navData}
                          navMetrics={navMetrics}
                          downloadNavCsv={() => downloadNavCsv(navData)}
                        />
                      </div>
                    )}
                    
                    {!navMetrics && !navLoading && (
                      <div className="nav-placeholder">
                        <p>📊 Calculate NAV to see portfolio performance analysis</p>
                        <p>This will use your condition results with Signal=1 to simulate trading performance.</p>
                      </div>
                    )}
                  </div>
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
                        <th 
                          onClick={() => handleSort('Daily_Gain_Pct')}
                          className="sortable"
                          title="Click to sort"
                        >
                          Daily Gain % {sortConfig.key === 'Daily_Gain_Pct' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                        </th>
                        <th 
                          onClick={() => handleSort('Forward_Gain_Pct')}
                          className="sortable"
                          title="Click to sort"
                        >
                          Forward Gain % {sortConfig.key === 'Forward_Gain_Pct' && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {databaseData.map((row, index) => (
                        <tr key={index}>
                          <td>{row.Date}</td>
                          <td>{row.Ticker}</td>
                          <td>${row.Adj_Close?.toFixed(2)}</td>
                          <td>
                            <span className={`gain-value ${row.Daily_Gain_Pct >= 0 ? 'positive' : 'negative'}`}>
                              {row.Daily_Gain_Pct ? (row.Daily_Gain_Pct * 100).toFixed(2) + '%' : 'N/A'}
                            </span>
                          </td>
                          <td>
                            <span className={`gain-value ${row.Forward_Gain_Pct >= 0 ? 'positive' : 'negative'}`}>
                              {row.Forward_Gain_Pct ? (row.Forward_Gain_Pct * 100).toFixed(2) + '%' : 'N/A'}
                            </span>
                          </td>
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

      {/* Toast Notification */}
      {showToast && (
        <div className="toast-notification">
          {toastMessage}
        </div>
      )}
    </div>
  );
}

export default App;