import React, { useState, useEffect } from 'react';
import './App.css';
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
import { Line } from 'react-chartjs-2';

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
  const [abortController, setAbortController] = useState(null);
  const [canStop, setCanStop] = useState(false);

  // Condition processing states
  const [conditionPrompt, setConditionPrompt] = useState('');
  const [conditionLoading, setConditionLoading] = useState(false);
  const [conditionResult, setConditionResult] = useState(null);
  const [conditionCsvData, setConditionCsvData] = useState([]);
  const [conditionCsvColumns, setConditionCsvColumns] = useState([]);
  const [conditionCsvCurrentPage, setConditionCsvCurrentPage] = useState(1);
  const [conditionCsvTotalPages, setConditionCsvTotalPages] = useState(1);
  const [conditionCsvSortConfig, setConditionCsvSortConfig] = useState({ key: 'Date', direction: 'desc' });
  const [loadingConditionCsv, setLoadingConditionCsv] = useState(false);
  const [activeTab, setActiveTab] = useState('results'); // 'results', 'condition', or 'nav'
  const [showConditionForm, setShowConditionForm] = useState(true);
  
  // NAV-related state
  const [navData, setNavData] = useState([]);
  const [navGraph, setNavGraph] = useState('');
  const [navMetrics, setNavMetrics] = useState(null);
  const [navLoading, setNavLoading] = useState(false);
  const [navError, setNavError] = useState(null);
  const [navSettings, setNavSettings] = useState({
    initialAmount: 100000,
    amountToInvest: 1
  });

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

  useEffect(() => {
    if (conditionCsvData.length > 0) {
      fetchConditionCsvData(1);
    }
  }, [conditionCsvSortConfig]);

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

  const handleStop = async () => {
    if (abortController) {
      console.log('Stopping processing...');
      abortController.abort();
      
      // Send stop request to backend
      try {
        await fetch(`${API_BASE_URL}/stop_processing`, {
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
    setShowFullOutput(false);
    setShowPromptDetails(false);
    setShowRefinedPrompt(false);
    setShowProcessingStatus(true);
    setCanStop(true);

    try {
      // Create abort controller for this request
      const controller = new AbortController();
      setAbortController(controller);
      
      // Fallback to regular fetch for EventSource POST limitation
      const response = await fetch(`${API_BASE_URL}/process_prompt_stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt }),
        signal: controller.signal,
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
                setCanStop(false);
                // Hide processing status after successful completion
                setTimeout(() => {
                  setShowProcessingStatus(false);
                }, 2000);
              } else if (data.type === 'final_error') {
                setResult({ error: data.message });
                setIsStreaming(false);
                setCanStop(false);
                // Hide processing status after error as well
                setTimeout(() => {
                  setShowProcessingStatus(false);
                }, 3000);
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

  const downloadNavCsv = () => {
    if (navData.length > 0) {
      const headers = Object.keys(navData[0]).join(',');
      const csvContent = navData.map(row => 
        Object.values(row).map(val => 
          typeof val === 'string' && val.includes(',') ? `"${val}"` : val
        ).join(',')
      ).join('\n');
      const fullCsv = headers + '\n' + csvContent;
      
      const blob = new Blob([fullCsv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'nav_data.csv';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const fetchMainCsvDirectly = async () => {
    try {
      await fetchCsvData(1);
      // Create a minimal result object to show the results section
      if (!result) {
        setResult({
          success: true,
          direct_csv_load: true,
          message: 'CSV data loaded directly'
        });
      }
      // Switch to results tab to show the loaded data
      setActiveTab('results');
    } catch (error) {
      console.error('Error fetching main CSV:', error);
    }
  };

  const fetchConditionCsvDirectly = async () => {
    try {
      await fetchConditionCsvData(1);
      // Create a minimal condition result to show the condition section
      if (!conditionResult) {
        setConditionResult({
          success: true,
          direct_csv_load: true,
          message: 'Condition CSV data loaded directly'
        });
      }
      // Switch to condition tab to show the loaded data
      setActiveTab('condition');
    } catch (error) {
      console.error('Error fetching condition CSV:', error);
    }
  };

  const getChartData = () => {
    if (!navData || navData.length === 0) return null;

    // Show ALL data points - no sampling
    const dates = navData.map(item => {
      const date = new Date(item.Date);
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: '2-digit' });
    });
    const navValues = navData.map(item => item.NAV);

    return {
      labels: dates,
      datasets: [
        {
          label: 'Portfolio NAV',
          data: navValues,
          borderColor: '#ff9800',
          backgroundColor: 'rgba(255, 152, 0, 0.08)',
          borderWidth: 2,
          fill: true,
          pointBackgroundColor: '#ff9800',
          pointBorderColor: '#fff',
          pointBorderWidth: 1,
          pointRadius: 1, // Small points to show all data
          pointHoverRadius: 4,
          pointHoverBackgroundColor: '#f57c00',
          pointHoverBorderWidth: 2,
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: 'rgba(0, 0, 0, 0.85)',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: 'rgb(255, 152, 0)',
        borderWidth: 2,
        cornerRadius: 8,
        callbacks: {
          label: function(context) {
            return `NAV: $${context.parsed.y.toLocaleString()}`;
          },
          afterLabel: function(context) {
            const initialValue = navMetrics?.initial_amount || 100000;
            const currentValue = context.parsed.y;
            const returnPct = ((currentValue - initialValue) / initialValue * 100).toFixed(2);
            return `Return: ${returnPct}%`;
          }
        }
      }
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Date',
          color: '#666',
          font: {
            size: 12,
            weight: 500
          }
        },
        ticks: {
          color: '#666',
          maxTicksLimit: 8,
          font: {
            size: 11
          }
        },
        grid: {
          display: false
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Portfolio Value ($)',
          color: '#666',
          font: {
            size: 12,
            weight: 500
          }
        },
        ticks: {
          color: '#666',
          font: {
            size: 11
          },
          callback: function(value) {
            return '$' + (value / 1000).toFixed(0) + 'K';
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
          drawBorder: false
        }
      }
    },
    interaction: {
      intersect: false,
      mode: 'index'
    },
    animation: {
      duration: 800,
      easing: 'easeOutCubic'
    },
    elements: {
      point: {
        radius: 2,
        hoverRadius: 6
      },
      line: {
        tension: 0.2
      }
    }
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

  // Condition processing functions
  const handleConditionSubmit = async (e) => {
    e.preventDefault();
    if (!conditionPrompt.trim()) return;

    setConditionLoading(true);
    setConditionResult(null);
    setConditionCsvData([]);
    setConditionCsvColumns([]);

    try {
      const response = await fetch(`${API_BASE_URL}/process_condition`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ condition: conditionPrompt }),
      });

      const data = await response.json();
      
      if (response.ok) {
        setConditionResult(data);
        setShowConditionForm(false); // Hide form after successful processing
        setActiveTab('condition'); // Switch to condition tab
        // Fetch condition CSV data after processing
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

  // NAV calculation function
  const calculateNav = async () => {
    setNavLoading(true);
    setNavError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/calculate_nav`, {
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
        setNavGraph(data.graph_base64);
        setNavMetrics(data.metrics);
        setActiveTab('nav'); // Switch to NAV tab
      } else {
        setNavError(data.error);
      }
    } catch (error) {
      setNavError(`Failed to calculate NAV: ${error.message}`);
    } finally {
      setNavLoading(false);
    }
  };

  const fetchConditionCsvData = async (page, retryCount = 0) => {
    setLoadingConditionCsv(true);
    try {
      const sortOrder = conditionCsvSortConfig.direction === 'asc' ? 'ASC' : 'DESC';
      const params = new URLSearchParams({
        page: page.toString(),
        per_page: '50',
        sort_order: sortOrder,
      });
      
      if (conditionCsvSortConfig.key) {
        params.append('sort_by', conditionCsvSortConfig.key);
      }
      
      const response = await fetch(`${API_BASE_URL}/condition_csv_data?${params}`);
      const data = await response.json();
      
      if (data.error) {
        console.error('Condition CSV data error:', data.error);
        if (data.error.includes('No condition output file found') && retryCount < 5) {
          const delay = Math.min(1000 * Math.pow(2, retryCount), 5000);
          console.log(`Retrying condition CSV fetch in ${delay}ms... (attempt ${retryCount + 1}/5)`);
          setTimeout(() => {
            fetchConditionCsvData(page, retryCount + 1);
          }, delay);
          return;
        }
        setConditionCsvData([]);
        setConditionCsvColumns([]);
      } else {
        setConditionCsvData(data.data || []);
        setConditionCsvColumns(data.columns || []);
        setConditionCsvCurrentPage(data.page || 1);
        setConditionCsvTotalPages(data.total_pages || 1);
      }
    } catch (error) {
      console.error('Error fetching condition CSV data:', error);
      if (retryCount < 4) {
        const delay = Math.min(1000 * Math.pow(2, retryCount), 4000);
        setTimeout(() => {
          fetchConditionCsvData(page, retryCount + 1);
        }, delay);
        return;
      }
      setConditionCsvData([]);
      setConditionCsvColumns([]);
    }
    setLoadingConditionCsv(false);
  };

  const handleConditionCsvSort = (key) => {
    let direction = 'asc';
    if (conditionCsvSortConfig.key === key && conditionCsvSortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setConditionCsvSortConfig({ key, direction });
  };

  const handleConditionCsvPageChange = (newPage) => {
    if (newPage >= 1 && newPage <= conditionCsvTotalPages) {
      fetchConditionCsvData(newPage);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Prompt-to-Code Testing System</h1>
      </header>

      <div className="main-container">
        <div className="left-panel">
          {/* Always Available CSV Fetch Buttons */}
          <div className="persistent-fetch-section">
            <h2>Quick Data Access</h2>
            <p className="fetch-description">Load existing CSV data without running new prompts</p>
            <div className="persistent-fetch-buttons">
              <button 
                className="fetch-csv-btn main-fetch"
                onClick={fetchMainCsvDirectly}
                disabled={loadingCsv}
                title="Load the latest main results CSV file"
              >
                {loadingCsv ? '🔄 Loading Main CSV...' : '📊 Load Main Results CSV'}
              </button>
              <button 
                className="fetch-csv-btn condition-fetch"
                onClick={fetchConditionCsvDirectly}
                disabled={loadingConditionCsv}
                title="Load the latest condition results CSV file"
              >
                {loadingConditionCsv ? '🔄 Loading Condition CSV...' : '🔍 Load Condition CSV'}
              </button>
            </div>
          </div>

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
              <div className="button-group">
                <button 
                  type="button"
                  onClick={handleRefinePrompt}
                  disabled={isRefining || loading || !prompt.trim()}
                  className="refine-btn"
                >
                  {isRefining ? 'Refining...' : '✨ Refine Prompt'}
                </button>
                {canStop && (
                  <button 
                    type="button"
                    onClick={handleStop}
                    className="stop-btn"
                    title="Stop processing"
                  >
                    🛑 Stop
                  </button>
                )}
              </div>
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

          {/* Condition Prompt Section - shows only after successful data generation and when form is visible */}
          {result && result.success && showConditionForm && (
            <div className="condition-section">
              <div className="condition-header">
                <h2>Step 2: Condition Processing</h2>
                <button 
                  className="minimize-btn"
                  onClick={() => setShowConditionForm(false)}
                  title="Hide condition form"
                >
                  ✕
                </button>
              </div>
              <p className="condition-description">
                Now that you have generated data, create conditions in natural language to add binary (0/1) columns based on your calculated values.
              </p>
              
              <form onSubmit={handleConditionSubmit}>
                <div className="input-group">
                  <label htmlFor="condition-prompt">Enter your condition in plain English:</label>
                  <textarea
                    id="condition-prompt"
                    value={conditionPrompt}
                    onChange={(e) => setConditionPrompt(e.target.value)}
                    placeholder="e.g., when 10 day moving average is greater than 5 day moving average"
                    rows="3"
                    disabled={conditionLoading}
                  />
                  <small className="condition-hint">
                    Describe your condition in natural language. AI will automatically map it to your data columns.
                  </small>
                </div>
                <div className="button-group">
                  <button 
                    type="submit"
                    disabled={conditionLoading || !conditionPrompt.trim()}
                    className="condition-btn"
                  >
                    {conditionLoading ? 'Processing Condition...' : '🔍 Process Condition'}
                  </button>
                </div>
              </form>
            </div>
          )}
          
          {/* Show condition form toggle button when it's hidden */}
          {result && result.success && !showConditionForm && (
            <div className="show-condition-btn-container">
              <button 
                className="show-condition-btn"
                onClick={() => setShowConditionForm(true)}
              >
                + Add Condition Processing
              </button>
            </div>
          )}

          {/* Real-time Status Updates */}
          {(isStreaming || (statusMessages.length > 0 && showProcessingStatus)) && (
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
                <div className="logs-only">
                  <div className="logs-container">
                    {statusMessages.map((msg, index) => (
                      <div key={index} className={`log-message ${msg.type}`}>
                        <span className="timestamp">[{msg.timestamp || new Date().toLocaleTimeString()}]</span>
                        <span className="log-text">{msg.message}</span>
                        {msg.tokens && (
                          <span className="token-info"> | Tokens: {msg.tokens.input_tokens}↑ {msg.tokens.output_tokens}↓</span>
                        )}
                      </div>
                    ))}
                    {isStreaming && (
                      <div className="streaming-indicator">
                        <span className="pulse-dot"></span>
                        <span>Processing...</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {result && (
            <div className="results-section">
              <div className="results-tabs">
                <button 
                  className={`tab-btn ${activeTab === 'results' ? 'active' : ''}`}
                  onClick={() => setActiveTab('results')}
                >
                  📊 Main Results
                </button>
                {conditionResult && (
                  <button 
                    className={`tab-btn ${activeTab === 'condition' ? 'active' : ''}`}
                    onClick={() => setActiveTab('condition')}
                  >
                    🔍 Condition Results
                  </button>
                )}
                {conditionResult && (
                  <button 
                    className={`tab-btn ${activeTab === 'nav' ? 'active' : ''}`}
                    onClick={() => setActiveTab('nav')}
                  >
                    📈 NAV Analysis
                  </button>
                )}
              </div>
              
              {activeTab === 'results' && (
                <div className="tab-content">
                  {result.error ? (
                    <div className="error">
                      <h3>Error</h3>
                      <p>{result.error}</p>
                    </div>
                  ) : (
                    <div className="success">
                  
                  {/* Show direct CSV load message if applicable */}
                  {result.direct_csv_load && (
                    <div className="direct-load-notice">
                      <h3>📊 CSV Data Loaded Directly</h3>
                      <p>Displaying existing CSV data without running a new prompt.</p>
                    </div>
                  )}

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
                  
                  {!result.direct_csv_load && (
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
                      {result.had_complete_restarts && (
                        <div className="metric">
                          <span className="label">Complete Restarts:</span>
                          <span className="value">{result.complete_restart_attempts - 1}</span>
                        </div>
                      )}
                    </div>
                  )}

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

                  {!result.direct_csv_load && result.code && (
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
                  )}

                  {!result.direct_csv_load && result.explanation && (
                    <div className="section">
                      <h3>Explanation</h3>
                      <p>{result.explanation}</p>
                    </div>
                  )}

                  {!result.direct_csv_load && result.requirements && result.requirements.length > 0 && (
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
                        {result.analytics.generation_info.tokens && (
                          <>
                            <div className="analytics-item">
                              <span>Input Tokens:</span>
                              <span>{result.analytics.generation_info.tokens.input_tokens}</span>
                            </div>
                            <div className="analytics-item">
                              <span>Output Tokens:</span>
                              <span>{result.analytics.generation_info.tokens.output_tokens}</span>
                            </div>
                            <div className="analytics-item">
                              <span>Total Tokens:</span>
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
              
              {/* Condition Results Tab Content */}
              {activeTab === 'condition' && (
                <div className="tab-content condition-tab-content">
                  {conditionResult && conditionResult.error ? (
                    <div className="error">
                      <h3>Condition Error</h3>
                      <p>{conditionResult.error}</p>
                    </div>
                  ) : conditionResult ? (
                <div className="success">
                  
                  {/* Show direct CSV load message if applicable */}
                  {conditionResult.direct_csv_load && (
                    <div className="direct-load-notice">
                      <h3>🔍 Condition CSV Data Loaded Directly</h3>
                      <p>Displaying existing condition CSV data without running a new condition.</p>
                    </div>
                  )}

                  {!conditionResult.direct_csv_load && (
                    <div className="result-summary">
                      <div className="metric">
                        <span className="label">Condition:</span>
                        <span className="value">{conditionResult.condition_prompt}</span>
                      </div>
                      <div className="metric">
                        <span className="label">Execution Time:</span>
                        <span className="value">{conditionResult.execution_time?.toFixed(4)}s</span>
                      </div>
                    </div>
                  )}

                  {(conditionResult.result || conditionCsvData.length > 0) && (
                    <div className="section">
                      <div className="section-header">
                        <h3>Condition Output</h3>
                        <div className="header-buttons">
                          <button 
                            className="download-btn"
                            onClick={() => window.open(`${API_BASE_URL}/download_condition_csv`, '_blank')}
                            title="Download Condition CSV"
                          >
                            📥 Download Condition CSV
                          </button>
                        </div>
                      </div>
                      
                      {loadingConditionCsv ? (
                        <div className="loading">Loading condition data...</div>
                      ) : conditionCsvData.length > 0 ? (
                        <div className="csv-output">
                          <div className="table-container">
                            <table className="data-table">
                              <thead>
                                <tr>
                                  {conditionCsvColumns.map((column) => (
                                    <th 
                                      key={column}
                                      onClick={() => handleConditionCsvSort(column)}
                                      className="sortable"
                                      title="Click to sort"
                                    >
                                      {column} {conditionCsvSortConfig.key === column && (conditionCsvSortConfig.direction === 'asc' ? '↑' : '↓')}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {conditionCsvData.map((row, index) => (
                                  <tr key={index}>
                                    {conditionCsvColumns.map((column) => (
                                      <td key={column}>
                                        {column === 'Signal' ? (
                                          <span className={`condition-value ${row[column] === 1 ? 'true' : 'false'}`}>
                                            {row[column]}
                                          </span>
                                        ) : (
                                          typeof row[column] === 'number' && !Number.isInteger(row[column]) 
                                            ? row[column].toFixed(4) 
                                            : row[column]
                                        )}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          
                          <div className="pagination">
                            <button 
                              onClick={() => handleConditionCsvPageChange(conditionCsvCurrentPage - 1)}
                              disabled={conditionCsvCurrentPage === 1}
                            >
                              Previous
                            </button>
                            
                            <span className="page-info">
                              Page {conditionCsvCurrentPage} of {conditionCsvTotalPages}
                            </span>
                            
                            <button 
                              onClick={() => handleConditionCsvPageChange(conditionCsvCurrentPage + 1)}
                              disabled={conditionCsvCurrentPage === conditionCsvTotalPages}
                            >
                              Next
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div className="csv-error">
                          <div className="error-message">
                            <p>Condition data not available.</p>
                            <button 
                              className="refetch-btn"
                              onClick={() => fetchConditionCsvData(1, 0)}
                              disabled={loadingConditionCsv}
                            >
                              {loadingConditionCsv ? '🔄 Retrying...' : '🔄 Try Again'}
                            </button>
                          </div>
                          {conditionResult.result && (
                            <div className="text-output">
                              <pre className="code-block">
                                {conditionResult.result}
                              </pre>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="section">
                    <div className="section-header">
                      <h3>Generated Condition Code</h3>
                      <button 
                        className="download-btn"
                        onClick={() => downloadOutput(conditionResult.code, 'condition_generated_code.py')}
                        title="Download condition code"
                      >
                        📥 Download
                      </button>
                    </div>
                    <pre className="code-block">{conditionResult.code}</pre>
                  </div>

                  <div className="section">
                    <h3>Condition Explanation</h3>
                    <p>{conditionResult.explanation}</p>
                  </div>

                  {conditionResult.requirements && conditionResult.requirements.length > 0 && (
                    <div className="section">
                      <h3>Condition Requirements</h3>
                      <ul>
                        {conditionResult.requirements.map((req, index) => (
                          <li key={index}>{req}</li>
                        ))}
                      </ul>
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
              
              {/* NAV Analysis Tab Content */}
              {activeTab === 'nav' && (
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
                                initialAmount: parseFloat(e.target.value) || 100000
                              })}
                              min="1000"
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
                                amountToInvest: parseFloat(e.target.value) || 1
                              })}
                              min="0.1"
                              max="2"
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
                    
                    {navGraph && navMetrics && (
                      <div className="nav-results">
                        <div className="nav-metrics">
                          <div className="metric-card">
                            <span className="metric-label">Initial Amount</span>
                            <span className="metric-value">${(navMetrics.initial_amount / 1000).toFixed(0)}K</span>
                          </div>
                          <div className="metric-card">
                            <span className="metric-label">Final NAV</span>
                            <span className="metric-value">${(navMetrics.final_nav / 1000000).toFixed(2)}M</span>
                          </div>
                          <div className="metric-card">
                            <span className="metric-label">Total Return</span>
                            <span className={`metric-value ${navMetrics.total_return_pct >= 0 ? 'positive' : 'negative'}`}>
                              {navMetrics.total_return_pct.toFixed(2)}%
                            </span>
                          </div>
                          <div className="metric-card">
                            <span className="metric-label">Total Signals</span>
                            <span className="metric-value">{navMetrics.total_signals}</span>
                          </div>
                        </div>
                        
                        <div className="nav-graph">
                          <div className="nav-chart-header">
                            <h4>Portfolio Performance Over Time</h4>
                            <button
                              className="download-btn nav-download-btn"
                              onClick={downloadNavCsv}
                              title="Download NAV Data CSV"
                              disabled={!navData || navData.length === 0}
                            >
                              📥 Download CSV
                            </button>
                          </div>
                          {getChartData() ? (
                            <div className="nav-chart-container">
                              <Line data={getChartData()} options={chartOptions} />
                            </div>
                          ) : (
                            <div className="nav-chart-placeholder">
                              <p>No data available for charting</p>
                            </div>
                          )}
                        </div>
                        
                        {navData.length > 0 && (
                          <div className="nav-data-table">
                            <h4>NAV Data Points ({navData.length} entries)</h4>
                            <div className="table-container">
                              <table className="data-table">
                                <thead>
                                  <tr>
                                    <th>Date</th>
                                    <th>NAV ($)</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {navData.slice(0, 10).map((row, index) => (
                                    <tr key={index}>
                                      <td>{row.Date}</td>
                                      <td>${row.NAV.toLocaleString()}</td>
                                    </tr>
                                  ))}
                                  {navData.length > 10 && (
                                    <tr>
                                      <td colSpan="2" className="table-more">
                                        ... and {navData.length - 10} more entries
                                      </td>
                                    </tr>
                                  )}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                    
                    {!navGraph && !navLoading && (
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
    </div>
  );
}

export default App;