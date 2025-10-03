import React from 'react';

const FilterSection = ({ 
  tickerCount, 
  setTickerCount,
  startDate, 
  setStartDate, 
  endDate, 
  setEndDate,
  totalAvailableTickers = 500,
  loading = false
}) => {
  
  // Performance warnings based on selection
  const getPerformanceWarning = () => {
    const count = tickerCount === 'all' ? totalAvailableTickers : parseInt(tickerCount);
    if (count > 100) return { 
      message: "⚠️ Large dataset - may take 5+ minutes", 
      class: "warning-high" 
    };
    if (count > 50) return { 
      message: "⚠️ Medium dataset - may take 2-3 minutes", 
      class: "warning-medium" 
    };
    if (count > 20) return { 
      message: "ℹ️ Processing ~30-60 seconds", 
      class: "warning-low" 
    };
    return { 
      message: "✅ Fast processing (~10-30 seconds)", 
      class: "warning-good" 
    };
  };

  const performanceInfo = getPerformanceWarning();
  
  // Calculate date range duration
  const getDateRangeDuration = () => {
    if (!startDate || !endDate) return '';
    const start = new Date(startDate);
    const end = new Date(endDate);
    const diffTime = Math.abs(end - start);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    const diffYears = (diffDays / 365).toFixed(1);
    
    if (diffDays < 30) return `${diffDays} days`;
    if (diffDays < 365) return `${Math.round(diffDays / 30)} months`;
    return `${diffYears} years`;
  };

  return (
    <div className="filter-section">
      <h3>
        Data Filters 
        <span className="total-count">
          ({totalAvailableTickers} tickers available)
        </span>
      </h3>
      
      <div className="filter-container">
        <div className="filter-row">
          <div className="filter-item">
            <label htmlFor="ticker-count">Number of Random Tickers:</label>
            <select 
              id="ticker-count"
              value={tickerCount} 
              onChange={(e) => setTickerCount(e.target.value)}
              disabled={loading}
              className="filter-select"
            >
              <option value="5">5 Random Tickers (Recommended)</option>
              <option value="10">10 Random Tickers</option>
              <option value="20">20 Random Tickers</option>
              <option value="50">50 Random Tickers</option>
              <option value="100">100 Random Tickers</option>
              <option value="all">All {totalAvailableTickers} Tickers (Slow!)</option>
            </select>
            <small className="filter-info">
              Random tickers will be selected during code generation
            </small>
          </div>
          
          <div className="filter-item">
            <label htmlFor="start-date">Start Date:</label>
            <input 
              id="start-date"
              type={'date'}
              value={startDate || ''}
              placeholder={'YYYY-MM-DD (e.g., 2025-01-01)'}
              onChange={(e) => setStartDate(e.target.value)}
              disabled={loading}
              className="filter-input"
            />
          </div>
          
          <div className="filter-item">
            <label htmlFor="end-date">End Date:</label>
              <input 
                id="end-date"
                type="date"
                value={endDate || ''}
                onChange={(e) => setEndDate(e.target.value)}
                disabled={loading}
                className="filter-input"
              />
          </div>
        </div>
        
        <div className="filter-summary">
          <div className="summary-item">
            <strong>Selected Data:</strong> {' '}
            {tickerCount === 'all' ? 'All' : tickerCount} tickers
            {startDate && endDate && (
              <span> • {getDateRangeDuration()} of data</span>
            )}
          </div>
          
          <div className={`performance-indicator ${performanceInfo.class}`}>
            {performanceInfo.message}
          </div>
        </div>
      </div>
    </div>
  );
};

export default FilterSection;