import React from 'react';

const PersistentCsvFetch = ({ 
  fetchMainCsvDirectly, 
  fetchConditionCsvDirectly, 
  loadingCsv, 
  loadingConditionCsv 
}) => {
  return (
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
  );
};

export default PersistentCsvFetch;