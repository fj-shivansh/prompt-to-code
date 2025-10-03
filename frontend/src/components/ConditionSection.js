import React from 'react';

const ConditionSection = ({ 
  result, 
  showConditionForm, 
  setShowConditionForm,
  conditionPrompt, 
  setConditionPrompt, 
  handleConditionSubmit, 
  conditionLoading 
}) => {
  if (!result || !result.success) return null;

  if (!showConditionForm) {
    return (
      <div className="show-condition-btn-container">
        <button 
          className="show-condition-btn"
          onClick={() => setShowConditionForm(true)}
        >
          + Add Condition Processing
        </button>
      </div>
    );
  }

  return (
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

      {result.suggested_conditions && result.suggested_conditions.length > 0 && (
        <div className="suggested-conditions">
          <h3>💡 Suggested Profitable Conditions</h3>
          <p className="suggestions-hint">Click on any suggestion to use it:</p>
          <div className="suggestions-list">
            {result.suggested_conditions.map((condition, idx) => (
              <div
                key={idx}
                className="suggestion-item"
                onClick={() => setConditionPrompt(condition)}
              >
                <span className="suggestion-number">{idx + 1}</span>
                <span className="suggestion-text">{condition}</span>
              </div>
            ))}
          </div>
        </div>
      )}

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
  );
};

export default ConditionSection;