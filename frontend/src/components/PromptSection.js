import React from 'react';

const PromptSection = ({ 
  prompt, 
  setPrompt, 
  handleSubmit, 
  handleRefinePrompt, 
  handleStop,
  loading, 
  isRefining, 
  canStop 
}) => {
  return (
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
          {false && canStop && (
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
  );
};

export default PromptSection;