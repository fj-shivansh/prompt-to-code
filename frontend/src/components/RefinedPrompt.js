import React from 'react';

const RefinedPrompt = ({ 
  showRefinedPrompt, 
  refinedPrompt, 
  setRefinedPrompt, 
  useRefinedPrompt, 
  dismissRefinedPrompt,
  refinementError,
  setRefinementError 
}) => {
  if (refinementError) {
    return (
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
    );
  }

  if (!showRefinedPrompt) return null;

  return (
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
  );
};

export default RefinedPrompt;