import React from 'react';

const StatusSection = ({ 
  isStreaming, 
  statusMessages, 
  showProcessingStatus, 
  setShowProcessingStatus 
}) => {
  if (!isStreaming && (!statusMessages.length || !showProcessingStatus)) return null;

  return (
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
  );
};

export default StatusSection;