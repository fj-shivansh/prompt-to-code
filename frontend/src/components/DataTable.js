import React from 'react';

const DataTable = ({ 
  data, 
  columns, 
  sortConfig, 
  onSort, 
  currentPage, 
  totalPages, 
  onPageChange,
  loading,
  title,
  downloadUrl,
  onRefetch 
}) => {
  if (loading) {
    return <div className="loading">Loading data...</div>;
  }

  if (!data || data.length === 0) {
    return (
      <div className="csv-error">
        <div className="error-message">
          <p>No data available.</p>
          {onRefetch && (
            <button 
              className="refetch-btn"
              onClick={() => onRefetch()}
            >
              🔄 Try Again
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="section">
      <div className="section-header">
        <h3>{title}</h3>
        <div className="header-buttons">
          {downloadUrl && (
            <button 
              className="download-btn"
              onClick={() => window.open(downloadUrl, '_blank')}
              title="Download CSV"
            >
              📥 Download CSV
            </button>
          )}
        </div>
      </div>
      
      <div className="csv-output">
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                {columns.map((column) => (
                  <th 
                    key={column}
                    onClick={() => onSort && onSort(column)}
                    className={onSort ? "sortable" : ""}
                    title={onSort ? "Click to sort" : ""}
                  >
                    {column} {sortConfig?.key === column && (sortConfig.direction === 'asc' ? '↑' : '↓')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, index) => (
                <tr key={index}>
                  {columns.map((column) => (
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
        
        {totalPages > 1 && (
          <div className="pagination">
            <button 
              onClick={() => onPageChange(currentPage - 1)}
              disabled={currentPage === 1}
            >
              Previous
            </button>
            
            <span className="page-info">
              Page {currentPage} of {totalPages}
            </span>
            
            <button 
              onClick={() => onPageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataTable;