import React from 'react';
import { Line } from 'react-chartjs-2';

const NavChart = ({ navData, navMetrics, downloadNavCsv }) => {
  const getChartData = () => {
    if (!navData || navData.length === 0) return null;

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

  return (
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
  );
};

export default NavChart;