import React, { useRef, useEffect, useMemo, useCallback } from 'react';
import { Line } from 'react-chartjs-2';

const NavChart = React.memo(({ navData, navMetrics, downloadNavCsv }) => {
  const chartRef = useRef(null);
  const isFirstRender = useRef(true);

  const getChartData = useCallback(() => {
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
          pointBorderWidth: 0,
          pointRadius: 0, // Remove points for better performance
          pointHoverRadius: 4,
          pointHoverBackgroundColor: '#f57c00',
          pointHoverBorderWidth: 2,
        }
      ]
    };
  }, [navData]);

  // Optimize chart options for performance
  const chartOptions = useMemo(() => ({
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
      mode: 'nearest'
    },
    // Disable animations after first render for better performance
    animation: isFirstRender.current ? {
      duration: 400,
      easing: 'easeOutCubic'
    } : false,
    elements: {
      point: {
        radius: 0, // Remove points for better performance
        hoverRadius: 4
      },
      line: {
        tension: 0.1
      }
    }
  }), [isFirstRender.current]);

  // Cleanup effect to destroy chart on unmount and track first render
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
    }
    
    return () => {
      if (chartRef.current && chartRef.current.chartInstance) {
        chartRef.current.chartInstance.destroy();
      }
    };
  }, []);

  // Cleanup chart instance on data changes
  useEffect(() => {
    return () => {
      if (chartRef.current && chartRef.current.chartInstance) {
        chartRef.current.chartInstance.destroy();
      }
    };
  }, [navData]);

  const chartData = getChartData();

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
      {chartData ? (
        <div className="nav-chart-container">
          <Line 
            ref={chartRef}
            data={chartData} 
            options={chartOptions} 
            key={`chart-${navData?.length}`} // Force re-render on data change
          />
        </div>
      ) : (
        <div className="nav-chart-placeholder">
          <p>No data available for charting</p>
        </div>
      )}
    </div>
  );
});

export default NavChart;