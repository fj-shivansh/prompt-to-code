// Configuration object for the entire application
const config = {
  // API Configuration
  api: {
    baseUrl: process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api',
    timeout: 30000, // 30 seconds
  },
  
  // Pagination settings
  pagination: {
    perPage: parseInt(process.env.REACT_APP_PAGINATION_PER_PAGE) || 50,
    defaultPage: 1,
  },
  
  // CSV processing settings
  csv: {
    retryMax: parseInt(process.env.REACT_APP_CSV_RETRY_MAX) || 5,
    retryDelay: parseInt(process.env.REACT_APP_CSV_RETRY_DELAY) || 1000,
    maxRetryDelay: 5000,
  },
  
  // Sort configuration
  sort: {
    directions: {
      ASC: 'asc',
      DESC: 'desc'
    },
    default: {
      key: 'Date',
      direction: 'desc'
    }
  },
  
  // Tab names
  tabs: {
    RESULTS: 'results',
    CONDITION: 'condition',
    NAV: 'nav'
  },
  
  // NAV settings
  nav: {
    defaultSettings: {
      initialAmount: 100000,
      amountToInvest: 0.7,
      maxPositionEachTicker: 0.2,
      traderCost: 0
    },
    limits: {
      minAmount: 1000,
      maxAmount: 10000000,
      minInvestment: 0,
      maxInvestment: 1.0,
      minPosition: 0,
      maxPosition: 1.0,
      minTraderCost: 0,
      maxTraderCost: 100
    }
  },
  
  // UI settings
  ui: {
    statusUpdateDelay: 2000,
    errorDisplayDelay: 3000,
    chartAnimationDuration: 800,
  },
  
  // Development settings
  dev: {
    enableDebugLogs: process.env.NODE_ENV === 'development',
    showDetailedErrors: process.env.NODE_ENV === 'development',
  }
};

export default config;