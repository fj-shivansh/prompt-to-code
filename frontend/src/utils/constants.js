import config from '../config/config';

// Re-export config values for backward compatibility and easier access
export const API_BASE_URL = config.api.baseUrl;
export const PAGINATION = config.pagination;
export const SORT_DIRECTIONS = config.sort.directions;
export const TAB_NAMES = config.tabs;

// Export the full config for advanced usage
export { config };