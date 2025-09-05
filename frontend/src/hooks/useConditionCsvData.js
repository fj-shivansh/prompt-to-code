import { useState } from 'react';
import { config } from '../utils/constants';

export const useConditionCsvData = () => {
  const [conditionCsvData, setConditionCsvData] = useState([]);
  const [conditionCsvColumns, setConditionCsvColumns] = useState([]);
  const [conditionCsvCurrentPage, setConditionCsvCurrentPage] = useState(1);
  const [conditionCsvTotalPages, setConditionCsvTotalPages] = useState(1);
  const [conditionCsvSortConfig, setConditionCsvSortConfig] = useState({ key: 'Date', direction: 'desc' });
  const [loadingConditionCsv, setLoadingConditionCsv] = useState(false);

  const fetchConditionCsvData = async (page, retryCount = 0) => {
    setLoadingConditionCsv(true);
    try {
      const sortOrder = conditionCsvSortConfig.direction === 'asc' ? 'ASC' : 'DESC';
      const params = new URLSearchParams({
        page: page.toString(),
        per_page: config.pagination.perPage.toString(),
        sort_order: sortOrder,
      });
      
      if (conditionCsvSortConfig.key) {
        params.append('sort_by', conditionCsvSortConfig.key);
      }
      
      const response = await fetch(`${config.api.baseUrl}/condition_csv_data?${params}`);
      const data = await response.json();
      
      if (data.error) {
        console.error('Condition CSV data error:', data.error);
        if (data.error.includes('No condition output file found') && retryCount < config.csv.retryMax) {
          const delay = Math.min(config.csv.retryDelay * Math.pow(2, retryCount), config.csv.maxRetryDelay);
          console.log(`Retrying condition CSV fetch in ${delay}ms... (attempt ${retryCount + 1}/${config.csv.retryMax})`);
          setTimeout(() => {
            fetchConditionCsvData(page, retryCount + 1);
          }, delay);
          return;
        }
        setConditionCsvData([]);
        setConditionCsvColumns([]);
      } else {
        setConditionCsvData(data.data || []);
        setConditionCsvColumns(data.columns || []);
        setConditionCsvCurrentPage(data.page || 1);
        setConditionCsvTotalPages(data.total_pages || 1);
      }
    } catch (error) {
      console.error('Error fetching condition CSV data:', error);
      if (retryCount < 4) {
        const delay = Math.min(1000 * Math.pow(2, retryCount), 4000);
        setTimeout(() => {
          fetchConditionCsvData(page, retryCount + 1);
        }, delay);
        return;
      }
      setConditionCsvData([]);
      setConditionCsvColumns([]);
    }
    setLoadingConditionCsv(false);
  };

  const handleConditionCsvSort = (key) => {
    let direction = 'asc';
    if (conditionCsvSortConfig.key === key && conditionCsvSortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setConditionCsvSortConfig({ key, direction });
  };

  const handleConditionCsvPageChange = (newPage) => {
    if (newPage >= 1 && newPage <= conditionCsvTotalPages) {
      fetchConditionCsvData(newPage);
    }
  };

  return {
    conditionCsvData,
    conditionCsvColumns,
    conditionCsvCurrentPage,
    conditionCsvTotalPages,
    conditionCsvSortConfig,
    loadingConditionCsv,
    fetchConditionCsvData,
    handleConditionCsvSort,
    handleConditionCsvPageChange,
    setConditionCsvSortConfig
  };
};

export default useConditionCsvData;