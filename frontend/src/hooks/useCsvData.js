import { useState } from 'react';
import { config } from '../utils/constants';

export const useCsvData = () => {
  const [csvData, setCsvData] = useState([]);
  const [csvColumns, setCsvColumns] = useState([]);
  const [csvCurrentPage, setCsvCurrentPage] = useState(1);
  const [csvTotalPages, setCsvTotalPages] = useState(1);
  const [csvSortConfig, setCsvSortConfig] = useState({ key: '', direction: 'desc' });
  const [loadingCsv, setLoadingCsv] = useState(false);

  const fetchCsvData = async (page, retryCount = 0) => {
    setLoadingCsv(true);
    try {
      const sortOrder = csvSortConfig.direction === 'asc' ? 'ASC' : 'DESC';
      const params = new URLSearchParams({
        page: page.toString(),
        per_page: config.pagination.perPage.toString(),
        sort_order: sortOrder,
      });
      
      if (csvSortConfig.key) {
        params.append('sort_by', csvSortConfig.key);
      }
      
      const response = await fetch(`${config.api.baseUrl}/csv_data?${params}`);
      const data = await response.json();
      
      if (data.error) {
        console.error('CSV data error:', data.error);
        if (data.error.includes('No CSV output file found') && retryCount < config.csv.retryMax) {
          const delay = Math.min(config.csv.retryDelay * Math.pow(2, retryCount), config.csv.maxRetryDelay);
          console.log(`Retrying CSV fetch in ${delay}ms... (attempt ${retryCount + 1}/${config.csv.retryMax})`);
          setTimeout(() => {
            fetchCsvData(page, retryCount + 1);
          }, delay);
          return;
        }
        setCsvData([]);
        setCsvColumns([]);
      } else {
        setCsvData(data.data || []);
        setCsvColumns(data.columns || []);
        setCsvCurrentPage(data.page || 1);
        setCsvTotalPages(data.total_pages || 1);
      }
    } catch (error) {
      console.error('Error fetching CSV data:', error);
      if (retryCount < 4) {
        const delay = Math.min(1000 * Math.pow(2, retryCount), 4000);
        console.log(`Retrying CSV fetch due to network error in ${delay}ms... (attempt ${retryCount + 1}/4)`);
        setTimeout(() => {
          fetchCsvData(page, retryCount + 1);
        }, delay);
        return;
      }
      setCsvData([]);
      setCsvColumns([]);
    }
    setLoadingCsv(false);
  };

  const handleCsvSort = (key) => {
    let direction = 'asc';
    if (csvSortConfig.key === key && csvSortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setCsvSortConfig({ key, direction });
  };

  const handleCsvPageChange = (newPage) => {
    if (newPage >= 1 && newPage <= csvTotalPages) {
      fetchCsvData(newPage);
    }
  };

  return {
    csvData,
    csvColumns,
    csvCurrentPage,
    csvTotalPages,
    csvSortConfig,
    loadingCsv,
    fetchCsvData,
    handleCsvSort,
    handleCsvPageChange,
    setCsvSortConfig
  };
};

export default useCsvData;