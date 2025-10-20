"""
Database manager for stock data operations
"""
import sqlite3
from typing import List, Dict, Any, Optional


class DatabaseManager:
    """Manages database operations for stock market data"""

    def __init__(self, db_path: str = "historical_data_500_tickers_with_gains.db"):
        self.db_path = db_path

    def get_sample_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get sample data from database for testing"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct "
            "FROM stock_data "
            "WHERE Adj_Close IS NOT NULL "
            "AND Daily_Gain_Pct IS NOT NULL "
            "AND Forward_Gain_Pct IS NOT NULL "
            "LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "Date": row[0],
                "Ticker": row[1],
                "Adj_Close": row[2],
                "Daily_Gain_Pct": row[3],
                "Forward_Gain_Pct": row[4]
            }
            for row in rows
        ]

    def get_all_data(self) -> List[Dict[str, Any]]:
        """Get all data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct "
            "FROM stock_data "
            "WHERE Adj_Close IS NOT NULL "
            "AND Daily_Gain_Pct IS NOT NULL "
            "AND Forward_Gain_Pct IS NOT NULL"
        )
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "Date": row[0],
                "Ticker": row[1],
                "Adj_Close": row[2],
                "Daily_Gain_Pct": row[3],
                "Forward_Gain_Pct": row[4]
            }
            for row in rows
        ]

    def get_filtered_data(
        self,
        ticker_list: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get filtered data based on tickers and date range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT Date, Ticker, Adj_Close, Daily_Gain_Pct, Forward_Gain_Pct
            FROM stock_data
            WHERE Adj_Close IS NOT NULL
            AND Daily_Gain_Pct IS NOT NULL
            AND Forward_Gain_Pct IS NOT NULL
        """
        params = []

        if ticker_list:
            placeholders = ','.join(['?' for _ in ticker_list])
            query += f" AND Ticker IN ({placeholders})"
            params.extend(ticker_list)

        if start_date:
            query += " AND Date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND Date <= ?"
            params.append(end_date)

        query += " ORDER BY Ticker, Date"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "Date": row[0],
                "Ticker": row[1],
                "Adj_Close": row[2],
                "Daily_Gain_Pct": row[3],
                "Forward_Gain_Pct": row[4]
            }
            for row in rows
        ]

    def get_unique_tickers(self) -> List[str]:
        """Get list of unique tickers"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT Ticker
            FROM stock_data
            WHERE Adj_Close IS NOT NULL
            AND Daily_Gain_Pct IS NOT NULL
            AND Forward_Gain_Pct IS NOT NULL
            ORDER BY Ticker
        """)

        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def get_date_range(self) -> Dict[str, str]:
        """Get min and max dates available in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT MIN(Date) as min_date, MAX(Date) as max_date
            FROM stock_data
            WHERE Adj_Close IS NOT NULL
        """)

        result = cursor.fetchone()
        conn.close()

        return {
            'min_date': result[0],
            'max_date': result[1]
        }

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                COUNT(DISTINCT Ticker) as total_tickers,
                COUNT(*) as total_rows,
                MIN(Date) as min_date,
                MAX(Date) as max_date
            FROM stock_data
            WHERE Adj_Close IS NOT NULL
        """)

        stats = cursor.fetchone()
        conn.close()

        return {
            'total_tickers': stats[0],
            'total_rows': stats[1],
            'min_date': stats[2],
            'max_date': stats[3],
            'avg_rows_per_ticker': stats[1] // stats[0] if stats[0] > 0 else 0
        }

    def get_random_tickers(
        self,
        count: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[str]:
        """Get random tickers with optional date filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT DISTINCT Ticker
            FROM stock_data
            WHERE Adj_Close IS NOT NULL
            AND Daily_Gain_Pct IS NOT NULL
            AND Forward_Gain_Pct IS NOT NULL
        """
        params = []

        if start_date and end_date:
            query += " AND Date >= ? AND Date <= ?"
            params.extend([start_date, end_date])

        query += " ORDER BY RANDOM() LIMIT ?"
        params.append(count)

        cursor.execute(query, params)
        tickers = [row[0] for row in cursor.fetchall()]
        conn.close()

        return tickers
