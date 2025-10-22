"""
Strategy optimization database manager
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional


class StrategyDatabase:
    """Manages storage and retrieval of strategy optimization runs"""

    def __init__(self, db_path: str = "strategy_optimization.db"):
        # Use absolute path relative to project root
        import os
        if not os.path.isabs(db_path):
            # Get project root (parent of backend/)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.db_path = os.path.join(project_root, db_path)
        else:
            self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Create database and tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                iteration_number INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                main_prompt TEXT NOT NULL,
                condition_prompt TEXT,
                nav_metrics TEXT,
                status TEXT NOT NULL,
                error_message TEXT,
                ticker_filters TEXT,
                date_filters TEXT,
                UNIQUE(run_id, iteration_number)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_run_id ON strategy_runs(run_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON strategy_runs(status)
        """)

        conn.commit()
        conn.close()

    def save_strategy_run(self, run_id: str, iteration: int, main_prompt: str,
                         condition_prompt: str, nav_metrics: Optional[Dict],
                         status: str, error_message: Optional[str] = None,
                         ticker_filters: Optional[Dict] = None,
                         date_filters: Optional[Dict] = None) -> int:
        """Save a strategy run result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO strategy_runs
            (run_id, iteration_number, timestamp, main_prompt, condition_prompt,
             nav_metrics, status, error_message, ticker_filters, date_filters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            iteration,
            datetime.now().isoformat(),
            main_prompt,
            condition_prompt,
            json.dumps(nav_metrics) if nav_metrics else None,
            status,
            error_message,
            json.dumps(ticker_filters) if ticker_filters else None,
            json.dumps(date_filters) if date_filters else None
        ))

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def get_run_results(self, run_id: str) -> List[Dict]:
        """Get all iterations for a specific run"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM strategy_runs
            WHERE run_id = ?
            ORDER BY iteration_number ASC
        """, (run_id,))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            result = dict(row)
            if result['nav_metrics']:
                result['nav_metrics'] = json.loads(result['nav_metrics'])
            if result['ticker_filters']:
                result['ticker_filters'] = json.loads(result['ticker_filters'])
            if result['date_filters']:
                result['date_filters'] = json.loads(result['date_filters'])
            results.append(result)

        return results

    def get_best_strategies(self, top_n: int = 10) -> List[Dict]:
        """Get top N strategies by ratio across all runs"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM strategy_runs
            WHERE status = 'success' AND nav_metrics IS NOT NULL
            ORDER BY id DESC
            LIMIT 100
        """)

        rows = cursor.fetchall()
        conn.close()

        # Parse and sort by ratio
        strategies = []
        for row in rows:
            result = dict(row)
            if result['nav_metrics']:
                result['nav_metrics'] = json.loads(result['nav_metrics'])
                strategies.append(result)

        # Sort by ratio (handle cases where ratio might be missing)
        strategies.sort(
            key=lambda x: x['nav_metrics'].get('ratio', 0) if x['nav_metrics'] else 0,
            reverse=True
        )

        return strategies[:top_n]

    def get_run_history_for_prompt(self, run_id: str, max_items: int = 5) -> List[Dict]:
        """Get recent successful runs formatted for feeding to Gemini"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT iteration_number, main_prompt, condition_prompt, nav_metrics
            FROM strategy_runs
            WHERE run_id = ? AND status = 'success' AND nav_metrics IS NOT NULL
            ORDER BY iteration_number DESC
            LIMIT ?
        """, (run_id, max_items))

        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            metrics = json.loads(row['nav_metrics']) if row['nav_metrics'] else {}
            history.append({
                'iteration': row['iteration_number'],
                'main_prompt': row['main_prompt'],
                'condition_prompt': row['condition_prompt'],
                'ratio': metrics.get('ratio', 0),
                'annual_return': metrics.get('annual_return', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'final_nav': metrics.get('final_nav', 0)
            })

        # Return in chronological order
        return list(reversed(history))

    def get_all_runs(self) -> List[Dict]:
        """Get all unique run IDs with summary stats"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                run_id,
                MIN(timestamp) as start_time,
                MAX(timestamp) as end_time,
                COUNT(*) as total_iterations,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_iterations
            FROM strategy_runs
            GROUP BY run_id
            ORDER BY start_time DESC
        """)

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def delete_run(self, run_id: str):
        """Delete all iterations for a run"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM strategy_runs WHERE run_id = ?", (run_id,))
        conn.commit()
        conn.close()
