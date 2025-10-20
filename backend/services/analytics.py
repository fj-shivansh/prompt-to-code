"""
Analytics service for test results
"""
from typing import List, Dict, Any
from ..models.base import CodeGeneration, TestResult


class Analytics:
    """Analytics for test results"""

    @staticmethod
    def analyze_results(
        generations: List[CodeGeneration],
        test_results: List[TestResult],
        data_stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate analytics from test results"""
        successful_tests = [r for r in test_results if r.success]
        failed_tests = [r for r in test_results if not r.success]

        analytics = {
            "summary": {
                "total_tests": len(test_results),
                "successful": len(successful_tests),
                "failed": len(failed_tests),
                "success_rate": len(successful_tests) / len(test_results) if test_results else 0
            },
            "performance": {
                "avg_execution_time": sum(r.execution_time for r in successful_tests) / len(successful_tests) if successful_tests else 0,
                "fastest_execution": min(r.execution_time for r in successful_tests) if successful_tests else 0,
                "slowest_execution": max(r.execution_time for r in successful_tests) if successful_tests else 0
            },
            "errors": [r.error for r in failed_tests],
            "data_stats": data_stats,
            "results": [r.result for r in successful_tests]
        }

        return analytics
