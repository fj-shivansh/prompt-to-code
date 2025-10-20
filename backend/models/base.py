"""
Base data models
"""
from dataclasses import dataclass
from typing import Any, Optional, List


@dataclass
class TestResult:
    """Result from code execution test"""
    success: bool
    result: Any
    execution_time: float
    error: Optional[str] = None


@dataclass
class CodeGeneration:
    """Generated code with metadata"""
    code: str
    explanation: str
    task: str
    requirements: List[str] = None
    tokens: Optional[dict] = None
