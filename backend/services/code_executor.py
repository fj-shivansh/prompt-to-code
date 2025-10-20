"""
Code execution service
"""
import os
import sys
import time
import subprocess
from typing import List, Any
from ..models.base import TestResult


class CodeExecutor:
    """Executes generated Python code"""

    def __init__(self):
        self.namespace = {
            "List": list, "Dict": dict, "Any": Any,
            "list": list, "dict": dict, "str": str, "float": float, "int": int,
            "typing": __import__("typing")
        }
        self.generated_file_path = "generated_code.py"
        self.current_process = None
        self.process_callback = None

    def write_code_to_file(self, code: str, filename: str = None) -> str:
        """Write generated code to separate file"""
        if filename is None:
            filename = self.generated_file_path

        full_code = f'''#!/usr/bin/env python3
"""
Generated Code Execution Module
This file contains AI-generated code for execution.
"""

{code}
'''

        with open(filename, "w") as f:
            f.write(full_code)

        return filename

    def install_requirements(self, requirements: List[str]) -> bool:
        """Install required packages using pip"""
        if not requirements:
            return True

        # Filter out built-in modules that shouldn't be installed via pip
        builtin_modules = {
            'sqlite3', 'os', 'sys', 'time', 'json', 'csv', 're', 'math',
            'random', 'datetime', 'collections', 'itertools', 'functools',
            'typing', 'dataclasses', 'abc', 'io', 'pathlib', 'traceback'
        }

        # Filter requirements to only include actual packages
        packages_to_install = [
            pkg for pkg in requirements
            if pkg.lower() not in builtin_modules and pkg.strip()
        ]

        if not packages_to_install:
            print("No external packages to install (all are built-in modules)")
            return True

        print(f"Installing requirements: {', '.join(packages_to_install)}")
        try:
            for package in packages_to_install:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install requirements: {e}")
            return False

    def execute_code(self, code: str, filename: str = None) -> TestResult:
        """Execute generated code directly as a standalone script"""
        start_time = time.time()

        try:
            # Write code to separate file
            code_file = self.write_code_to_file(code, filename)

            # Execute the code as a subprocess to capture output
            self.current_process = subprocess.Popen(
                [sys.executable, code_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Call the process callback if set (for tracking in backend)
            if self.process_callback:
                self.process_callback(self.current_process)

            try:
                stdout, stderr = self.current_process.communicate(timeout=500)
                result_returncode = self.current_process.returncode
            except subprocess.TimeoutExpired:
                self.current_process.kill()
                stdout, stderr = self.current_process.communicate()
                raise

            execution_time = time.time() - start_time

            # More robust success detection - check both return code and CSV file creation
            expected_csv = filename.replace('.py', '.csv').replace('generated_code_', 'output_')
            csv_exists = os.path.exists(expected_csv)

            # Consider success if return code is 0 OR None (and CSV was created)
            is_successful = (result_returncode == 0) or (result_returncode is None and csv_exists)

            if is_successful:
                # Debug logging
                print(f"=== DEBUG: Code Execution Success ===")
                print(f"STDOUT length: {len(stdout)}")
                print(f"Expected CSV file: {expected_csv}")
                print(f"CSV file exists: {csv_exists}")
                print(f"Return code: {result_returncode}")

                # Return the full output instead of just the last line
                full_output = stdout.strip()
                final_result = full_output if full_output else ""

                return TestResult(
                    success=True,
                    result=final_result,
                    execution_time=execution_time
                )
            else:
                return TestResult(
                    success=False,
                    result=None,
                    execution_time=execution_time,
                    error=f"Execution failed with return code {result_returncode}. STDERR: {stderr}. STDOUT: {stdout}"
                )

        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return TestResult(
                success=False,
                result=None,
                execution_time=execution_time,
                error="Code execution timed out after 500 seconds."
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                success=False,
                result=None,
                execution_time=execution_time,
                error=str(e)
            )
