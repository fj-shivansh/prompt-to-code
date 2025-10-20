"""
Process management for background tasks
"""
import threading
import subprocess


class ProcessManager:
    """Manages background processes"""

    def __init__(self):
        self.current_process = None
        self.stop_requested = False
        self.process_lock = threading.Lock()

    def stop_processing(self) -> dict:
        """Stop current processing"""
        try:
            with self.process_lock:
                self.stop_requested = True

                # If there's a current subprocess, terminate it
                if self.current_process and self.current_process.poll() is None:
                    self.current_process.terminate()
                    try:
                        self.current_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.current_process.kill()
                        self.current_process.wait()

            return {
                'success': True,
                'message': 'Stop request sent successfully'
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to stop processing: {str(e)}'
            }

    def reset(self):
        """Reset the process manager state"""
        with self.process_lock:
            self.stop_requested = False
            self.current_process = None
