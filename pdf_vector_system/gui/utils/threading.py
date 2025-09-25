"""
Threading utilities for PDF Vector System GUI.

This module provides thread management for background tasks in the GUI.
"""

from typing import Optional, Callable, Any, Dict
from PySide6.QtCore import QThread, QObject, Signal, QMutex, QWaitCondition
from PySide6.QtWidgets import QApplication


class WorkerSignals(QObject):
    """Signals for worker threads."""
    
    # Emitted when task starts
    started = Signal()
    
    # Emitted when task finishes successfully
    finished = Signal(object)  # result
    
    # Emitted when task encounters an error
    error = Signal(str)  # error message
    
    # Emitted for progress updates
    progress = Signal(int)  # percentage (0-100)
    
    # Emitted for status messages
    status = Signal(str)  # status message


class WorkerThread(QThread):
    """Generic worker thread for background tasks."""
    
    def __init__(self, task_func: Callable, *args, **kwargs):
        """
        Initialize worker thread.
        
        Args:
            task_func: Function to execute in background
            *args: Arguments for task function
            **kwargs: Keyword arguments for task function
        """
        super().__init__()
        
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        
        # Thread control
        self._mutex = QMutex()
        self._should_stop = False
        
    def run(self) -> None:
        """Execute the task in the background thread."""
        try:
            self.signals.started.emit()
            
            # Execute the task
            result = self.task_func(*self.args, **self.kwargs)
            
            # Check if we should stop
            self._mutex.lock()
            should_stop = self._should_stop
            self._mutex.unlock()
            
            if not should_stop:
                self.signals.finished.emit(result)
                
        except Exception as e:
            self.signals.error.emit(str(e))
            
    def stop(self) -> None:
        """Request the thread to stop."""
        self._mutex.lock()
        self._should_stop = True
        self._mutex.unlock()
        
    def should_stop(self) -> bool:
        """Check if thread should stop."""
        self._mutex.lock()
        should_stop = self._should_stop
        self._mutex.unlock()
        return should_stop


class TaskRunner(QObject):
    """Task runner for managing background operations."""
    
    # Signals
    task_started = Signal(str)  # task_id
    task_finished = Signal(str, object)  # task_id, result
    task_error = Signal(str, str)  # task_id, error_message
    task_progress = Signal(str, int)  # task_id, percentage
    task_status = Signal(str, str)  # task_id, status_message
    
    def __init__(self, parent: Optional[QObject] = None):
        """
        Initialize task runner.
        
        Args:
            parent: Parent QObject
        """
        super().__init__(parent)
        
        self._active_tasks: Dict[str, WorkerThread] = {}
        self._task_counter = 0
        
    def run_task(self, task_func: Callable, task_id: Optional[str] = None, 
                 *args, **kwargs) -> str:
        """
        Run a task in the background.
        
        Args:
            task_func: Function to execute
            task_id: Optional task identifier
            *args: Arguments for task function
            **kwargs: Keyword arguments for task function
            
        Returns:
            Task ID
        """
        # Generate task ID if not provided
        if task_id is None:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}"
            
        # Stop existing task with same ID
        if task_id in self._active_tasks:
            self.stop_task(task_id)
            
        # Create and start worker thread
        worker = WorkerThread(task_func, *args, **kwargs)
        
        # Connect signals
        worker.signals.started.connect(lambda: self.task_started.emit(task_id))
        worker.signals.finished.connect(lambda result: self._on_task_finished(task_id, result))
        worker.signals.error.connect(lambda error: self._on_task_error(task_id, error))
        worker.signals.progress.connect(lambda progress: self.task_progress.emit(task_id, progress))
        worker.signals.status.connect(lambda status: self.task_status.emit(task_id, status))
        
        # Store and start thread
        self._active_tasks[task_id] = worker
        worker.start()
        
        return task_id
        
    def stop_task(self, task_id: str) -> bool:
        """
        Stop a running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was stopped, False if not found
        """
        if task_id in self._active_tasks:
            worker = self._active_tasks[task_id]
            worker.stop()
            worker.wait(5000)  # Wait up to 5 seconds
            
            if worker.isRunning():
                worker.terminate()
                worker.wait(2000)  # Wait up to 2 more seconds
                
            del self._active_tasks[task_id]
            return True
            
        return False
        
    def stop_all_tasks(self) -> None:
        """Stop all running tasks."""
        task_ids = list(self._active_tasks.keys())
        for task_id in task_ids:
            self.stop_task(task_id)
            
    def is_task_running(self, task_id: str) -> bool:
        """
        Check if a task is running.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task is running
        """
        return task_id in self._active_tasks
        
    def get_active_tasks(self) -> list[str]:
        """
        Get list of active task IDs.
        
        Returns:
            List of active task IDs
        """
        return list(self._active_tasks.keys())
        
    def _on_task_finished(self, task_id: str, result: Any) -> None:
        """Handle task completion."""
        if task_id in self._active_tasks:
            del self._active_tasks[task_id]
        self.task_finished.emit(task_id, result)
        
    def _on_task_error(self, task_id: str, error: str) -> None:
        """Handle task error."""
        if task_id in self._active_tasks:
            del self._active_tasks[task_id]
        self.task_error.emit(task_id, error)


def run_in_background(func: Callable) -> Callable:
    """
    Decorator to run a function in the background.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        # Get the current QApplication
        app = QApplication.instance()
        if app:
            # Create a task runner if it doesn't exist
            if not hasattr(app, '_task_runner'):
                app._task_runner = TaskRunner()
            
            # Run the function in background
            return app._task_runner.run_task(func, *args, **kwargs)
        else:
            # No QApplication, run synchronously
            return func(*args, **kwargs)
            
    return wrapper
