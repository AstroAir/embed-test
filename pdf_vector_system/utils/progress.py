"""Progress tracking utilities for PDF Vector System."""

import time
from typing import Optional, Any, Iterator, Dict
from contextlib import contextmanager

from rich.console import Console
from rich.progress import (
    Progress,
    TaskID,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)
from loguru import logger


class ProgressTracker:
    """Enhanced progress tracker with rich console output and logging."""
    
    def __init__(self, console: Optional[Console] = None):
        """
        Initialize progress tracker.
        
        Args:
            console: Rich console instance (creates new one if None)
        """
        self.console = console or Console()
        self._progress: Optional[Progress] = None
        self._tasks: dict[str, TaskID] = {}
    
    def __enter__(self) -> "ProgressTracker":
        """Enter context manager."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        self._progress.__enter__()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        if self._progress:
            self._progress.__exit__(exc_type, exc_val, exc_tb)
            self._progress = None
            self._tasks.clear()
    
    def add_task(
        self,
        name: str,
        description: str,
        total: Optional[int] = None,
        **kwargs: Any
    ) -> str:
        """
        Add a new progress task.
        
        Args:
            name: Unique name for the task
            description: Description to display
            total: Total number of steps (None for indeterminate)
            **kwargs: Additional arguments for rich Progress.add_task
            
        Returns:
            Task name for updating progress
        """
        if not self._progress:
            raise RuntimeError("ProgressTracker must be used as context manager")
        
        task_id = self._progress.add_task(description, total=total, **kwargs)
        self._tasks[name] = task_id
        
        logger.debug(f"Added progress task: {name} - {description}")
        return name
    
    def update_task(
        self,
        name: str,
        advance: Optional[int] = None,
        completed: Optional[int] = None,
        description: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Update progress for a task.
        
        Args:
            name: Task name
            advance: Number of steps to advance
            completed: Set absolute completed count
            description: Update task description
            **kwargs: Additional arguments for rich Progress.update
        """
        if not self._progress or name not in self._tasks:
            return
        
        task_id = self._tasks[name]
        update_kwargs = kwargs.copy()
        
        if advance is not None:
            update_kwargs["advance"] = advance
        if completed is not None:
            update_kwargs["completed"] = completed
        if description is not None:
            update_kwargs["description"] = description
        
        self._progress.update(task_id, **update_kwargs)
    
    def complete_task(self, name: str) -> None:
        """
        Mark a task as completed.
        
        Args:
            name: Task name
        """
        if not self._progress or name not in self._tasks:
            return
        
        task_id = self._tasks[name]
        task = self._progress.tasks[task_id]
        
        if task.total is not None:
            self._progress.update(task_id, completed=task.total)
        
        logger.debug(f"Completed progress task: {name}")
    
    def remove_task(self, name: str) -> None:
        """
        Remove a task from progress tracking.
        
        Args:
            name: Task name
        """
        if not self._progress or name not in self._tasks:
            return
        
        task_id = self._tasks[name]
        self._progress.remove_task(task_id)
        del self._tasks[name]
        
        logger.debug(f"Removed progress task: {name}")


@contextmanager
def track_progress(
    description: str,
    total: Optional[int] = None,
    console: Optional[Console] = None
) -> Iterator[ProgressTracker]:
    """
    Context manager for simple progress tracking.
    
    Args:
        description: Task description
        total: Total number of steps
        console: Rich console instance
        
    Yields:
        ProgressTracker instance with a single task
    """
    with ProgressTracker(console) as tracker:
        task_name = tracker.add_task("main", description, total)
        yield tracker


class PerformanceTimer:
    """Timer for measuring operation performance."""
    
    def __init__(self, operation_name: str, log_result: bool = True):
        """
        Initialize performance timer.
        
        Args:
            operation_name: Name of the operation being timed
            log_result: Whether to log the result automatically
        """
        self.operation_name = operation_name
        self.log_result = log_result
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self) -> "PerformanceTimer":
        """Start timing."""
        self.start_time = time.time()
        logger.debug(f"Started timing: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop timing and optionally log result."""
        self.end_time = time.time()

        if self.log_result and self.duration is not None:
            logger.info(f"Performance: {self.operation_name} completed in {self.duration:.2f}s")
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.start_time is None:
            return None
        
        end_time = self.end_time or time.time()
        return end_time - self.start_time
    
    def lap(self, lap_name: str) -> float:
        """
        Record a lap time.
        
        Args:
            lap_name: Name of the lap
            
        Returns:
            Lap duration in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        lap_time = time.time()
        lap_duration = lap_time - self.start_time
        
        logger.debug(f"Lap: {self.operation_name} - {lap_name}: {lap_duration:.2f}s")
        return lap_duration


def time_operation(operation_name: str, log_result: bool = True) -> PerformanceTimer:
    """
    Create a performance timer for an operation.
    
    Args:
        operation_name: Name of the operation
        log_result: Whether to log the result automatically
        
    Returns:
        PerformanceTimer instance
    """
    return PerformanceTimer(operation_name, log_result)
