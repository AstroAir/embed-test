"""Tests for progress tracking utility classes and functions."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from pdf_vector_system.utils.progress import (
    ProgressTracker, PerformanceTimer, track_progress, time_operation
)


class TestProgressTracker:
    """Test ProgressTracker class."""
    
    def test_context_manager(self):
        """Test ProgressTracker as context manager."""
        with ProgressTracker() as tracker:
            assert tracker is not None
            assert tracker._progress is not None
    
    def test_add_task(self):
        """Test adding tasks to progress tracker."""
        with ProgressTracker() as tracker:
            task_name = tracker.add_task("test_task", "Testing progress", total=100)
            
            assert task_name == "test_task"
            assert "test_task" in tracker._tasks
    
    def test_update_task(self):
        """Test updating task progress."""
        with ProgressTracker() as tracker:
            task_name = tracker.add_task("test_task", "Testing", total=100)
            
            # Should not raise an error
            tracker.update_task(task_name, advance=10)
            tracker.update_task(task_name, completed=50)
            tracker.update_task(task_name, description="Updated description")
    
    def test_update_nonexistent_task(self):
        """Test updating a task that doesn't exist."""
        with ProgressTracker() as tracker:
            # Should not raise an error, just ignore
            tracker.update_task("nonexistent", advance=10)
    
    def test_complete_task(self):
        """Test completing a task."""
        with ProgressTracker() as tracker:
            task_name = tracker.add_task("test_task", "Testing", total=100)
            tracker.complete_task(task_name)
            
            # Should not raise an error
    
    def test_complete_nonexistent_task(self):
        """Test completing a task that doesn't exist."""
        with ProgressTracker() as tracker:
            # Should not raise an error, just ignore
            tracker.complete_task("nonexistent")
    
    def test_add_task_outside_context(self):
        """Test adding task outside context manager."""
        tracker = ProgressTracker()
        
        with pytest.raises(RuntimeError, match="ProgressTracker must be used as context manager"):
            tracker.add_task("test", "description")
    
    def test_multiple_tasks(self):
        """Test managing multiple tasks."""
        with ProgressTracker() as tracker:
            task1 = tracker.add_task("task1", "First task", total=50)
            task2 = tracker.add_task("task2", "Second task", total=100)
            
            assert task1 == "task1"
            assert task2 == "task2"
            assert len(tracker._tasks) == 2
            
            # Update both tasks
            tracker.update_task(task1, advance=25)
            tracker.update_task(task2, advance=50)
            
            # Complete first task
            tracker.complete_task(task1)
    
    def test_task_with_no_total(self):
        """Test adding task with indeterminate progress."""
        with ProgressTracker() as tracker:
            task_name = tracker.add_task("indeterminate", "Processing...", total=None)
            
            assert task_name == "indeterminate"
            # Should be able to update without total
            tracker.update_task(task_name, description="Still processing...")


class TestPerformanceTimer:
    """Test PerformanceTimer class."""
    
    def test_context_manager(self):
        """Test PerformanceTimer as context manager."""
        with PerformanceTimer("test_operation") as timer:
            assert timer.start_time is not None
            time.sleep(0.01)  # Small delay
        
        assert timer.end_time is not None
        assert timer.duration > 0
    
    def test_manual_timing(self):
        """Test manual start/stop timing."""
        timer = PerformanceTimer("test_operation", log_result=False)
        
        timer.start()
        assert timer.start_time is not None
        
        time.sleep(0.01)  # Small delay
        
        duration = timer.stop()
        assert duration > 0
        assert timer.end_time is not None
        assert timer.duration == duration
    
    def test_lap_timing(self):
        """Test lap timing functionality."""
        timer = PerformanceTimer("test_operation", log_result=False)
        
        timer.start()
        time.sleep(0.01)
        
        lap_duration = timer.lap("checkpoint_1")
        assert lap_duration > 0
        
        time.sleep(0.01)
        
        lap_duration_2 = timer.lap("checkpoint_2")
        assert lap_duration_2 > lap_duration
    
    def test_lap_without_start(self):
        """Test lap timing without starting timer."""
        timer = PerformanceTimer("test_operation")
        
        with pytest.raises(RuntimeError, match="Timer not started"):
            timer.lap("checkpoint")
    
    def test_stop_without_start(self):
        """Test stopping timer without starting."""
        timer = PerformanceTimer("test_operation")
        
        with pytest.raises(RuntimeError, match="Timer not started"):
            timer.stop()
    
    def test_duration_property(self):
        """Test duration property."""
        timer = PerformanceTimer("test_operation", log_result=False)
        
        # Before timing
        assert timer.duration is None
        
        with timer:
            time.sleep(0.01)
        
        # After timing
        assert timer.duration is not None
        assert timer.duration > 0
    
    @patch('pdf_vector_system.utils.progress.logger')
    def test_automatic_logging(self, mock_logger):
        """Test automatic result logging."""
        with PerformanceTimer("test_operation", log_result=True):
            time.sleep(0.01)
        
        # Should have logged the result
        mock_logger.info.assert_called()
    
    def test_timer_properties(self):
        """Test timer properties and state."""
        timer = PerformanceTimer("test_op", log_result=False)
        
        # Initial state
        assert timer.operation_name == "test_op"
        assert timer.start_time is None
        assert timer.end_time is None
        assert timer.duration is None
        assert timer.log_result is False
        
        # After starting
        timer.start()
        assert timer.start_time is not None
        assert timer.end_time is None
        assert timer.duration is None
        
        # After stopping
        time.sleep(0.01)
        duration = timer.stop()
        assert timer.end_time is not None
        assert timer.duration == duration
        assert duration > 0
    
    def test_multiple_laps(self):
        """Test multiple lap measurements."""
        timer = PerformanceTimer("multi_lap_test", log_result=False)
        
        timer.start()
        
        laps = []
        for i in range(3):
            time.sleep(0.01)
            lap_time = timer.lap(f"lap_{i}")
            laps.append(lap_time)
        
        # Each lap should be longer than the previous
        assert laps[0] > 0
        assert laps[1] > laps[0]
        assert laps[2] > laps[1]


class TestTrackProgress:
    """Test track_progress context manager."""
    
    def test_track_progress_context(self):
        """Test track_progress context manager."""
        with track_progress("Test operation", total=100) as tracker:
            assert isinstance(tracker, ProgressTracker)
            assert "main" in tracker._tasks
    
    def test_track_progress_with_updates(self):
        """Test track_progress with progress updates."""
        with track_progress("Test operation", total=50) as tracker:
            # Should be able to update the main task
            tracker.update_task("main", advance=10)
            tracker.update_task("main", advance=20)
    
    def test_track_progress_no_total(self):
        """Test track_progress with indeterminate progress."""
        with track_progress("Indeterminate operation") as tracker:
            assert isinstance(tracker, ProgressTracker)
            assert "main" in tracker._tasks
            
            # Should be able to update description
            tracker.update_task("main", description="Processing...")
    
    def test_track_progress_with_custom_console(self):
        """Test track_progress with custom console."""
        from rich.console import Console
        custom_console = Console()
        
        with track_progress("Custom console test", console=custom_console) as tracker:
            assert isinstance(tracker, ProgressTracker)
            assert tracker.console == custom_console


class TestTimeOperation:
    """Test time_operation function."""
    
    def test_time_operation_factory(self):
        """Test time_operation factory function."""
        timer = time_operation("test_operation", log_result=False)
        
        assert isinstance(timer, PerformanceTimer)
        assert timer.operation_name == "test_operation"
        assert timer.log_result is False
    
    def test_time_operation_with_logging(self):
        """Test time_operation with logging enabled."""
        timer = time_operation("logged_operation", log_result=True)
        
        assert timer.log_result is True
    
    def test_time_operation_default_params(self):
        """Test time_operation with default parameters."""
        timer = time_operation("default_test")
        
        assert timer.operation_name == "default_test"
        # Default should be to log results
        assert timer.log_result is True
    
    @patch('pdf_vector_system.utils.progress.logger')
    def test_time_operation_usage(self, mock_logger):
        """Test using time_operation in practice."""
        timer = time_operation("practical_test", log_result=True)
        
        with timer:
            time.sleep(0.01)
            # Simulate some work
            result = "work_done"
        
        # Should have logged the operation
        mock_logger.info.assert_called()
        
        # Timer should have recorded the duration
        assert timer.duration > 0
