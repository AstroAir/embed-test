"""Mock implementations for file system operations."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from unittest.mock import Mock, patch, mock_open
from contextlib import contextmanager


class MockFileSystem:
    """Mock file system for testing file operations."""
    
    def __init__(self):
        self.files: Dict[str, str] = {}
        self.directories: set[str] = set()
        self.file_stats: Dict[str, Dict[str, Any]] = {}
        self.access_log: List[str] = []
    
    def create_file(self, path: Union[str, Path], content: str = "", size: Optional[int] = None) -> None:
        """Create a mock file with content."""
        path_str = str(path)
        self.files[path_str] = content
        
        # Create parent directories
        parent = str(Path(path).parent)
        if parent != path_str:
            self.directories.add(parent)
        
        # Set file stats
        self.file_stats[path_str] = {
            "st_size": size or len(content),
            "st_mtime": 1640995200.0,  # Fixed timestamp
            "st_ctime": 1640995200.0,
            "st_atime": 1640995200.0
        }
        
        self.access_log.append(f"CREATE: {path_str}")
    
    def create_directory(self, path: Union[str, Path]) -> None:
        """Create a mock directory."""
        path_str = str(path)
        self.directories.add(path_str)
        self.access_log.append(f"MKDIR: {path_str}")
    
    def exists(self, path: Union[str, Path]) -> bool:
        """Check if file or directory exists."""
        path_str = str(path)
        self.access_log.append(f"EXISTS: {path_str}")
        return path_str in self.files or path_str in self.directories
    
    def is_file(self, path: Union[str, Path]) -> bool:
        """Check if path is a file."""
        path_str = str(path)
        self.access_log.append(f"ISFILE: {path_str}")
        return path_str in self.files
    
    def is_dir(self, path: Union[str, Path]) -> bool:
        """Check if path is a directory."""
        path_str = str(path)
        self.access_log.append(f"ISDIR: {path_str}")
        return path_str in self.directories
    
    def read_file(self, path: Union[str, Path]) -> str:
        """Read file content."""
        path_str = str(path)
        self.access_log.append(f"READ: {path_str}")
        if path_str not in self.files:
            raise FileNotFoundError(f"File not found: {path_str}")
        return self.files[path_str]
    
    def write_file(self, path: Union[str, Path], content: str) -> None:
        """Write content to file."""
        path_str = str(path)
        self.files[path_str] = content
        self.file_stats[path_str] = {
            "st_size": len(content),
            "st_mtime": 1640995200.0,
            "st_ctime": 1640995200.0,
            "st_atime": 1640995200.0
        }
        self.access_log.append(f"WRITE: {path_str}")
    
    def delete_file(self, path: Union[str, Path]) -> None:
        """Delete a file."""
        path_str = str(path)
        if path_str in self.files:
            del self.files[path_str]
            del self.file_stats[path_str]
        self.access_log.append(f"DELETE: {path_str}")
    
    def get_file_stats(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Get file statistics."""
        path_str = str(path)
        self.access_log.append(f"STAT: {path_str}")
        if path_str not in self.file_stats:
            raise FileNotFoundError(f"File not found: {path_str}")
        return self.file_stats[path_str].copy()
    
    def list_directory(self, path: Union[str, Path]) -> List[str]:
        """List directory contents."""
        path_str = str(path)
        self.access_log.append(f"LISTDIR: {path_str}")
        
        if path_str not in self.directories:
            raise NotADirectoryError(f"Not a directory: {path_str}")
        
        contents = []
        
        # Add files in this directory
        for file_path in self.files:
            if str(Path(file_path).parent) == path_str:
                contents.append(Path(file_path).name)
        
        # Add subdirectories
        for dir_path in self.directories:
            if str(Path(dir_path).parent) == path_str:
                contents.append(Path(dir_path).name)
        
        return contents
    
    def clear(self) -> None:
        """Clear all files and directories."""
        self.files.clear()
        self.directories.clear()
        self.file_stats.clear()
        self.access_log.clear()


@contextmanager
def mock_file_system(mock_fs: Optional[MockFileSystem] = None):
    """Context manager to mock file system operations."""
    if mock_fs is None:
        mock_fs = MockFileSystem()
    
    def mock_path_exists(path):
        return mock_fs.exists(path)
    
    def mock_path_is_file(path):
        return mock_fs.is_file(path)
    
    def mock_path_is_dir(path):
        return mock_fs.is_dir(path)
    
    def mock_path_stat(path):
        stats = mock_fs.get_file_stats(path)
        mock_stat = Mock()
        for key, value in stats.items():
            setattr(mock_stat, key, value)
        return mock_stat
    
    def mock_path_mkdir(path, parents=False, exist_ok=False):
        if not exist_ok and mock_fs.exists(path):
            raise FileExistsError(f"Directory exists: {path}")
        mock_fs.create_directory(path)
    
    def mock_open_file(path, mode='r', **kwargs):
        if 'r' in mode:
            content = mock_fs.read_file(path)
            return mock_open(read_data=content)()
        elif 'w' in mode or 'a' in mode:
            # For write mode, return a mock that captures written content
            mock_file = mock_open()()
            original_write = mock_file.write
            
            def capture_write(content):
                mock_fs.write_file(path, content)
                return original_write(content)
            
            mock_file.write = capture_write
            return mock_file
    
    with patch('pathlib.Path.exists', side_effect=mock_path_exists), \
         patch('pathlib.Path.is_file', side_effect=mock_path_is_file), \
         patch('pathlib.Path.is_dir', side_effect=mock_path_is_dir), \
         patch('pathlib.Path.stat', side_effect=mock_path_stat), \
         patch('pathlib.Path.mkdir', side_effect=mock_path_mkdir), \
         patch('builtins.open', side_effect=mock_open_file), \
         patch('os.path.exists', side_effect=mock_path_exists), \
         patch('os.path.isfile', side_effect=mock_path_is_file), \
         patch('os.path.isdir', side_effect=mock_path_is_dir):
        
        yield mock_fs


class MockTemporaryDirectory:
    """Mock temporary directory for testing."""
    
    def __init__(self, suffix=None, prefix=None, dir=None):
        self.suffix = suffix
        self.prefix = prefix
        self.dir = dir
        self.name = "/tmp/mock_temp_dir"
        self._created = False
    
    def __enter__(self):
        self._created = True
        return self.name
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._created = False
    
    def cleanup(self):
        """Mock cleanup method."""
        self._created = False


@contextmanager
def mock_temporary_directory():
    """Context manager for mocking temporary directories."""
    with patch('tempfile.TemporaryDirectory', MockTemporaryDirectory):
        yield


def create_test_pdf_file(temp_dir: Path, filename: str = "test.pdf", content: str = "Mock PDF") -> Path:
    """Create a test PDF file in the temporary directory."""
    pdf_path = temp_dir / filename
    pdf_path.write_text(content)
    return pdf_path


def create_test_files(temp_dir: Path, files: Dict[str, str]) -> Dict[str, Path]:
    """Create multiple test files in the temporary directory."""
    created_files = {}
    for filename, content in files.items():
        file_path = temp_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        created_files[filename] = file_path
    return created_files
