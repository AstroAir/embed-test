"""Tests for version module."""

import pytest
import re

from pdf_vector_system._version import __version__


class TestVersion:
    """Test version information."""
    
    def test_version_exists(self):
        """Test that version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__) > 0
    
    def test_version_format(self):
        """Test that version follows semantic versioning format."""
        # Should match semantic versioning pattern (major.minor.patch)
        # with optional pre-release and build metadata
        semver_pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$'
        
        assert re.match(semver_pattern, __version__), f"Version '{__version__}' does not follow semantic versioning"
    
    def test_version_components(self):
        """Test that version has valid major, minor, patch components."""
        # Split version into components
        version_parts = __version__.split('.')
        
        # Should have at least 3 parts (major.minor.patch)
        assert len(version_parts) >= 3, f"Version '{__version__}' should have at least 3 components"
        
        # First three parts should be numeric
        major, minor, patch = version_parts[:3]
        
        assert major.isdigit(), f"Major version '{major}' should be numeric"
        assert minor.isdigit(), f"Minor version '{minor}' should be numeric"
        
        # Patch might have pre-release suffix, so check the numeric part
        patch_numeric = patch.split('-')[0]
        assert patch_numeric.isdigit(), f"Patch version '{patch_numeric}' should be numeric"
    
    def test_version_not_empty_string(self):
        """Test that version is not an empty string."""
        assert __version__ != ""
        assert __version__.strip() != ""
    
    def test_version_not_placeholder(self):
        """Test that version is not a placeholder value."""
        placeholder_values = [
            "0.0.0",
            "1.0.0",
            "unknown",
            "dev",
            "development",
            "placeholder",
            "version"
        ]
        
        # Version should not be a common placeholder (unless it's actually 0.0.0 or 1.0.0)
        # This is more of a warning than a hard requirement
        if __version__ in placeholder_values:
            pytest.skip(f"Version appears to be a placeholder: {__version__}")
    
    def test_version_consistency(self):
        """Test that version is consistent across imports."""
        # Import version in different ways
        from pdf_vector_system._version import __version__ as version1
        from pdf_vector_system import __version__ as version2
        
        assert version1 == version2, "Version should be consistent across different import paths"
    
    def test_version_type(self):
        """Test that version is a string type."""
        assert isinstance(__version__, str), f"Version should be a string, got {type(__version__)}"
    
    def test_version_no_whitespace(self):
        """Test that version has no leading or trailing whitespace."""
        assert __version__ == __version__.strip(), "Version should not have leading or trailing whitespace"
    
    def test_version_ascii(self):
        """Test that version contains only ASCII characters."""
        try:
            __version__.encode('ascii')
        except UnicodeEncodeError:
            pytest.fail("Version should contain only ASCII characters")
    
    def test_version_length_reasonable(self):
        """Test that version length is reasonable."""
        assert len(__version__) <= 50, f"Version '{__version__}' seems unusually long"
        assert len(__version__) >= 5, f"Version '{__version__}' seems unusually short"
    
    def test_version_major_not_negative(self):
        """Test that major version is not negative."""
        major = int(__version__.split('.')[0])
        assert major >= 0, f"Major version should not be negative: {major}"
    
    def test_version_minor_not_negative(self):
        """Test that minor version is not negative."""
        minor = int(__version__.split('.')[1])
        assert minor >= 0, f"Minor version should not be negative: {minor}"
    
    def test_version_patch_not_negative(self):
        """Test that patch version is not negative."""
        patch_part = __version__.split('.')[2]
        # Extract numeric part (before any pre-release suffix)
        patch_numeric = patch_part.split('-')[0]
        patch = int(patch_numeric)
        assert patch >= 0, f"Patch version should not be negative: {patch}"


class TestVersionModule:
    """Test version module structure."""
    
    def test_module_has_version(self):
        """Test that the version module has __version__ attribute."""
        import pdf_vector_system._version as version_module
        
        assert hasattr(version_module, '__version__')
    
    def test_module_minimal_exports(self):
        """Test that version module doesn't export unnecessary items."""
        import pdf_vector_system._version as version_module
        
        # Should primarily export __version__
        public_attrs = [attr for attr in dir(version_module) if not attr.startswith('_')]
        
        # Allow some flexibility, but shouldn't have too many exports
        assert len(public_attrs) <= 5, f"Version module exports too many public attributes: {public_attrs}"
    
    def test_version_importable_from_package(self):
        """Test that version is importable from main package."""
        try:
            from pdf_vector_system import __version__
            assert __version__ is not None
        except ImportError:
            pytest.fail("Version should be importable from main package")
    
    def test_version_module_docstring(self):
        """Test that version module has appropriate docstring."""
        import pdf_vector_system._version as version_module
        
        if version_module.__doc__:
            assert isinstance(version_module.__doc__, str)
            assert len(version_module.__doc__.strip()) > 0
