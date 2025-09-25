# GUI Testing Framework

This directory contains comprehensive test coverage for the PDF Vector System GUI module. The testing framework is built using pytest with Qt-specific extensions for GUI testing.

## 📁 Directory Structure

```
tests/gui/
├── __init__.py                     # Package initialization
├── conftest.py                     # Shared fixtures and configuration
├── test_runner.py                  # Test runner utilities
├── test_main_window.py            # Main window integration tests
├── README.md                      # This documentation
├── test_widgets/                  # Widget-specific tests
│   ├── __init__.py
│   ├── test_base_widget.py        # Base widget functionality
│   ├── test_processing_widget.py  # PDF processing widget
│   ├── test_search_widget.py      # Search functionality widget
│   ├── test_document_widget.py    # Document management widget
│   ├── test_config_widget.py      # Configuration widget
│   ├── test_status_widget.py      # Status monitoring widget
│   └── test_log_widget.py         # Log display widget
├── test_controllers/              # Controller logic tests
│   ├── __init__.py
│   ├── test_main_controller.py    # Main application controller
│   ├── test_processing_controller.py # Processing logic controller
│   ├── test_search_controller.py  # Search logic controller
│   ├── test_document_controller.py # Document management controller
│   ├── test_config_controller.py  # Configuration controller
│   └── test_status_controller.py  # Status monitoring controller
├── test_dialogs/                  # Dialog component tests
│   ├── __init__.py
│   ├── test_about_dialog.py       # About dialog
│   ├── test_settings_dialog.py    # Settings dialog
│   ├── test_confirm_dialog.py     # Confirmation dialogs
│   ├── test_error_dialog.py       # Error display dialogs
│   └── test_progress_dialog.py    # Progress dialogs
├── test_utils/                    # Utility function tests
│   ├── __init__.py
│   ├── test_qt_utils.py          # Qt utility functions
│   ├── test_validators.py        # Input validation utilities
│   ├── test_icons.py             # Icon management
│   ├── test_styling.py           # UI styling utilities
│   └── test_threading.py         # Threading utilities
└── test_integration/             # Integration and workflow tests
    ├── __init__.py
    ├── test_app_lifecycle.py      # Application lifecycle
    ├── test_workflows.py          # End-to-end workflows
    └── test_error_scenarios.py    # Error handling scenarios
```

## 🧪 Test Categories

### 1. Widget Tests (`test_widgets/`)
Tests for individual GUI widgets including:
- **UI Initialization**: Widget creation, layout setup, component initialization
- **User Interactions**: Button clicks, text input, selection changes
- **Signal/Slot Connections**: Event handling and communication
- **Data Display**: Content rendering, table updates, progress indicators
- **State Management**: Widget state changes, enable/disable logic

### 2. Controller Tests (`test_controllers/`)
Tests for business logic controllers including:
- **Initialization**: Controller setup, dependency injection
- **Business Logic**: Core functionality, data processing
- **Background Tasks**: Threading, async operations
- **Error Handling**: Exception handling, error recovery
- **Signal Emission**: Event notification, status updates

### 3. Dialog Tests (`test_dialogs/`)
Tests for modal dialogs including:
- **Modal Behavior**: Dialog display, focus management
- **User Input**: Form validation, input handling
- **Return Values**: Dialog results, button responses
- **Error Display**: Error message formatting, details expansion

### 4. Utility Tests (`test_utils/`)
Tests for utility functions including:
- **Qt Utilities**: Window management, dialog helpers
- **Validation**: Input validation, format checking
- **Icons**: Icon loading, resource management
- **Styling**: Theme application, style utilities
- **Threading**: Background task management

### 5. Integration Tests (`test_integration/`)
End-to-end workflow tests including:
- **Application Lifecycle**: Startup, shutdown, cleanup
- **Complete Workflows**: Multi-step user scenarios
- **Cross-Component**: Inter-widget communication
- **Error Scenarios**: Error handling, recovery workflows

## 🔧 Test Configuration

### Dependencies
The testing framework requires these additional packages:
```bash
uv add --group dev pytest-qt pytest-mock pytest-asyncio pytest-html pytest-cov
```

### Markers
Tests are organized using pytest markers:
- `@pytest.mark.gui` - All GUI tests
- `@pytest.mark.widget` - Widget-specific tests
- `@pytest.mark.controller` - Controller tests
- `@pytest.mark.dialog` - Dialog tests
- `@pytest.mark.utils` - Utility tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.smoke` - Basic functionality tests
- `@pytest.mark.performance` - Performance tests

### Fixtures
Key fixtures provided in `conftest.py`:
- `qapp` - Qt application instance
- `qtbot` - Qt test bot for UI interactions
- `mock_config` - Mock configuration object
- `mock_pipeline` - Mock PDF processing pipeline
- `mock_file_dialog` - Mock file dialogs
- `mock_message_box` - Mock message boxes
- `temp_dir` - Temporary directory for file operations

## 🚀 Running Tests

### Using the Test Runner
```bash
# Run all GUI tests
python tests/gui/test_runner.py --all

# Run specific categories
python tests/gui/test_runner.py --widgets
python tests/gui/test_runner.py --controllers
python tests/gui/test_runner.py --integration

# Run by marker
python tests/gui/test_runner.py --marker gui
python tests/gui/test_runner.py --marker smoke

# Generate HTML report
python tests/gui/test_runner.py --report
```

### Using pytest directly
```bash
# Run all GUI tests with coverage
pytest tests/gui/ -v --cov=pdf_vector_system.gui --cov-report=html

# Run specific test categories
pytest tests/gui/test_widgets/ -v -m widget
pytest tests/gui/test_controllers/ -v -m controller

# Run specific test file
pytest tests/gui/test_widgets/test_processing_widget.py -v

# Run with specific markers
pytest tests/gui/ -v -m "gui and not integration"
```

## 📝 Writing Tests

### Widget Test Pattern
```python
@pytest.mark.gui
@pytest.mark.widget
class TestMyWidget:
    """Test cases for MyWidget."""
    
    def test_widget_initialization(self, qtbot, mock_config):
        """Test widget initializes correctly."""
        widget = MyWidget(config=mock_config)
        qtbot.addWidget(widget)
        
        assert widget is not None
        assert widget.config == mock_config
        
    def test_user_interaction(self, qtbot, mock_config):
        """Test user interaction handling."""
        widget = MyWidget(config=mock_config)
        qtbot.addWidget(widget)
        
        # Simulate button click
        qtbot.mouseClick(widget.my_button, Qt.MouseButton.LeftButton)
        
        # Verify expected behavior
        assert widget.some_state_changed
```

### Controller Test Pattern
```python
@pytest.mark.gui
@pytest.mark.controller
class TestMyController:
    """Test cases for MyController."""
    
    def test_controller_initialization(self, mock_config, mock_pipeline):
        """Test controller initializes correctly."""
        controller = MyController(mock_config, mock_pipeline)
        
        assert controller.config == mock_config
        assert controller.pipeline == mock_pipeline
        
    def test_business_logic(self, mock_config, mock_pipeline):
        """Test core business logic."""
        controller = MyController(mock_config, mock_pipeline)
        
        # Mock dependencies
        mock_pipeline.some_method.return_value = expected_result
        
        # Execute logic
        result = controller.do_something()
        
        # Verify results
        assert result == expected_result
        mock_pipeline.some_method.assert_called_once()
```

### Integration Test Pattern
```python
@pytest.mark.gui
@pytest.mark.integration
class TestMyWorkflow:
    """Test complete workflow scenarios."""
    
    def test_complete_workflow(self, qtbot, mock_config, mock_pipeline):
        """Test end-to-end workflow."""
        window = MainWindow(config=mock_config)
        qtbot.addWidget(window)
        
        # Step 1: Navigate to tab
        window.tab_widget.setCurrentWidget(window.my_widget)
        
        # Step 2: Perform actions
        qtbot.mouseClick(window.my_widget.action_button, Qt.MouseButton.LeftButton)
        
        # Step 3: Verify results
        assert expected_outcome_occurred
```

## 🎯 Best Practices

### 1. Test Organization
- Group related tests in classes
- Use descriptive test method names
- Follow the AAA pattern (Arrange, Act, Assert)
- Keep tests focused and atomic

### 2. Mocking Strategy
- Mock external dependencies (file system, network, database)
- Use `pytest-mock` for clean mock management
- Mock at the appropriate abstraction level
- Verify mock interactions when relevant

### 3. Qt-Specific Testing
- Always use `qtbot.addWidget()` for widgets under test
- Use `qtbot.mouseClick()` and `qtbot.keyClicks()` for interactions
- Wait for signals with `qtbot.waitSignal()` when needed
- Test both positive and negative scenarios

### 4. Error Testing
- Test error conditions and edge cases
- Verify error messages and user feedback
- Test recovery scenarios
- Use appropriate exception assertions

### 5. Performance Considerations
- Keep tests fast and focused
- Use mocks to avoid slow operations
- Consider using `pytest-benchmark` for performance tests
- Avoid unnecessary UI rendering in unit tests

## 📊 Coverage Goals

Target coverage levels:
- **Overall GUI Coverage**: ≥ 80%
- **Widget Coverage**: ≥ 85%
- **Controller Coverage**: ≥ 90%
- **Utility Coverage**: ≥ 95%
- **Integration Coverage**: ≥ 70%

## 🐛 Debugging Tests

### Common Issues
1. **Qt Application Context**: Ensure `qapp` fixture is used
2. **Widget Cleanup**: Use `qtbot.addWidget()` for proper cleanup
3. **Signal Timing**: Use `qtbot.waitSignal()` for async operations
4. **Mock Configuration**: Verify mock setup matches actual usage

### Debug Tools
- Use `pytest -s` to see print statements
- Use `pytest --pdb` to drop into debugger on failure
- Use `qtbot.screenshot()` to capture widget state
- Enable Qt logging for detailed Qt behavior

## 📈 Continuous Integration

The GUI tests are designed to run in CI environments:
- Use virtual display (Xvfb) on Linux
- Configure appropriate Qt platform plugins
- Set environment variables for headless testing
- Generate coverage reports and test artifacts

Example CI configuration:
```yaml
- name: Run GUI Tests
  run: |
    export QT_QPA_PLATFORM=offscreen
    pytest tests/gui/ --cov=pdf_vector_system.gui --cov-report=xml
  env:
    DISPLAY: :99
```
