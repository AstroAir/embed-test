# VectorFlow Examples

This directory contains comprehensive examples demonstrating all aspects of the VectorFlow project. Examples are organized by functional area and complexity level to help you learn and implement the system effectively.

## 📁 Directory Structure

### [01_basic_usage/](01_basic_usage/)

Fundamental examples for getting started with VectorFlow.

- Simple PDF processing
- Basic configuration
- First search operations
- Health checking

### [02_embedding_providers/](02_embedding_providers/)

Comprehensive examples for all available embedding providers.

- Sentence Transformers (local)
- OpenAI embeddings
- Cohere embeddings
- Hugging Face models
- Google USE and Gemini
- Provider comparison and monitoring

### [03_cli_usage/](03_cli_usage/)

Complete command-line interface examples and automation scripts.

- All CLI commands
- Batch processing
- Search operations
- Collection management
- Automation workflows

### [04_gui_applications/](04_gui_applications/)

Graphical user interface examples and integration patterns.

- Basic GUI usage
- Custom integrations
- GUI features
- GUI automation

### [05_vector_database/](05_vector_database/)

Vector database operations and search patterns.

- Search techniques
- Collection management
- Metadata filtering
- Performance optimization

### [06_text_processing/](06_text_processing/)

Text processing strategies and optimization techniques.

- Chunking strategies
- Text cleaning
- Custom splitters
- Quality assessment

### [07_configuration/](07_configuration/)

Configuration management patterns for different environments.

- Environment variables
- Programmatic configuration
- Production setups
- Security considerations

### [08_integration/](08_integration/)

Integration with external systems and workflow automation.

- Batch processing workflows
- API integrations
- Cloud storage
- Microservices

### [09_performance/](09_performance/)

Performance optimization, monitoring, and benchmarking.

- Benchmarking
- Memory optimization
- Parallel processing
- Monitoring

### [10_production/](10_production/)

Production deployment, monitoring, and maintenance.

- Deployment patterns
- Container orchestration
- Monitoring and alerting
- Security hardening

## 🚀 Quick Start

### Prerequisites

1. **Install VectorFlow**

   ```bash
   pip install vectorflow
   # or
   uv add vectorflow
   ```

2. **Set up environment** (for cloud providers)

   ```bash
   export OPENAI_API_KEY="your-openai-key"
   export COHERE_API_KEY="your-cohere-key"
   export GOOGLE_GEMINI_API_KEY="your-gemini-key"
   ```

3. **Prepare sample data**

   ```bash
   # Create sample PDFs directory
   mkdir -p sample_pdfs
   # Add your PDF files to this directory
   ```

### Running Your First Example

```bash
# Start with basic usage
cd examples/01_basic_usage
python simple_pdf_processing.py

# Try CLI commands
cd ../03_cli_usage
python basic_cli_operations.py

# Explore embedding providers
cd ../02_embedding_providers
python sentence_transformers_examples.py
```

## 🔍 Quality Assurance

This examples collection includes comprehensive quality assurance tools:

### Code Quality Review

```bash
python code_quality_review.py
```

- Checks coding standards compliance
- Validates documentation completeness
- Reviews code structure and style
- Provides improvement recommendations

### Example Testing

```bash
python test_all_examples.py
```

- Validates syntax and imports
- Tests functionality and execution
- Checks documentation standards
- Generates comprehensive reports

### Contributing Guidelines

See `CONTRIBUTING.md` for:

- Adding new examples
- Quality standards
- Review process
- Testing requirements

### Quality Standards

All examples follow strict quality standards:

- ✅ Comprehensive documentation with required sections
- ✅ Type hints and proper error handling
- ✅ PEP 8 compliance and code formatting
- ✅ Functional testing and validation
- ✅ Clear learning objectives and prerequisites

## 📚 Learning Path

### Beginner

1. **01_basic_usage/** - Start here to understand core concepts
2. **03_cli_usage/** - Learn command-line operations
3. **02_embedding_providers/** - Explore different embedding options

### Intermediate

1. **05_vector_database/** - Search and database operations
2. **06_text_processing/** - Optimize text processing
3. **07_configuration/** - Master configuration management

### Expert

1. **08_integration/** - Build complex workflows
2. **09_performance/** - Optimize for production
3. **10_production/** - Deploy and maintain in production

## 🔧 Configuration Templates

### Basic Configuration (.env)

```bash
# Copy to .env and customize
EMBEDDING__MODEL_TYPE=sentence-transformers
EMBEDDING__MODEL_NAME=all-MiniLM-L6-v2
CHROMA_DB__PERSIST_DIRECTORY=./chroma_db
DEBUG=false
```

### Production Configuration

```bash
EMBEDDING__MODEL_TYPE=openai
EMBEDDING__MODEL_NAME=text-embedding-3-small
OPENAI_API_KEY=your-production-key
CHROMA_DB__PERSIST_DIRECTORY=/data/chroma_db
LOGGING__LEVEL=INFO
LOGGING__FILE_PATH=/var/log/vectorflow.log
MAX_WORKERS=8
```

## 🛠️ Development Setup

For contributing to examples or developing custom solutions:

```bash
# Clone the repository
git clone https://github.com/your-username/vectorflow.git
cd vectorflow

# Install development dependencies
uv sync --extra dev

# Run examples in development mode
cd examples
python -m examples.01_basic_usage.simple_pdf_processing
```

## 📖 Documentation

- **API Reference**: See main README.md for complete API documentation
- **Configuration Guide**: Check 07_configuration/ for detailed configuration options
- **Performance Guide**: See 09_performance/ for optimization techniques
- **Deployment Guide**: Check 10_production/ for deployment strategies

## 🤝 Contributing

Found an issue or want to add an example?

1. Check existing examples for similar patterns
2. Follow the established directory structure
3. Include comprehensive documentation
4. Add error handling and logging
5. Test with different configurations

## 📝 Example Template

When creating new examples, follow this structure:

```python
"""
Brief description of what this example demonstrates.

Prerequisites:
- List any special requirements
- API keys needed
- Dependencies

Usage:
    python example_name.py
"""

import os
from pathlib import Path
from vectorflow import Config, PDFVectorPipeline


def main():
    """Main example function with clear documentation."""
    # Configuration
    config = Config()
    # ... configuration code

    # Example logic
    # ... implementation

    # Results and cleanup
    # ... results handling


if __name__ == "__main__":
    main()
```

## 🆘 Getting Help

- **Issues**: Check the main repository issues
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Refer to individual example READMEs
- **Community**: Join the community discussions

## 📄 License

All examples are provided under the same license as the main project (MIT License).
