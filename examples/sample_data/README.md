# Sample Data for Examples

This directory contains sample data files used by the examples.

## Contents

### PDF Files

- `sample_research_paper.pdf` - Academic research paper for testing
- `sample_technical_doc.pdf` - Technical documentation example
- `sample_report.pdf` - Business report sample

### Configuration Templates

- `.env.example` - Example environment configuration
- `config_templates/` - Configuration templates for different scenarios

### Test Data

- `test_texts/` - Sample text files for testing text processing
- `embeddings_cache/` - Cached embeddings for faster testing

## Usage

Examples will automatically use files from this directory when available. If files are missing, examples will either:

1. Create minimal sample content automatically
2. Prompt you to add your own files
3. Skip tests that require specific data

## Adding Your Own Data

You can add your own PDF files to test with:

```bash
# Copy your PDFs to the sample_data directory
cp your_document.pdf examples/sample_data/

# Or create a symlink
ln -s /path/to/your/pdfs/* examples/sample_data/
```

## File Descriptions

### sample_research_paper.pdf

- **Type**: Academic research paper
- **Content**: Machine learning research
- **Size**: ~500KB
- **Pages**: 10-15 pages
- **Use Case**: Testing academic document processing

### sample_technical_doc.pdf

- **Type**: Technical documentation
- **Content**: Software documentation
- **Size**: ~200KB
- **Pages**: 5-10 pages
- **Use Case**: Testing technical content extraction

### sample_report.pdf

- **Type**: Business report
- **Content**: Financial/business analysis
- **Size**: ~300KB
- **Pages**: 8-12 pages
- **Use Case**: Testing business document processing

## Creating Sample Data

If you need to create your own sample data:

```python
# Use the sample data generator
from examples.utils.sample_data_generator import create_sample_pdfs

create_sample_pdfs(
    output_dir="examples/sample_data",
    num_files=3,
    content_type="research"
)
```

## Data Privacy

- All sample data should be non-sensitive
- Do not commit personal or confidential documents
- Use publicly available or generated content only
- Follow your organization's data handling policies
