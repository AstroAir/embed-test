#!/usr/bin/env python
"""Script to fix Weaviate test mocks."""

import re

# Read the file
with open("tests/vector_db/test_weaviate_client.py", encoding="utf-8") as f:
    content = f.read()

# Replace all occurrences of weaviate.Client with weaviate.connect_to_custom
content = content.replace(
    '"pdf_vector_system.vector_db.weaviate_client.weaviate.Client"',
    '"pdf_vector_system.vector_db.weaviate_client.weaviate.connect_to_custom"',
)

# Also need to update mock_client_class to mock_connect
content = re.sub(r"as mock_client_class\)", "as mock_connect)", content)

content = re.sub(
    r"mock_client_class\.return_value = mock_client",
    "mock_connect.return_value = mock_client",
    content,
)

content = re.sub(
    r"mock_client_class\.assert_called_once\(\)",
    "mock_connect.assert_called_once()",
    content,
)

# Write back
with open("tests/vector_db/test_weaviate_client.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed all Weaviate test mocks")
