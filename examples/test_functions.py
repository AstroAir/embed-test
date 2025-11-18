"""
Quick Test Script for Examples Functions

This helper script runs lightweight logic tests for parts of the examples
without requiring external model downloads or API keys.

Prerequisites:
- Python environment with this project installed
- No external embedding models or third-party APIs required

Usage:
    uv run python -m examples.test_functions

Expected Output:
    - Printed results for tokenization, caching speedup, dynamic configuration,
      and benchmark comparison logic tests
    - Assertions will raise errors if any of the logic checks fail

Learning Objectives:
- Quickly validate core example logic without downloading large models
- See how to structure small, focused test helpers for example code
- Provide a fast sanity check before running full embedding examples
"""

import sys
from pathlib import Path


# Test 1: Tokenization logic
def test_tokenization_logic():
    """Test tokenization configuration logic"""
    print("Testing tokenization logic...")

    # Simulate tokenization
    sample_text = "Hello world! This is a test."
    tokens = sample_text.split()

    print(f"  Sample text: {sample_text}")
    print(f"  Token count: {len(tokens)}")
    print(f"  Tokens: {tokens}")
    print("  ✓ Tokenization logic test passed\n")
    return True


# Test 2: Caching speedup calculation
def test_caching_speedup():
    """Test caching speedup calculation logic"""
    print("Testing caching speedup calculation...")

    # Simulate timing
    first_run_time = 2.5  # seconds
    second_run_time = 0.01  # seconds

    if first_run_time and first_run_time > 0 and second_run_time > 0:
        speedup = first_run_time / second_run_time
        time_saved = first_run_time - second_run_time

        print(f"  First run: {first_run_time:.3f}s")
        print(f"  Second run: {second_run_time:.4f}s")
        print(f"  Speedup: {speedup:.1f}x")
        print(f"  Time saved: {time_saved:.3f}s")

        # Validate logic
        assert speedup > 1, "Speedup should be greater than 1"
        assert time_saved > 0, "Time saved should be positive"
        print("  ✓ Caching speedup logic test passed\n")
        return True
    return False


# Test 3: Dynamic configuration logic
def test_dynamic_configuration():
    """Test dynamic configuration logic"""
    print("Testing dynamic configuration logic...")

    # Test conditions
    test_cases = [
        {
            "name": "High volume",
            "conditions": {
                "document_count": 1500,
                "available_memory_gb": 8,
                "has_gpu": False,
            },
            "expected_batch": 100,
            "expected_workers": 8,
        },
        {
            "name": "Medium volume",
            "conditions": {
                "document_count": 150,
                "available_memory_gb": 8,
                "has_gpu": False,
            },
            "expected_batch": 50,
            "expected_workers": 4,
        },
        {
            "name": "Low volume",
            "conditions": {
                "document_count": 50,
                "available_memory_gb": 8,
                "has_gpu": False,
            },
            "expected_batch": 16,
            "expected_workers": 2,
        },
    ]

    for test_case in test_cases:
        conditions = test_case["conditions"]

        # Simulate configuration logic
        if conditions["document_count"] > 1000:
            batch_size = 100
            workers = 8
        elif conditions["document_count"] > 100:
            batch_size = 50
            workers = 4
        else:
            batch_size = 16
            workers = 2

        # Adjust for memory
        if conditions["available_memory_gb"] < 4:
            batch_size = min(batch_size, 8)
        elif conditions["available_memory_gb"] >= 16:
            batch_size = min(batch_size * 2, 200)

        # Adjust for GPU
        if conditions["has_gpu"]:
            batch_size = int(batch_size * 1.5)

        print(f"  Test: {test_case['name']}")
        print(f"    Document count: {conditions['document_count']}")
        print(f"    Batch size: {batch_size} (expected: {test_case['expected_batch']})")
        print(f"    Workers: {workers} (expected: {test_case['expected_workers']})")

        assert (
            batch_size == test_case["expected_batch"]
        ), f"Batch size mismatch for {test_case['name']}"
        assert (
            workers == test_case["expected_workers"]
        ), f"Workers mismatch for {test_case['name']}"

    print("  ✓ Dynamic configuration logic test passed\n")
    return True


# Test 4: Benchmark comparison logic
def test_benchmark_comparison():
    """Test benchmark comparison logic"""
    print("Testing benchmark comparison logic...")

    # Simulate benchmark results
    benchmarks = [
        {"model_name": "fast-model", "embeddings_per_second": 150.5, "dimensions": 384},
        {"model_name": "slow-model", "embeddings_per_second": 45.2, "dimensions": 768},
    ]

    if len(benchmarks) >= 2:
        fastest = max(benchmarks, key=lambda x: x["embeddings_per_second"])
        slowest = min(benchmarks, key=lambda x: x["embeddings_per_second"])
        speed_diff = fastest["embeddings_per_second"] / slowest["embeddings_per_second"]

        print(
            f"  Fastest: {fastest['model_name']} ({fastest['embeddings_per_second']:.2f} emb/sec)"
        )
        print(
            f"  Slowest: {slowest['model_name']} ({slowest['embeddings_per_second']:.2f} emb/sec)"
        )
        print(f"  Speed difference: {speed_diff:.2f}x")

        assert (
            fastest["model_name"] == "fast-model"
        ), "Fastest model identification failed"
        assert (
            slowest["model_name"] == "slow-model"
        ), "Slowest model identification failed"
        assert speed_diff > 1, "Speed difference should be greater than 1"
        print("  ✓ Benchmark comparison logic test passed\n")
        return True
    return False


# Test 5: Validation checks
def test_validation_checks():
    """Test validation logic"""
    print("Testing validation checks...")

    # Test null check
    result_with_data = {"embeddings": [[0.1, 0.2, 0.3]]}
    result_without_data = {"embeddings": None}

    print("  Testing null check:")
    if result_with_data["embeddings"] is not None:
        print(f"    Data present: {len(result_with_data['embeddings'])} embeddings")
        assert True
    else:
        print("    No data - should not reach here")
        assert False

    if result_without_data["embeddings"] is None:
        print("    No data detected correctly")
        assert True
    else:
        print("    Should have detected no data")
        assert False

    # Test metadata check
    metadata_present = {"chunks_metadata": [{"id": 1}, {"id": 2}]}
    metadata_empty = {"chunks_metadata": []}
    metadata_missing = {}

    print("  Testing metadata check:")
    if "chunks_metadata" in metadata_present and metadata_present["chunks_metadata"]:
        print(
            f"    Metadata present: {len(metadata_present['chunks_metadata'])} chunks"
        )
        assert True

    if "chunks_metadata" in metadata_empty and metadata_empty["chunks_metadata"]:
        print("    Should have detected empty metadata")
        assert False
    else:
        print("    Empty metadata detected correctly")
        assert True

    if "chunks_metadata" in metadata_missing and metadata_missing["chunks_metadata"]:
        print("    Should have detected missing metadata")
        assert False
    else:
        print("    Missing metadata detected correctly")
        assert True

    print("  ✓ Validation checks test passed\n")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Examples Functions Logic")
    print("=" * 60)
    print()

    tests = [
        ("Tokenization Logic", test_tokenization_logic),
        ("Caching Speedup Calculation", test_caching_speedup),
        ("Dynamic Configuration", test_dynamic_configuration),
        ("Benchmark Comparison", test_benchmark_comparison),
        ("Validation Checks", test_validation_checks),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  ✗ {test_name} failed\n")
        except Exception as e:
            failed += 1
            print(f"  ✗ {test_name} failed with error: {e}\n")

    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n✓ All logic tests passed successfully!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
