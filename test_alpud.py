#!/usr/bin/env python
"""
Simple test script to validate the ALPUD Python implementation.
"""

import numpy as np
from alpud import compute_informativeness_with_known_prior, compute_informativeness_with_unknown_prior


def test_known_prior():
    """Test the known prior function with 1D data."""
    print("Testing compute_informativeness_with_known_prior...")
    
    # Create some simple test data
    positive_data = np.array([1.0, 1.2, 1.1, 1.3, 1.15])
    unlabeled_data = np.array([0.5, 2.0, 1.5, 0.8, 2.5])
    query_samples = np.array([0.9, 1.0, 1.5, 2.0])
    P = 0.3
    
    # Compute informativeness
    informativeness = compute_informativeness_with_known_prior(
        positive_data, unlabeled_data, P, query_samples
    )
    
    print(f"  Positive data: {positive_data}")
    print(f"  Unlabeled data: {unlabeled_data}")
    print(f"  Query samples: {query_samples}")
    print(f"  Prior P: {P}")
    print(f"  Informativeness: {informativeness}")
    
    # Basic sanity checks
    assert informativeness.shape == query_samples.shape, "Output shape mismatch"
    assert np.all(informativeness >= 0), "Informativeness should be non-negative"
    assert np.all(informativeness <= 1), "Informativeness should be <= 1"
    
    print("  ✓ Test passed!")
    return informativeness


def test_unknown_prior():
    """Test the unknown prior function with 1D data."""
    print("\nTesting compute_informativeness_with_unknown_prior...")
    
    # Create some simple test data
    positive_data = np.array([1.0, 1.2, 1.1, 1.3, 1.15])
    unlabeled_data = np.array([0.5, 2.0, 1.5, 0.8, 2.5])
    query_samples = np.array([0.9, 1.0, 1.5, 2.0])
    
    # Compute informativeness
    informativeness = compute_informativeness_with_unknown_prior(
        positive_data, unlabeled_data, query_samples
    )
    
    print(f"  Positive data: {positive_data}")
    print(f"  Unlabeled data: {unlabeled_data}")
    print(f"  Query samples: {query_samples}")
    print(f"  Informativeness: {informativeness}")
    
    # Basic sanity checks
    assert informativeness.shape == query_samples.shape, "Output shape mismatch"
    # Note: informativeness can be outside [-1, 1] range depending on the data
    
    print("  ✓ Test passed!")
    return informativeness


def test_2d_data():
    """Test with 2D data."""
    print("\nTesting with 2D data...")
    
    # Create 2D test data
    np.random.seed(42)
    positive_data = np.random.randn(2, 10) + np.array([[1], [1]])
    unlabeled_data = np.random.randn(2, 15)
    query_samples = np.random.randn(2, 5)
    P = 0.4
    
    # Test known prior
    informativeness_known = compute_informativeness_with_known_prior(
        positive_data, unlabeled_data, P, query_samples
    )
    
    # Test unknown prior
    informativeness_unknown = compute_informativeness_with_unknown_prior(
        positive_data, unlabeled_data, query_samples
    )
    
    print(f"  Positive data shape: {positive_data.shape}")
    print(f"  Unlabeled data shape: {unlabeled_data.shape}")
    print(f"  Query samples shape: {query_samples.shape}")
    print(f"  Informativeness (known prior) shape: {informativeness_known.shape}")
    print(f"  Informativeness (unknown prior) shape: {informativeness_unknown.shape}")
    
    assert informativeness_known.shape[0] == query_samples.shape[1], "Output shape mismatch"
    assert informativeness_unknown.shape[0] == query_samples.shape[1], "Output shape mismatch"
    
    print("  ✓ Test passed!")


if __name__ == "__main__":
    print("="*60)
    print("ALPUD Python Implementation Tests")
    print("="*60)
    
    try:
        test_known_prior()
        test_unknown_prior()
        test_2d_data()
        
        print("\n" + "="*60)
        print("All tests passed successfully! ✓")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
