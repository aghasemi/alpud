#!/usr/bin/env python
"""
Example usage of the ALPUD Python implementation.
This script demonstrates how to use the informativeness computation functions.
"""

import numpy as np
from alpud import compute_informativeness_with_known_prior, compute_informativeness_with_unknown_prior


def main():
    print("="*60)
    print("ALPUD Python Implementation - Example Usage")
    print("="*60)
    
    # Example with 1D data
    print("\n1. Example with 1D data:")
    print("-" * 40)
    
    # Generate some sample data
    np.random.seed(42)
    positive_data = np.random.randn(20) + 1.0  # Centered around 1.0
    unlabeled_data = np.random.randn(50) * 1.5  # Wider distribution around 0
    query_samples = np.linspace(-2, 3, 10)  # Test points across the range
    
    print(f"Positive samples: {len(positive_data)}")
    print(f"Unlabeled samples: {len(unlabeled_data)}")
    print(f"Query samples: {query_samples}")
    
    # Compute informativeness with known prior
    P = 0.3  # Assume 30% of samples are positive
    info_known = compute_informativeness_with_known_prior(
        positive_data, unlabeled_data, P, query_samples
    )
    
    print(f"\nInformativeness (known prior P={P}):")
    for i, (q, inf) in enumerate(zip(query_samples, info_known)):
        print(f"  Query point {q:6.2f}: {inf:8.4f}")
    
    # Compute informativeness with unknown prior
    info_unknown = compute_informativeness_with_unknown_prior(
        positive_data, unlabeled_data, query_samples
    )
    
    print(f"\nInformativeness (unknown prior):")
    for i, (q, inf) in enumerate(zip(query_samples, info_unknown)):
        print(f"  Query point {q:6.2f}: {inf:8.4f}")
    
    # Example with 2D data
    print("\n\n2. Example with 2D data:")
    print("-" * 40)
    
    # Generate 2D sample data
    positive_data_2d = np.random.randn(2, 30) + np.array([[1], [1]])
    unlabeled_data_2d = np.random.randn(2, 80)
    query_samples_2d = np.array([
        [0, 0.5, 1.0, 1.5, 2.0],
        [0, 0.5, 1.0, 1.5, 2.0]
    ])
    
    print(f"Positive samples: {positive_data_2d.shape[1]}")
    print(f"Unlabeled samples: {unlabeled_data_2d.shape[1]}")
    print(f"Query samples shape: {query_samples_2d.shape}")
    
    # Compute informativeness
    info_known_2d = compute_informativeness_with_known_prior(
        positive_data_2d, unlabeled_data_2d, 0.3, query_samples_2d
    )
    
    info_unknown_2d = compute_informativeness_with_unknown_prior(
        positive_data_2d, unlabeled_data_2d, query_samples_2d
    )
    
    print(f"\nInformativeness (known prior):")
    for i in range(query_samples_2d.shape[1]):
        q = query_samples_2d[:, i]
        print(f"  Query point ({q[0]:5.1f}, {q[1]:5.1f}): {info_known_2d[i]:8.4f}")
    
    print(f"\nInformativeness (unknown prior):")
    for i in range(query_samples_2d.shape[1]):
        q = query_samples_2d[:, i]
        print(f"  Query point ({q[0]:5.1f}, {q[1]:5.1f}): {info_unknown_2d[i]:8.4f}")
    
    # Find most informative samples
    print("\n\n3. Finding most informative samples:")
    print("-" * 40)
    
    most_informative_idx = np.argmax(info_known)
    print(f"Most informative query point (known prior): "
          f"{query_samples[most_informative_idx]:.2f} "
          f"(informativeness: {info_known[most_informative_idx]:.4f})")
    
    most_informative_idx = np.argmax(np.abs(info_unknown))
    print(f"Most informative query point (unknown prior): "
          f"{query_samples[most_informative_idx]:.2f} "
          f"(informativeness: {info_unknown[most_informative_idx]:.4f})")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
