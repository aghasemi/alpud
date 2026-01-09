"""
ALPUD: Active Learning from Positive and Unlabeled Data

This module contains a Python implementation of the ALPUD algorithm 
(Active Learning from Positive and Unlabeled Data), introduced and presented 
in the following paper:

- Ghasemi, Alireza, Hamid R. Rabiee, Mohsen Fadaee, Mohammad T. Manzuri, 
  and Mohammad H. Rohban. "Active learning from positive and unlabeled data." 
  In Data Mining Workshops (ICDMW), 2011 IEEE 11th International Conference on, 
  pp. 244-250. IEEE, 2011.

The paper can be found on IEEE Xplore and downloaded from:
https://arxiv.org/pdf/1602.07495.pdf

Copyright (c) 2016 Alireza Ghasemi
Published under MIT license.
"""

import numpy as np
from scipy.stats import gaussian_kde


def compute_informativeness_with_known_prior(positive_data, unlabeled_data, P, query_samples):
    """
    Computes the amount of informativeness of a data sample in a semi-supervised 
    one-class learning scenario where only positive and unlabeled data are available. 
    Please see the ALPUD paper (cited in the README) for more information. 
    
    This function requires that a prior knowledge of the proportion of positive 
    examples in the sample space (P) be available.
    
    Parameters
    ----------
    positive_data : array-like
        Data samples known to belong to the target class
    unlabeled_data : array-like
        Data samples we don't know whether they belong to target class or not
    P : float
        The prior probability of a sample belonging to the target class
    query_samples : array-like
        Data samples for which we want to compute the informativeness
    
    Returns
    -------
    informativeness : ndarray
        The informativeness score for each query sample
    """
    # Convert inputs to numpy arrays
    positive_data = np.asarray(positive_data)
    unlabeled_data = np.asarray(unlabeled_data)
    query_samples = np.asarray(query_samples)
    
    # Ensure data is properly shaped for KDE
    if positive_data.ndim == 1:
        positive_data = positive_data.reshape(1, -1)
    if unlabeled_data.ndim == 1:
        unlabeled_data = unlabeled_data.reshape(1, -1)
    if query_samples.ndim == 1:
        query_samples = query_samples.reshape(1, -1)
    
    # p(x|+)
    positive_kde = gaussian_kde(positive_data)
    positive_likelihood = positive_kde.evaluate(query_samples)
    
    # p(x)
    combined_data = np.concatenate([positive_data, unlabeled_data], axis=1)
    total_kde = gaussian_kde(combined_data)
    total_likelihood = total_kde.evaluate(query_samples)
    
    # a_x = p(x|+)/p(x)
    # Add small epsilon to avoid division by zero
    epsilon = np.finfo(float).eps
    a_of_x = positive_likelihood / (total_likelihood + epsilon)
    
    # Now we can compute the informativeness based on Eq. 8 in the paper
    informativeness = np.abs(1 - 2 * P * a_of_x)
    
    return informativeness


def compute_informativeness_with_unknown_prior(positive_data, unlabeled_data, query_samples):
    """
    Computes the amount of informativeness of a data sample in a semi-supervised 
    one-class learning scenario where only positive and unlabeled data are available. 
    Please see the ALPUD paper (cited in the README) for more information.
    
    This function does not require that a prior knowledge of the proportion of 
    positive examples in the sample space (P) be available. Instead it computes 
    the expected value over all possible Ps using a uniform prior distribution.
    
    Parameters
    ----------
    positive_data : array-like
        Data samples known to belong to the target class
    unlabeled_data : array-like
        Data samples we don't know whether they belong to target class or not
    query_samples : array-like
        Data samples for which we want to compute the informativeness
    
    Returns
    -------
    informativeness : ndarray
        The informativeness score for each query sample
    """
    # Convert inputs to numpy arrays
    positive_data = np.asarray(positive_data)
    unlabeled_data = np.asarray(unlabeled_data)
    query_samples = np.asarray(query_samples)
    
    # Ensure data is properly shaped for KDE
    if positive_data.ndim == 1:
        positive_data = positive_data.reshape(1, -1)
    if unlabeled_data.ndim == 1:
        unlabeled_data = unlabeled_data.reshape(1, -1)
    if query_samples.ndim == 1:
        query_samples = query_samples.reshape(1, -1)
    
    # p(x|+)
    positive_kde = gaussian_kde(positive_data)
    positive_likelihood = positive_kde.evaluate(query_samples)
    
    # p(x)
    combined_data = np.concatenate([positive_data, unlabeled_data], axis=1)
    total_kde = gaussian_kde(combined_data)
    total_likelihood = total_kde.evaluate(query_samples)
    
    # a_x = p(x|+)/p(x)
    # Add small epsilon to avoid division by zero
    epsilon = np.finfo(float).eps
    a_of_x = positive_likelihood / (total_likelihood + epsilon)
    
    # Now we can compute the informativeness based on Eq. 8 in the paper
    # Note: MATLAB's sgn is equivalent to numpy's sign
    informativeness = (1 - a_of_x) * np.sign(0.5 - a_of_x)
    
    return informativeness
