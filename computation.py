"""
W Boson Mass and Width Analysis
Step 1: Data Loading and Validation
Step 2: Calculate Muon Energy
Step 3: Reconstruct Neutrino 4-Momentum
Step 4: Calculate Transverse Quantities
Step 5: Compute Transverse Mass
Step 6: Create Histogram and Fit Distribution
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
import sys

# Physical constants
MUON_MASS = 0.10566  # GeV/c^2 (PDG value)
CM_ENERGY = 7000.0   # GeV (sqrt(s) for pp collision)

# Configuration: Neutrino reconstruction method
# Change this value to switch between methods:
#   'conservation' - Calculate from momentum/energy conservation laws
#   'csv'          - Use generator-level neutrino from CSV file
NEUTRINO_METHOD = 'csv'  # <<< CHANGE THIS TO SWITCH METHODS

# Configuration: Fitting method
# Change this value to switch between fitting functions:
#   'breit_wigner'     - Simple Breit-Wigner (full range 40-120 GeV)
#   'crystal_ball'     - Crystal Ball function (asymmetric peak + power-law tail)
#   'restricted_range' - Breit-Wigner with restricted range (60-95 GeV)
FIT_METHOD = 'crystal_ball'  # <<< CHANGE THIS TO SWITCH FIT METHODS


def load_and_validate_data(csv_path, load_neutrino=False):
    """
    Load W boson analysis data from CSV file and perform validation checks.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the CSV file containing the analysis data
    load_neutrino : bool, optional
        If True, load generator-level neutrino momentum from CSV
        If False, neutrino will be reconstructed later (default: False)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'p_mu': numpy array of shape (N, 3) with muon momentum components [px, py, pz]
        - 'p_other': numpy array of shape (N, 3) with other particles momentum [px, py, pz]
        - 'e_other': numpy array of shape (N,) with other particles energy
        - 'p_nu': numpy array of shape (N, 3) with neutrino momentum [only if load_neutrino=True]
        - 'n_events': int, number of events
        
    Raises:
    -------
    FileNotFoundError
        If CSV file doesn't exist
    ValueError
        If data validation fails
    """
    # Load the CSV file
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} events")
    print(f"Columns in file: {list(df.columns)}")
    
    # Extract muon momentum components
    p_mu_x = np.asarray(df['pμ_px'].values)
    p_mu_y = np.asarray(df['pμ_py'].values)
    p_mu_z = np.asarray(df['pμ_pz'].values)
    p_mu = np.column_stack([p_mu_x, p_mu_y, p_mu_z])
    
    # Extract other particles' 4-momentum
    p_other_x = np.asarray(df['p_other_px'].values)
    p_other_y = np.asarray(df['p_other_py'].values)
    p_other_z = np.asarray(df['p_other_pz'].values)
    e_other = np.asarray(df['p_other_e'].values)
    p_other = np.column_stack([p_other_x, p_other_y, p_other_z])
    
    # Optionally extract neutrino momentum (generator-level values)
    p_nu = None
    if load_neutrino:
        p_nu_x = np.asarray(df['pν_px'].values)
        p_nu_y = np.asarray(df['pν_py'].values)
        p_nu_z = np.asarray(df['pν_pz'].values)
        p_nu = np.column_stack([p_nu_x, p_nu_y, p_nu_z])
        print("Using generator-level neutrino momentum from CSV")
    
    print("\n=== Data Validation ===")
    
    # Check for NaN values
    if np.any(np.isnan(p_mu)):
        raise ValueError("Found NaN values in muon momentum data")
    if np.any(np.isnan(p_other)):
        raise ValueError("Found NaN values in other particles momentum data")
    if np.any(np.isnan(e_other)):
        raise ValueError("Found NaN values in other particles energy data")
    print("✓ No NaN values found")
    
    # Check for infinite values
    if np.any(np.isinf(p_mu)):
        raise ValueError("Found infinite values in muon momentum data")
    if np.any(np.isinf(p_other)):
        raise ValueError("Found infinite values in other particles momentum data")
    if np.any(np.isinf(e_other)):
        raise ValueError("Found infinite values in other particles energy data")
    print("✓ No infinite values found")
    
    # Physical consistency checks
    p_mu_mag = np.sqrt(np.sum(p_mu**2, axis=1))
    p_other_mag = np.sqrt(np.sum(p_other**2, axis=1))
    
    # Check that energy is positive and greater than momentum magnitude (E^2 = p^2 + m^2)
    if np.any(e_other < 0):
        n_negative = np.sum(e_other < 0)
        raise ValueError(f"Found {n_negative} events with negative energy for other particles")
    print("✓ All energies are positive")
    
    # For other particles (massive system), energy should be >= momentum
    energy_momentum_ratio = e_other / (p_other_mag + 1e-10)  # Add small value to avoid division by zero
    if np.any(energy_momentum_ratio < 0.999):  # Allow small numerical errors
        n_violations = np.sum(energy_momentum_ratio < 0.999)
        print(f"⚠ Warning: {n_violations} events have E < |p| for other particles (possible numerical precision issue)")
        print(f"  Min E/|p| ratio: {np.min(energy_momentum_ratio):.6f}")
    else:
        print("✓ Energy-momentum relation satisfied for other particles")
    
    # Summary statistics
    print("\n=== Data Summary ===")
    print(f"Muon momentum magnitude: {np.mean(p_mu_mag):.2f} ± {np.std(p_mu_mag):.2f} GeV (mean ± std)")
    print(f"  Range: [{np.min(p_mu_mag):.2f}, {np.max(p_mu_mag):.2f}] GeV")
    print(f"Other particles momentum: {np.mean(p_other_mag):.2f} ± {np.std(p_other_mag):.2f} GeV (mean ± std)")
    print(f"  Range: [{np.min(p_other_mag):.2f}, {np.max(p_other_mag):.2f}] GeV")
    print(f"Other particles energy: {np.mean(e_other):.2f} ± {np.std(e_other):.2f} GeV (mean ± std)")
    print(f"  Range: [{np.min(e_other):.2f}, {np.max(e_other):.2f}] GeV")
    
    # Return structured data
    result = {
        'p_mu': p_mu,
        'p_other': p_other,
        'e_other': e_other,
        'n_events': len(df)
    }
    
    if p_nu is not None:
        result['p_nu'] = p_nu
    
    return result


def calculate_muon_energy(p_mu):
    """
    Calculate muon energy from 3-momentum using relativistic energy-momentum relation.
    
    E_μ = sqrt(|p_μ|^2 + m_μ^2)
    
    Parameters:
    -----------
    p_mu : numpy array of shape (N, 3)
        Muon 3-momentum components [px, py, pz] in GeV
        
    Returns:
    --------
    numpy array of shape (N,)
        Muon energy in GeV
    """
    # Calculate momentum magnitude: |p| = sqrt(px^2 + py^2 + pz^2)
    p_mu_squared = np.sum(p_mu**2, axis=1)
    
    # Calculate energy: E = sqrt(p^2 + m^2)
    e_mu = np.sqrt(p_mu_squared + MUON_MASS**2)
    
    print("\n=== Muon Energy Calculation ===")
    print(f"Muon mass used: {MUON_MASS} GeV/c²")
    print(f"Muon energy: {np.mean(e_mu):.2f} ± {np.std(e_mu):.2f} GeV (mean ± std)")
    print(f"  Range: [{np.min(e_mu):.2f}, {np.max(e_mu):.2f}] GeV")
    
    # Check if muon is ultra-relativistic (E >> m)
    p_mu_mag = np.sqrt(p_mu_squared)
    relativistic_fraction = np.sum(p_mu_mag > 10 * MUON_MASS) / len(p_mu_mag)
    print(f"Ultra-relativistic fraction (|p| > 10m_μ): {relativistic_fraction*100:.1f}%")
    
    return e_mu


def reconstruct_neutrino(p_mu, e_mu, p_other, e_other):
    """
    Reconstruct neutrino 4-momentum using momentum and energy conservation.
    
    Conservation laws:
    - Momentum: p_ν = p_initial - p_μ - p_other = -p_μ - p_other (since p_initial = 0)
    - Energy: E_ν = E_initial - E_μ - E_other = 7000 GeV - E_μ - E_other
    
    Parameters:
    -----------
    p_mu : numpy array of shape (N, 3)
        Muon 3-momentum [px, py, pz] in GeV
    e_mu : numpy array of shape (N,)
        Muon energy in GeV
    p_other : numpy array of shape (N, 3)
        Other particles 3-momentum [px, py, pz] in GeV
    e_other : numpy array of shape (N,)
        Other particles energy in GeV
        
    Returns:
    --------
    tuple (p_nu, e_nu)
        p_nu : numpy array of shape (N, 3) - neutrino 3-momentum in GeV
        e_nu : numpy array of shape (N,) - neutrino energy in GeV
    """
    print("\n=== Neutrino Reconstruction ===")
    print(f"Using momentum and energy conservation")
    print(f"Initial state: pp collision at √s = {CM_ENERGY} GeV")
    
    # Momentum conservation: p_ν = -p_μ - p_other
    # (Initial momentum is zero in lab frame)
    p_nu = -p_mu - p_other
    
    # Energy conservation: E_ν = E_initial - E_μ - E_other
    e_nu = CM_ENERGY - e_mu - e_other
    
    # Calculate momentum magnitude for validation
    p_nu_mag = np.sqrt(np.sum(p_nu**2, axis=1))
    
    # Validation checks
    n_negative = np.sum(e_nu <= 0)
    n_positive = np.sum(e_nu > 0)
    
    if n_negative > 0:
        print(f"\n⚠ Warning: {n_negative}/{len(e_nu)} events with E_ν ≤ 0 (unphysical)")
        print(f"  Min E_ν: {np.min(e_nu):.2f} GeV")
        print(f"  This indicates conservation method failure for this dataset")
    else:
        print(f"\n✓ All {n_positive} events have positive neutrino energy")
    
    # Check momentum and energy conservation
    total_momentum = p_mu + p_nu + p_other
    momentum_mag = np.sqrt(np.sum(total_momentum**2, axis=1))
    total_energy = e_mu + e_nu + e_other
    energy_diff = np.abs(total_energy - CM_ENERGY)
    
    print(f"Momentum conservation: max |p_total| = {np.max(momentum_mag):.2e} GeV")
    print(f"Energy conservation: max deviation = {np.max(energy_diff):.2e} GeV")
    
    print("\n--- Neutrino Statistics ---")
    print(f"Neutrino energy: {np.mean(e_nu):.2f} ± {np.std(e_nu):.2f} GeV (mean ± std)")
    print(f"  Range: [{np.min(e_nu):.2f}, {np.max(e_nu):.2f}] GeV")
    print(f"Neutrino |p|: {np.mean(p_nu_mag):.2f} ± {np.std(p_nu_mag):.2f} GeV (mean ± std)")
    print(f"  Range: [{np.min(p_nu_mag):.2f}, {np.max(p_nu_mag):.2f}] GeV")
    
    # Transverse momentum
    p_t_nu = np.sqrt(p_nu[:, 0]**2 + p_nu[:, 1]**2)
    print(f"Neutrino pT: {np.mean(p_t_nu):.2f} ± {np.std(p_t_nu):.2f} GeV (mean ± std)")
    print(f"  Range: [{np.min(p_t_nu):.2f}, {np.max(p_t_nu):.2f}] GeV")
    
    return p_nu, e_nu


def calculate_neutrino_energy_from_momentum(p_nu):
    """
    Calculate neutrino energy from 3-momentum, assuming massless neutrino.
    Used when neutrino momentum is loaded from CSV.
    
    E_ν = |p_ν| (for massless particle)
    
    Parameters:
    -----------
    p_nu : numpy array of shape (N, 3)
        Neutrino 3-momentum components [px, py, pz] in GeV
        
    Returns:
    --------
    numpy array of shape (N,)
        Neutrino energy in GeV
    """
    print("\n=== Neutrino Energy Calculation (from CSV momentum) ===")
    print(f"Assuming massless neutrino: E_ν = |p_ν|")
    
    # Calculate momentum magnitude
    p_nu_mag = np.sqrt(np.sum(p_nu**2, axis=1))
    e_nu = p_nu_mag
    
    print(f"Neutrino energy: {np.mean(e_nu):.2f} ± {np.std(e_nu):.2f} GeV (mean ± std)")
    print(f"  Range: [{np.min(e_nu):.2f}, {np.max(e_nu):.2f}] GeV")
    
    return e_nu


def calculate_transverse_quantities(p_mu, p_nu):
    """
    Calculate transverse momenta and azimuthal angle difference.
    
    Parameters:
    -----------
    p_mu : numpy array of shape (N, 3)
        Muon 3-momentum [px, py, pz] in GeV
    p_nu : numpy array of shape (N, 3)
        Neutrino 3-momentum [px, py, pz] in GeV
        
    Returns:
    --------
    tuple (p_t_mu, p_t_nu, delta_phi)
        p_t_mu : numpy array of shape (N,) - muon transverse momentum in GeV
        p_t_nu : numpy array of shape (N,) - neutrino transverse momentum in GeV
        delta_phi : numpy array of shape (N,) - azimuthal angle difference in radians [-π, π]
    """
    print("\n=== Transverse Quantities Calculation ===")
    
    # Calculate transverse momenta
    p_t_mu = np.sqrt(p_mu[:, 0]**2 + p_mu[:, 1]**2)
    p_t_nu = np.sqrt(p_nu[:, 0]**2 + p_nu[:, 1]**2)
    
    # Calculate azimuthal angles
    phi_mu = np.arctan2(p_mu[:, 1], p_mu[:, 0])
    phi_nu = np.arctan2(p_nu[:, 1], p_nu[:, 0])
    
    # Calculate angle difference, ensuring it's in [-π, π]
    delta_phi = phi_mu - phi_nu
    # Wrap to [-π, π]
    delta_phi = np.arctan2(np.sin(delta_phi), np.cos(delta_phi))
    
    # Statistics
    print(f"Muon pT: {np.mean(p_t_mu):.2f} ± {np.std(p_t_mu):.2f} GeV (mean ± std)")
    print(f"  Range: [{np.min(p_t_mu):.2f}, {np.max(p_t_mu):.2f}] GeV")
    print(f"Neutrino pT: {np.mean(p_t_nu):.2f} ± {np.std(p_t_nu):.2f} GeV (mean ± std)")
    print(f"  Range: [{np.min(p_t_nu):.2f}, {np.max(p_t_nu):.2f}] GeV")
    print(f"Δφ: {np.mean(delta_phi):.3f} ± {np.std(delta_phi):.3f} rad (mean ± std)")
    print(f"  Range: [{np.min(delta_phi):.3f}, {np.max(delta_phi):.3f}] rad")
    
    return p_t_mu, p_t_nu, delta_phi


def calculate_transverse_mass(p_t_mu, p_t_nu, delta_phi):
    """
    Calculate transverse mass of the W boson candidate.
    
    The transverse mass is defined as:
    m_T = sqrt(2 * p_T^μ * p_T^ν * (1 - cos(Δφ)))
    
    This is the key observable for measuring the W boson mass, as the full
    invariant mass cannot be reconstructed due to the unmeasured neutrino
    longitudinal momentum.
    
    Parameters:
    -----------
    p_t_mu : numpy array of shape (N,)
        Muon transverse momentum in GeV
    p_t_nu : numpy array of shape (N,)
        Neutrino transverse momentum in GeV
    delta_phi : numpy array of shape (N,)
        Azimuthal angle difference in radians
        
    Returns:
    --------
    numpy array of shape (N,)
        Transverse mass in GeV
    """
    print("\n=== Transverse Mass Calculation ===")
    
    # Calculate transverse mass
    m_t = np.sqrt(2 * p_t_mu * p_t_nu * (1 - np.cos(delta_phi)))
    
    # Statistics
    print(f"Transverse mass m_T: {np.mean(m_t):.2f} ± {np.std(m_t):.2f} GeV (mean ± std)")
    print(f"  Range: [{np.min(m_t):.2f}, {np.max(m_t):.2f}] GeV")
    print(f"  Median: {np.median(m_t):.2f} GeV")
    
    # Check for reasonable values
    n_near_mw = np.sum((m_t > 70) & (m_t < 90))
    print(f"\nEvents with 70 < m_T < 90 GeV: {n_near_mw} ({n_near_mw/len(m_t)*100:.1f}%)")
    print(f"(Expected W mass region around 80 GeV)")
    
    return m_t


def create_histogram(m_t, bins=100, range_min=0, range_max=150):
    """
    Create histogram of transverse mass distribution.
    
    Parameters:
    -----------
    m_t : numpy array
        Transverse mass values in GeV
    bins : int, optional
        Number of bins (default: 100)
    range_min : float, optional
        Minimum value of histogram range in GeV (default: 0)
    range_max : float, optional
        Maximum value of histogram range in GeV (default: 150)
        
    Returns:
    --------
    tuple (bin_centers, counts, bin_edges, errors)
        bin_centers : numpy array - center of each bin in GeV
        counts : numpy array - number of events in each bin
        bin_edges : numpy array - edges of bins
        errors : numpy array - statistical uncertainties (sqrt(N))
    """
    print("\n=== Histogram Creation ===")
    
    # Create histogram
    counts, bin_edges = np.histogram(m_t, bins=bins, range=(range_min, range_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Calculate statistical uncertainties (Poisson: σ = sqrt(N))
    errors = np.sqrt(counts)
    # For bins with zero counts, set error to 1 for plotting purposes
    errors[errors == 0] = 1
    
    print(f"Histogram range: [{range_min}, {range_max}] GeV")
    print(f"Number of bins: {bins}")
    print(f"Bin width: {bin_width:.2f} GeV")
    print(f"Total events in histogram: {np.sum(counts)} / {len(m_t)}")
    print(f"Events outside range: {len(m_t) - np.sum(counts)}")
    
    # Find peak region
    max_bin_idx = np.argmax(counts)
    peak_position = bin_centers[max_bin_idx]
    print(f"\nPeak bin position: {peak_position:.2f} GeV")
    print(f"Peak bin count: {counts[max_bin_idx]}")
    
    return bin_centers, counts, bin_edges, errors


def find_histogram_peak(bin_centers, counts, search_range=(70, 90)):
    """
    Find peak position in histogram using simple maximum finding.
    Provides model-independent mass estimate.
    
    Parameters:
    -----------
    bin_centers : array
        Bin center positions
    counts : array
        Event counts per bin
    search_range : tuple
        (min, max) range in GeV to search for peak
        
    Returns:
    --------
    tuple (peak_position, peak_height, peak_uncertainty)
    """
    mask = (bin_centers >= search_range[0]) & (bin_centers <= search_range[1])
    if np.sum(mask) == 0:
        return None, None, None
    
    peak_idx = np.argmax(counts[mask])
    peak_position = bin_centers[mask][peak_idx]
    peak_height = counts[mask][peak_idx]
    
    # Estimate uncertainty as bin width (conservative)
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0
    peak_uncertainty = bin_width / 2.0
    
    return peak_position, peak_height, peak_uncertainty


def breit_wigner(m, m_w, gamma_w, amplitude):
    """
    Relativistic Breit-Wigner function for fitting.
    
    BW(m) = amplitude * (m * Γ_W) / ((m^2 - m_W^2)^2 + m_W^2 * Γ_W^2)
    
    Parameters:
    -----------
    m : float or array
        Transverse mass in GeV
    m_w : float
        W boson mass in GeV
    gamma_w : float
        W boson decay width in GeV
    amplitude : float
        Overall normalization
        
    Returns:
    --------
    float or array
        Function value
    """
    numerator = amplitude * m * gamma_w
    denominator = (m**2 - m_w**2)**2 + m_w**2 * gamma_w**2
    return numerator / denominator


def crystal_ball(m, mean, sigma, alpha, n, amplitude):
    """
    Crystal Ball function for fitting asymmetric peaks.
    
    Combines Gaussian core with power-law tail for low mass side.
    Standard function used in particle physics for mass peaks.
    
    Parameters:
    -----------
    m : float or array
        Transverse mass in GeV
    mean : float
        Peak position (approximately m_W) in GeV
    sigma : float
        Width parameter (core resolution) in GeV
    alpha : float
        Transition point between Gaussian and power-law (>0)
    n : float
        Power-law exponent for tail (>1)
    amplitude : float
        Overall normalization
        
    Returns:
    --------
    float or array
        Function value
    """
    t = (m - mean) / sigma
    
    # Avoid division by zero
    abs_alpha = np.abs(alpha)
    
    # Constants for the power-law tail
    A = (n / abs_alpha)**n * np.exp(-0.5 * alpha**2)
    B = n / abs_alpha - abs_alpha
    
    # Piecewise function
    result = np.zeros_like(m, dtype=float)
    
    # Gaussian core (right side and small left deviations)
    mask_core = t > -abs_alpha
    result[mask_core] = amplitude * np.exp(-0.5 * t[mask_core]**2)
    
    # Power-law tail (large left deviations)
    mask_tail = t <= -abs_alpha
    result[mask_tail] = amplitude * A * (B - t[mask_tail])**(-n)
    
    return result


def fit_transverse_mass(bin_centers, counts, errors, fit_method='breit_wigner', initial_guess=None):
    """
    Fit transverse mass distribution with selected method.
    
    Parameters:
    -----------
    bin_centers : numpy array
        Center of histogram bins in GeV
    counts : numpy array
        Number of events in each bin
    errors : numpy array
        Statistical uncertainties for each bin
    fit_method : str
        Fitting method: 'breit_wigner', 'crystal_ball', or 'restricted_range'
    initial_guess : tuple, optional
        Initial parameter guess (depends on method)
        
    Returns:
    --------
    tuple (popt, pcov, fit_info)
        popt : array - optimal parameters
        pcov : array - covariance matrix
        fit_info : dict - additional fit information
    """
    print("\n=== Fitting Transverse Mass Distribution ===")
    print(f"Fit method: {fit_method}")
    
    # Select fitting range based on method
    if fit_method == 'restricted_range':
        range_min, range_max = 60, 95
        print(f"Using restricted fit range: [{range_min}, {range_max}] GeV")
    else:
        range_min, range_max = 40, 120
        print(f"Using standard fit range: [{range_min}, {range_max}] GeV")
    
    # Only fit bins with non-zero counts and in selected range
    mask = (counts > 0) & (bin_centers > range_min) & (bin_centers < range_max)
    
    if np.sum(mask) < 10:
        print("⚠ Warning: Too few bins with data for reliable fit")
        mask = counts > 0
    
    x_fit = bin_centers[mask]
    y_fit = counts[mask]
    sigma_fit = errors[mask]
    
    print(f"\nFitting {np.sum(mask)} bins with data")
    print(f"Fit range: [{np.min(x_fit):.1f}, {np.max(x_fit):.1f}] GeV")
    
    # Prepare initial guess and fit function based on method
    if fit_method == 'crystal_ball':
        if initial_guess is None:
            # Use histogram peak as initial guess for mean
            peak_pos, peak_height, _ = find_histogram_peak(bin_centers, counts)
            mean_init = peak_pos if peak_pos is not None else 80.0
            sigma_init = 8.0  # Narrower initial width
            alpha_init = 1.0  # Tail transition parameter  
            n_init = 1.5  # Power-law exponent
            amplitude_init = peak_height if peak_height is not None else np.max(counts)
            initial_guess = (mean_init, sigma_init, alpha_init, n_init, amplitude_init)
        
        print(f"Initial guess: mean={initial_guess[0]:.1f} GeV, σ={initial_guess[1]:.1f} GeV")
        
        fit_func = crystal_ball
        param_names = ['Mean (≈m_W)', 'Sigma', 'Alpha', 'n', 'Amplitude']
        
    else:  # breit_wigner or restricted_range
        if initial_guess is None:
            m_w_init = 80.0
            gamma_w_init = 2.0
            amplitude_init = np.max(counts) * 1000
            initial_guess = (m_w_init, gamma_w_init, amplitude_init)
        
        print(f"Initial guess: m_W={initial_guess[0]:.1f} GeV, Γ_W={initial_guess[1]:.1f} GeV")
        
        fit_func = breit_wigner
        param_names = ['m_W', 'Γ_W', 'Amplitude']
    
    try:
        # Perform fit
        popt, pcov = curve_fit(
            fit_func,
            x_fit,
            y_fit,
            p0=initial_guess,
            sigma=sigma_fit,
            absolute_sigma=True,
            maxfev=10000
        )
        
        # Uncertainties from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        
        print(f"\n{'='*50}")
        print("FIT RESULTS")
        print(f"{'='*50}")
        
        # Print all parameters
        for i, (name, val, err) in enumerate(zip(param_names, popt, perr)):
            if 'Amplitude' not in name:
                print(f"{name}: {val:.3f} ± {err:.3f} GeV" if 'GeV' in name or i < 2 else f"{name}: {val:.3f} ± {err:.3f}")
            else:
                print(f"{name}: {val:.2e} ± {err:.2e}")
        
        # Extract m_W and Γ_W based on method
        if fit_method == 'crystal_ball':
            m_w_fit = popt[0]  # Mean is approximately m_W
            m_w_err = perr[0]
            gamma_w_fit = 2.35 * popt[1]  # FWHM ≈ 2.35 * sigma for Gaussian core
            gamma_w_err = 2.35 * perr[1]
            print(f"\nDerived W parameters:")
            print(f"  m_W (from mean): {m_w_fit:.3f} ± {m_w_err:.3f} GeV")
            print(f"  Γ_W (from FWHM): {gamma_w_fit:.3f} ± {gamma_w_err:.3f} GeV")
        else:
            m_w_fit = popt[0]
            gamma_w_fit = popt[1]
            m_w_err = perr[0]
            gamma_w_err = perr[1]
        
        # Calculate chi-squared
        y_pred = fit_func(x_fit, *popt)
        chi2 = np.sum(((y_fit - y_pred) / sigma_fit)**2)
        ndf = len(x_fit) - len(popt)  # degrees of freedom
        chi2_ndf = chi2 / ndf if ndf > 0 else np.nan
        
        print(f"\nGoodness of fit:")
        print(f"  χ² = {chi2:.2f}")
        print(f"  NDF = {ndf}")
        print(f"  χ²/NDF = {chi2_ndf:.3f}")
        
        # Interpret fit quality (adjusted for transverse mass physics)
        if chi2_ndf < 1.5:
            print("  Status: Good fit")
        elif chi2_ndf < 3.5:
            print("  Status: Acceptable fit (transverse mass has kinematic effects)")
        elif chi2_ndf < 10:
            print("  Status: Poor fit (model may be inadequate)")
        else:
            print("  Status: Very poor fit (wrong model for transverse mass)")
        
        # Validate fitted mass with histogram peak
        peak_pos, _, peak_unc = find_histogram_peak(bin_centers, counts)
        if peak_pos is not None:
            print(f"\nValidation check:")
            print(f"  Histogram peak: {peak_pos:.2f} ± {peak_unc:.2f} GeV (model-independent)")
            print(f"  Fitted mass: {m_w_fit:.2f} ± {m_w_err:.2f} GeV (from {fit_method})")
            diff = abs(m_w_fit - peak_pos)
            if diff < 2.0:
                print(f"  Agreement: Excellent (Δ = {diff:.2f} GeV)")
            elif diff < 5.0:
                print(f"  Agreement: Reasonable (Δ = {diff:.2f} GeV)")
            else:
                print(f"  ⚠ Warning: Large deviation (Δ = {diff:.2f} GeV)")
        
        # Compare with known values
        print(f"\nComparison with PDG values:")
        print(f"  PDG m_W = 80.379 ± 0.012 GeV")
        print(f"  PDG Γ_W = 2.085 ± 0.042 GeV")
        print(f"  Difference in m_W: {abs(m_w_fit - 80.379):.3f} GeV ({abs(m_w_fit - 80.379)/m_w_err:.1f}σ)")
        print(f"  Difference in Γ_W: {abs(gamma_w_fit - 2.085):.3f} GeV ({abs(gamma_w_fit - 2.085)/gamma_w_err:.1f}σ)")
        
        # Important physics note about width
        if gamma_w_fit > 4.0:  # More than 2x PDG value
            print(f"\n  ⚠ Note: Transverse mass width ≠ W natural width")
            print(f"          Width includes kinematic smearing from W pT spectrum")
            print(f"          Real experiments use template fitting for accurate ΓW")
        
        fit_info = {
            'chi2': chi2,
            'ndf': ndf,
            'chi2_ndf': chi2_ndf,
            'fit_range': (np.min(x_fit), np.max(x_fit)),
            'n_points': len(x_fit),
            'fit_method': fit_method,
            'fit_func': fit_func,
            'm_w': m_w_fit,
            'm_w_err': m_w_err,
            'gamma_w': gamma_w_fit,
            'gamma_w_err': gamma_w_err
        }
        
        return popt, pcov, fit_info
        
    except Exception as e:
        print(f"\n✗ Fit failed: {e}")
        return None, None, None


def plot_fit_results(bin_centers, counts, errors, popt, pcov, fit_info, output_path=None):
    """
    Create publication-quality plot of histogram and fit.
    
    Parameters:
    -----------
    bin_centers : numpy array
        Bin centers in GeV
    counts : numpy array
        Event counts per bin
    errors : numpy array
        Statistical uncertainties
    popt : array
        Fit parameters [m_w, gamma_w, amplitude]
    pcov : array
        Covariance matrix
    fit_info : dict
        Fit quality information
    output_path : str or Path, optional
        Path to save the plot (default: show only)
    """
    if popt is None:
        print("Cannot plot: fit failed")
        return
    
    print("\n=== Creating Plot ===")
    
    # Extract uncertainties
    perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0, 0, 0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), 
                                     gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    
    # Main plot: data and fit
    ax1.errorbar(bin_centers, counts, yerr=errors, fmt='ko', 
                 markersize=4, capsize=2, label='Data', alpha=0.7)
    
    # Plot fit curve
    m_smooth = np.linspace(bin_centers[0], bin_centers[-1], 1000)
    fit_func = fit_info.get('fit_func', breit_wigner)
    fit_curve = fit_func(m_smooth, *popt)
    
    # Label based on fit method
    fit_method = fit_info.get('fit_method', 'breit_wigner')
    if fit_method == 'crystal_ball':
        fit_label = 'Crystal Ball Fit'
    elif fit_method == 'restricted_range':
        fit_label = 'Breit-Wigner Fit (Restricted)'
    else:
        fit_label = 'Breit-Wigner Fit'
    
    ax1.plot(m_smooth, fit_curve, 'r-', linewidth=2, label=fit_label)
    
    # Add fit results as text
    m_w = fit_info.get('m_w', popt[0])
    m_w_err = fit_info.get('m_w_err', perr[0])
    gamma_w = fit_info.get('gamma_w', popt[1] if len(popt) > 1 else 0)
    gamma_w_err = fit_info.get('gamma_w_err', perr[1] if len(perr) > 1 else 0)
    
    textstr = f'$m_W = {m_w:.2f} \\pm {m_w_err:.2f}$ GeV\n'
    textstr += f'$\\Gamma_W = {gamma_w:.2f} \\pm {gamma_w_err:.2f}$ GeV\n'
    if fit_info is not None:
        textstr += f'$\\chi^2$/NDF = {fit_info["chi2_ndf"]:.2f}'
    
    ax1.text(0.65, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_ylabel('Events per bin', fontsize=12)
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(bin_centers[0], bin_centers[-1])
    ax1.tick_params(labelbottom=False)
    
    # Residual plot
    fit_func = fit_info.get('fit_func', breit_wigner)
    y_pred = fit_func(bin_centers, *popt)
    residuals = (counts - y_pred) / errors
    
    ax2.axhline(0, color='r', linestyle='--', linewidth=1)
    ax2.errorbar(bin_centers, residuals, yerr=1, fmt='ko', 
                 markersize=4, capsize=2, alpha=0.7)
    ax2.axhline(2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(-2, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Transverse Mass $m_T$ [GeV]', fontsize=12)
    ax2.set_ylabel('Pull\n(Data-Fit)/$\\sigma$', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(bin_centers[0], bin_centers[-1])
    ax2.set_ylim(-5, 5)
    
    plt.suptitle('W Boson Transverse Mass Distribution', fontsize=14, fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main(neutrino_method='conservation'):
    """
    Main analysis pipeline.
    
    Parameters:
    -----------
    neutrino_method : str
        Method for obtaining neutrino 4-momentum:
        - 'conservation': Calculate from momentum/energy conservation laws
        - 'csv': Load generator-level truth from CSV file
    """
    print("="*70)
    print("W BOSON MASS AND WIDTH ANALYSIS")
    print("="*70)
    print(f"\nNeutrino reconstruction method: {neutrino_method.upper()}")
    print("-"*70)
    
    data_path = Path(__file__).parent / "w_boson_analysis.csv"
    
    try:
        # Step 1: Load and validate data
        load_neutrino = (neutrino_method == 'csv')
        data = load_and_validate_data(data_path, load_neutrino=load_neutrino)
        print(f"\n✓ Step 1 complete: Successfully loaded and validated {data['n_events']} events")
        
        # Step 2: Calculate muon energy
        e_mu = calculate_muon_energy(data['p_mu'])
        data['e_mu'] = e_mu
        print(f"✓ Step 2 complete: Calculated muon energies")
        
        # Step 3: Get neutrino 4-momentum
        if neutrino_method == 'conservation':
            # Reconstruct from conservation laws
            p_nu, e_nu = reconstruct_neutrino(data['p_mu'], data['e_mu'],
                                             data['p_other'], data['e_other'])
            data['p_nu'] = p_nu
            data['e_nu'] = e_nu
            print(f"✓ Step 3 complete: Reconstructed neutrino 4-momentum using conservation laws")
            
        elif neutrino_method == 'csv':
            # Use generator-level values from CSV
            if 'p_nu' not in data:
                raise ValueError("Neutrino data not loaded from CSV")
            e_nu = calculate_neutrino_energy_from_momentum(data['p_nu'])
            data['e_nu'] = e_nu
            print(f"✓ Step 3 complete: Calculated neutrino energy from CSV momentum")
            
        else:
            raise ValueError(f"Unknown neutrino method: {neutrino_method}. Use 'conservation' or 'csv'")
        
        # Step 4: Calculate transverse quantities
        p_t_mu, p_t_nu, delta_phi = calculate_transverse_quantities(data['p_mu'], data['p_nu'])
        data['p_t_mu'] = p_t_mu
        data['p_t_nu'] = p_t_nu
        data['delta_phi'] = delta_phi
        print(f"✓ Step 4 complete: Calculated transverse momenta and Δφ")
        
        # Step 5: Calculate transverse mass
        m_t = calculate_transverse_mass(p_t_mu, p_t_nu, delta_phi)
        data['m_t'] = m_t
        print(f"✓ Step 5 complete: Calculated transverse mass distribution")
        
        # Step 6: Create histogram and fit
        bin_centers, counts, bin_edges, errors = create_histogram(m_t, bins=80, range_min=0, range_max=150)
        data['histogram'] = {
            'bin_centers': bin_centers,
            'counts': counts,
            'bin_edges': bin_edges,
            'errors': errors
        }
        print(f"✓ Step 6a complete: Created histogram")
        
        # Fit the distribution
        popt, pcov, fit_info = fit_transverse_mass(bin_centers, counts, errors, fit_method=FIT_METHOD)
        if popt is not None:
            data['fit_results'] = {
                'parameters': popt,
                'covariance': pcov,
                'fit_info': fit_info,
                'm_w': popt[0],
                'm_w_err': np.sqrt(pcov[0, 0]) if pcov is not None else np.nan,
                'gamma_w': popt[1],
                'gamma_w_err': np.sqrt(pcov[1, 1]) if pcov is not None else np.nan
            }
            print(f"✓ Step 6b complete: Fitted distribution")
            
            # Create plot with descriptive filename based on methods used
            plot_filename = f"w_boson_fit_{neutrino_method}_{FIT_METHOD}.png"
            output_plot = Path(__file__).parent / plot_filename
            plot_fit_results(bin_centers, counts, errors, popt, pcov, fit_info, output_path=output_plot)
            print(f"✓ Step 6c complete: Created plot")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        
        if 'fit_results' in data:
            print("\n" + "="*70)
            print(f"FINAL RESULTS (Method: {FIT_METHOD})")
            print("="*70)
            print(f"W boson mass:  m_W = {data['fit_results']['m_w']:.3f} ± {data['fit_results']['m_w_err']:.3f} GeV")
            print(f"W boson width: Γ_W = {data['fit_results']['gamma_w']:.3f} ± {data['fit_results']['gamma_w_err']:.3f} GeV")
            print(f"χ²/NDF: {data['fit_results']['fit_info']['chi2_ndf']:.2f}")
            print("="*70)
        
        return data
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run main analysis with the configured method
    # To switch methods, change NEUTRINO_METHOD constant at the top of the file
    data = main(neutrino_method=NEUTRINO_METHOD)
