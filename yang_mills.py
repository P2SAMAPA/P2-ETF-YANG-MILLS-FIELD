import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def yang_mills_connection(returns, macro_factor):
    """
    Compute the Yang-Mills connection (gauge potential) A = returns * macro_factor.
    This is a U(1) gauge field.
    """
    # For each time step, the connection is the return vector
    # We treat the connection as a 1-form: A = returns * d(macro_factor)
    # Simplified: A_t = returns_t * macro_factor_t
    # We'll compute the derivative of the connection to get the curvature
    if len(returns) < 2:
        return np.zeros_like(returns)
    A = returns * macro_factor
    return A

def yang_mills_curvature(A):
    """
    Compute the Yang-Mills curvature F = dA + A ∧ A.
    For U(1), the curvature is simply F = dA (since A ∧ A = 0 for abelian).
    """
    if len(A) < 2:
        return np.zeros_like(A)
    # Finite difference approximation of dA
    dA = np.diff(A)
    # Pad to maintain length
    curvature = np.concatenate([[0], dA])
    return curvature

def yang_mills_score(returns, macro_df):
    """
    Compute per-ETF Yang-Mills field strength.
    The score is the norm of the curvature (field strength) at the last time step,
    scaled by macro factor.
    """
    if len(returns) < 5 or macro_df is None or len(macro_df) < 5:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 5:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)
    # Compute connection
    A = yang_mills_connection(returns, macro_factor)
    # Compute curvature
    F = yang_mills_curvature(A)
    # Score: absolute curvature at the last point
    score = abs(F[-1])
    return float(score)

def yang_mills_parallel_transport(returns, macro_df):
    """
    Compute Wilson loop (parallel transport) around a closed loop in the market manifold.
    For a closed loop, the Wilson loop is exp(i ∮ A · dx).
    Since we have a 1D manifold (time), we approximate the loop by the last two points.
    """
    if len(returns) < 2 or macro_df is None or len(macro_df) < 2:
        return 0.0
    # Compute connection
    macro_factor = compute_composite_macro_factor(macro_df)
    A = yang_mills_connection(returns, macro_factor)
    # Wilson loop around the last two time steps
    loop_integral = np.sum(A[-2:])  # ∮ A · dx ≈ sum of A over the loop
    # For simplicity, return the imaginary part of exp(i * loop_integral)
    wilson_loop = np.sin(loop_integral)
    return float(wilson_loop)
