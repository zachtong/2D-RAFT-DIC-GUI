"""
Strain calculation utilities for RAFT-DIC.

- Virtual Strain Gauge (VSG) method with weighted least-squares
- Rotation field via polar decomposition (Numba-optimized)
"""

import time

import numpy as np
from scipy.signal import fftconvolve

# Numba-optimized rotation angle calculation
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

@jit(nopython=True, parallel=True, cache=True)
def _compute_rotation_field_numba(du_dx, du_dy, dv_dx, dv_dy):
    """
    Compute rotation angle field using polar decomposition with analytical 2x2 solutions.
    Numba JIT-compiled for high performance.

    For each pixel:
    1. Build deformation gradient: F = I + grad(u)
    2. Compute right Cauchy-Green tensor: C = F^T * F
    3. Analytical eigendecomposition of 2x2 symmetric matrix C
    4. Compute U = sqrt(C) and R = F * U^(-1)
    5. Extract rotation angle: theta = atan2(R21, R11)

    Returns rotation angle in degrees.
    """
    H, W = du_dx.shape
    rotation = np.full((H, W), np.nan)

    for i in prange(H):
        for j in range(W):
            # Skip invalid pixels
            ux = du_dx[i, j]
            uy = du_dy[i, j]
            vx = dv_dx[i, j]
            vy = dv_dy[i, j]

            if np.isnan(ux) or np.isnan(uy) or np.isnan(vx) or np.isnan(vy):
                continue

            # Deformation gradient tensor: F = I + grad(u)
            # F = [[F11, F12], [F21, F22]]
            F11 = 1.0 + ux
            F12 = uy
            F21 = vx
            F22 = 1.0 + vy

            # Right Cauchy-Green tensor: C = F^T * F
            # C = [[C11, C12], [C12, C22]] (symmetric)
            C11 = F11 * F11 + F21 * F21
            C12 = F11 * F12 + F21 * F22
            C22 = F12 * F12 + F22 * F22

            # Analytical eigenvalue decomposition for 2x2 symmetric matrix
            # eigenvalues: lambda = (trace/2) +/- sqrt((trace/2)^2 - det)
            trace = C11 + C22
            det = C11 * C22 - C12 * C12

            half_trace = trace * 0.5
            discriminant = half_trace * half_trace - det

            if discriminant < 0:
                continue

            sqrt_disc = np.sqrt(discriminant)
            lambda1 = half_trace + sqrt_disc
            lambda2 = half_trace - sqrt_disc

            if lambda1 < 0 or lambda2 < 0:
                continue

            # Compute sqrt of eigenvalues
            sqrt_l1 = np.sqrt(lambda1)
            sqrt_l2 = np.sqrt(lambda2)

            # For 2x2 symmetric matrix, we can compute U = sqrt(C) directly
            # Using the formula: U = (C + sqrt(det)*I) / (sqrt(lambda1) + sqrt(lambda2))
            # This avoids explicit eigenvector computation
            sqrt_det = np.sqrt(det)
            denom = sqrt_l1 + sqrt_l2

            if abs(denom) < 1e-12:
                continue

            U11 = (C11 + sqrt_det) / denom
            U12 = C12 / denom
            U22 = (C22 + sqrt_det) / denom

            # Compute U^(-1) for 2x2 matrix
            det_U = U11 * U22 - U12 * U12
            if abs(det_U) < 1e-12:
                continue

            inv_det_U = 1.0 / det_U
            U_inv11 = U22 * inv_det_U
            U_inv12 = -U12 * inv_det_U
            U_inv22 = U11 * inv_det_U

            # Rotation matrix: R = F * U^(-1)
            R11 = F11 * U_inv11 + F12 * U_inv12
            R12 = F11 * U_inv12 + F12 * U_inv22
            R21 = F21 * U_inv11 + F22 * U_inv12
            R22 = F21 * U_inv12 + F22 * U_inv22

            # Ensure proper rotation (det(R) = 1)
            # For 2D, we can just use atan2 directly if R is close to orthogonal
            # The SVD orthogonalization is replaced by direct angle extraction
            # since the analytical solution should give a proper rotation matrix

            # Extract rotation angle (2D) in degrees
            rotation[i, j] = np.degrees(np.arctan2(R21, R11))

    return rotation

def calculate_strain_field(displacement_field: np.ndarray, method: str = 'green_lagrange',
                         vsg_size: int = 31, poly_order: int = 1, weighting: str = 'Gaussian', step: int = 1):
    """
    Calculate strain field from displacement using VSG method with robust boundary handling.
    Implements "Vectorized Weighted Least Squares" to handle invalid pixels (Strategy 2).

    Args:
        displacement_field: (H, W, 2) array with (u, v) displacements.
        method: 'green_lagrange' (default) or 'engineering'.
        vsg_size: Size of the local window (odd int).
        poly_order: Polynomial order (1 or 2).
        weighting: 'Uniform' or 'Gaussian'.
        step: Calculation stride (downsampling factor).

    Returns:
        strain_dict: Dictionary containing strain components.
    """
    import time
    t_start = time.perf_counter()

    u = displacement_field[..., 0]
    v = displacement_field[..., 1]

    H, W = u.shape
    print(f"[TIMING] Strain calc: input size {H}x{W}, VSG={vsg_size}, step={step}")

    # 1. Create Validity Mask (1 for valid, 0 for invalid)
    mask = (~np.isnan(u)) & (~np.isnan(v))
    mask_float = mask.astype(np.float64)

    if not np.any(mask):
        return None

    # Fill NaNs with 0 for correlation (they will be weighted by 0 via mask_float)
    u_filled = np.nan_to_num(u)
    v_filled = np.nan_to_num(v)

    # 2. Generate Basis Kernels for the Window
    if vsg_size % 2 == 0:
        raise ValueError("VSG Size must be odd.")

    half = vsg_size // 2
    x_range = np.arange(-half, half + 1)
    y_range = np.arange(-half, half + 1)
    X_grid, Y_grid = np.meshgrid(x_range, y_range) # Local coordinates

    # Weighting Kernel
    if weighting == 'Gaussian':
        sigma = vsg_size / 4.0
        dist_sq = X_grid**2 + Y_grid**2
        G = np.exp(-dist_sq / (2 * sigma**2))
    else:
        G = np.ones((vsg_size, vsg_size))

    # Define Basis Functions
    # Order 1: [1, x, y]
    # Order 2: [1, x, y, x^2, xy, y^2]
    basis_funcs = [np.ones_like(X_grid), X_grid, Y_grid]
    if poly_order == 2:
        basis_funcs.extend([X_grid**2, X_grid*Y_grid, Y_grid**2])

    num_params = len(basis_funcs)

    # 3. Construct Linear System M * a = b for each pixel
    # M_kl = sum(w * phi_k * phi_l) -> Correlate(mask_float, G * phi_k * phi_l)
    # b_k  = sum(w * u * phi_k)     -> Correlate(mask_float * u_filled, G * phi_k)

    # Pre-compute weighted basis kernels for M
    # We need to compute upper triangular part of M (symmetric)
    # M is (H, W, num_params, num_params)

    # To save memory, we can compute and downsample immediately if step > 1?
    # No, correlation needs full grid. But we can slice result immediately.

    # Output grid coordinates
    # If step > 1, we only solve for pixels on the grid
    # But correlation must run on full image to capture neighbors correctly.

    # Initialize M and b containers (downsampled size)
    # We use lists to store columns/elements to avoid huge 4D arrays

    # Slicing for downsampling
    s_slice = slice(None, None, step)

    t1 = time.perf_counter()
    print(f"[TIMING] Strain calc - setup: {t1-t_start:.3f}s")

    # Build M (Symmetric) - Using FFT convolution for ~40x speedup
    M_elements = [[None for _ in range(num_params)] for _ in range(num_params)]

    for i in range(num_params):
        for j in range(i, num_params):
            # Kernel for M_ij: G * phi_i * phi_j
            kernel_M = G * basis_funcs[i] * basis_funcs[j]
            # Use FFT convolution - flip kernel to match correlate behavior
            kernel_flipped = kernel_M[::-1, ::-1]
            res = fftconvolve(mask_float, kernel_flipped, mode='same')

            # Downsample
            res_down = res[s_slice, s_slice]
            M_elements[i][j] = res_down
            if i != j:
                M_elements[j][i] = res_down # Symmetry

    t2 = time.perf_counter()
    print(f"[TIMING] Strain calc - M matrix FFT ({6 if poly_order==1 else 21} convolutions): {t2-t1:.3f}s")

    # Build b for u and v
    # b_u_k = sum(w * u * phi_k)
    b_u_elements = []
    b_v_elements = []

    # Pre-multiply data by mask
    u_masked = u_filled * mask_float
    v_masked = v_filled * mask_float

    for k in range(num_params):
        # Kernel for b_k: G * phi_k
        kernel_b = G * basis_funcs[k]
        kernel_b_flipped = kernel_b[::-1, ::-1]

        res_u = fftconvolve(u_masked, kernel_b_flipped, mode='same')
        res_v = fftconvolve(v_masked, kernel_b_flipped, mode='same')

        b_u_elements.append(res_u[s_slice, s_slice])
        b_v_elements.append(res_v[s_slice, s_slice])

    t3 = time.perf_counter()
    print(f"[TIMING] Strain calc - b vector FFT ({3 if poly_order==1 else 6}x2 convolutions): {t3-t2:.3f}s")

    # 4. Solve Linear Systems
    # Stack into arrays
    # M_stack: (N_pixels, num_params, num_params)
    # b_u_stack: (N_pixels, num_params)

    # Flatten spatial dimensions for batch solving
    H_down, W_down = M_elements[0][0].shape
    N_pixels = H_down * W_down

    M_stack = np.empty((N_pixels, num_params, num_params))
    b_u_stack = np.empty((N_pixels, num_params))
    b_v_stack = np.empty((N_pixels, num_params))

    for i in range(num_params):
        b_u_stack[:, i] = b_u_elements[i].flatten()
        b_v_stack[:, i] = b_v_elements[i].flatten()
        for j in range(num_params):
            M_stack[:, i, j] = M_elements[i][j].flatten()

    # Solve
    # Check for singular matrices (too few points)
    # We can check condition number or determinant, but simplest is to try solve and catch,
    # or rely on pseudoinverse. Pinv is safer for ill-conditioned boundaries.

    # Using pinv is slower but robust.
    # M_inv = np.linalg.pinv(M_stack)
    # coeffs_u = M_inv @ b_u_stack[..., None]

    # Let's use solve for speed, but mask out bad pixels?
    # A pixel is "bad" if sum of weights (M_00) is too small.
    min_weight = 1e-6
    valid_solve_mask = M_stack[:, 0, 0] > min_weight

    coeffs_u = np.full((N_pixels, num_params), np.nan)
    coeffs_v = np.full((N_pixels, num_params), np.nan)

    t4 = time.perf_counter()
    print(f"[TIMING] Strain calc - stack arrays: {t4-t3:.3f}s, solving {np.sum(valid_solve_mask)}/{N_pixels} pixels")

    # Only solve for valid pixels
    if np.any(valid_solve_mask):
        try:
            # Standard solve
            M_valid = M_stack[valid_solve_mask]
            # Reshape b to (N, 3, 1) to ensure correct broadcasting
            b_u_valid = b_u_stack[valid_solve_mask][..., None]
            b_v_valid = b_v_stack[valid_solve_mask][..., None]

            # Result will be (N, 3, 1), squeeze back to (N, 3)
            coeffs_u[valid_solve_mask] = np.linalg.solve(M_valid, b_u_valid).squeeze(-1)
            coeffs_v[valid_solve_mask] = np.linalg.solve(M_valid, b_v_valid).squeeze(-1)
        except np.linalg.LinAlgError:
            # Fallback to lstsq or pinv if singular
            # This is slow, maybe just iterate or use pinv on the batch
            # For now, let's use pinv on the valid subset
            M_valid = M_stack[valid_solve_mask]
            b_u_valid = b_u_stack[valid_solve_mask][..., None]
            b_v_valid = b_v_stack[valid_solve_mask][..., None]

            M_inv = np.linalg.pinv(M_valid)
            coeffs_u[valid_solve_mask] = (M_inv @ b_u_valid).squeeze(-1)
            coeffs_v[valid_solve_mask] = (M_inv @ b_v_valid).squeeze(-1)

    t5 = time.perf_counter()
    print(f"[TIMING] Strain calc - linear solve: {t5-t4:.3f}s")

    # 5. Extract Gradients
    # a0, a1(x), a2(y), ...
    # du/dx = a1
    # du/dy = a2

    du_dx_flat = coeffs_u[:, 1]
    du_dy_flat = coeffs_u[:, 2]
    dv_dx_flat = coeffs_v[:, 1]
    dv_dy_flat = coeffs_v[:, 2]

    # Reshape back to grid
    du_dx = du_dx_flat.reshape(H_down, W_down)
    du_dy = du_dy_flat.reshape(H_down, W_down)
    dv_dx = dv_dx_flat.reshape(H_down, W_down)
    dv_dy = dv_dy_flat.reshape(H_down, W_down)

    # 6. Strain Calculation
    if method == 'green_lagrange':
        exx = du_dx + 0.5 * (du_dx**2 + dv_dx**2)
        eyy = dv_dy + 0.5 * (du_dy**2 + dv_dy**2)
        exy = 0.5 * (du_dy + dv_dx + du_dx*du_dy + dv_dx*dv_dy)

    elif method == 'engineering':
        exx = du_dx
        eyy = dv_dy
        exy = 0.5 * (du_dy + dv_dx)

    else:
        raise ValueError(f"Unknown strain method: {method}")

    # Principal Strains
    center = (exx + eyy) / 2.0
    radius = np.sqrt(((exx - eyy) / 2.0)**2 + exy**2)
    e1 = center + radius
    e2 = center - radius
    max_shear = radius
    von_mises = np.sqrt(e1**2 - e1*e2 + e2**2)

    # Rotation Angle via Polar Decomposition (Numba-optimized)
    # F = I + grad(u) -> C = F^T*F -> U = sqrt(C) -> R = F*U^(-1) -> theta = atan2(R21, R11)
    rotation = _compute_rotation_field_numba(du_dx, du_dy, dv_dx, dv_dy)

    # Mask out pixels that were originally invalid (to prevent ROI expansion)
    # Downsample mask
    mask_down = mask[s_slice, s_slice]

    # Apply mask to all components
    for key, val in locals().items():
        if key in ['exx', 'eyy', 'exy', 'e1', 'e2', 'max_shear', 'von_mises', 'rotation']:
            val[~mask_down] = np.nan

    return {
        'exx': exx,
        'eyy': eyy,
        'exy': exy,
        'e1': e1,
        'e2': e2,
        'max_shear': max_shear,
        'von_mises': von_mises,
        'rotation': rotation
    }


def smooth_strain_temporal(strain_results: list, sigma_t: float = 1.0) -> list:
    """Apply Gaussian smoothing along the time axis for each strain component.

    Args:
        strain_results: List of strain dicts (one per frame).
        sigma_t: Gaussian sigma in frame units (e.g. 1.0 = smooth over ~3 frames).

    Returns:
        New list of strain dicts with temporally smoothed values.
    """
    if not strain_results or sigma_t <= 0:
        return strain_results

    from scipy.ndimage import gaussian_filter1d

    keys = [k for k in strain_results[0].keys() if strain_results[0][k] is not None]
    T = len(strain_results)

    smoothed = [{} for _ in range(T)]
    for key in keys:
        stack = np.array([s[key] for s in strain_results])  # (T, H, W)
        valid = np.isfinite(stack)
        filled = np.nan_to_num(stack, nan=0.0)
        weight = valid.astype(np.float64)
        smoothed_data = gaussian_filter1d(filled, sigma=sigma_t, axis=0)
        smoothed_weight = gaussian_filter1d(weight, sigma=sigma_t, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(smoothed_weight > 0.01, smoothed_data / smoothed_weight, np.nan)
        for t in range(T):
            smoothed[t][key] = result[t].astype(np.float32)

    return smoothed
