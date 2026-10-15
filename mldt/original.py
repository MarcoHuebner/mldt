"""Original implementation from ``demoMLDT.ipynb``.

This module intentionally preserves the notebook's numerical behavior. Use
the other backends for the maintained, numerically robust implementation.
"""

import numpy as np
import numpy.typing as npt
from scipy.integrate import quad
from scipy.linalg import solve_banded


def calculateBoundaryConstant_A(n_tissue: float, n_outside: float = 1.0) -> float:
    """Calculate the notebook's exact partial-current boundary constant."""

    def fresnel_unpol(theta_i: float) -> float:
        if np.sin(theta_i) >= (n_outside / n_tissue):
            return 1.0
        theta_t = np.arcsin((n_tissue / n_outside) * np.sin(theta_i))
        rs = (
            (n_tissue * np.cos(theta_i) - n_outside * np.cos(theta_t))
            / (n_tissue * np.cos(theta_i) + n_outside * np.cos(theta_t))
        ) ** 2
        rp = (
            (n_outside * np.cos(theta_i) - n_tissue * np.cos(theta_t))
            / (n_outside * np.cos(theta_i) + n_tissue * np.cos(theta_t))
        ) ** 2
        return 0.5 * (rs + rp)

    # First angular moment (R_phi)
    def int_R_phi(theta: float) -> float:
        return 2.0 * np.sin(theta) * np.cos(theta) * fresnel_unpol(theta)

    # Second angular moment (R_j)
    def int_R_j(theta: float) -> float:
        return 3.0 * np.sin(theta) * (np.cos(theta) ** 2) * fresnel_unpol(theta)

    # Integrate over the hemisphere (0 to pi/2)
    r_phi, _ = quad(int_R_phi, 0, np.pi / 2)
    r_j, _ = quad(int_R_j, 0, np.pi / 2)

    # Calculate final proportionality constant A
    return (1.0 - r_phi) / (2.0 * (1.0 + r_j))


def getRd_MLDT(
    mua_layers: npt.ArrayLike,
    musp_layers: npt.ArrayLike,
    thicknesses: npt.ArrayLike,
    nr: float,
) -> np.ndarray:
    """Compute diffuse reflectance with the original notebook algorithm."""
    ###################################################
    #### 1. SETUP
    ###################################################
    eps = 1e-10
    mua_layers = np.clip(np.asarray(mua_layers, dtype=float), eps, None)
    musp_layers = np.clip(np.asarray(musp_layers, dtype=float), eps, None)
    thicknesses = np.clip(np.asarray(thicknesses, dtype=float), eps, None)
    nr = float(nr)

    # The function was not set-up to handle 1-layer models.
    # This statement creates a 2-layer model with identical
    # optical properties if only 1 layer was used as input.
    if thicknesses.size == 0:
        # Give the first layer an arbitrary thickness
        # Since properties are identical, this interface is transparent.
        thicknesses = np.array([10.0])

        # Duplicate the optical properties for the second layer
        mua_layers = np.column_stack((mua_layers, mua_layers))
        musp_layers = np.column_stack((musp_layers, musp_layers))

    nnm, _num_total_layers = mua_layers.shape
    num_finite_layers = len(thicknesses)
    num_coeffs = 2 * num_finite_layers + 1

    ###################################################
    #### 2. CALCULATE COEFFICIENTS
    ###################################################
    mutr_all = mua_layers + musp_layers
    zeta_all = 1.0 / (3.0 * mutr_all)

    denom_delta = 3 * mua_layers * mutr_all
    delta_all = 1 / np.sqrt(np.clip(denom_delta, eps, None))

    boundary_constant = calculateBoundaryConstant_A(nr, n_outside=1.0)
    phi_t = 1.0

    # Source term S0
    res_denom = 1 - (mutr_all**2 * delta_all**2)
    res_denom = np.where(np.abs(res_denom) < eps, eps, res_denom)
    s0_base_all = (phi_t * delta_all**2 * musp_layers) / (zeta_all * res_denom)

    ###################################################
    #### 3. ATTENUATION
    ###################################################
    mutr_prev = mutr_all[:, :-1]
    att_factors = np.exp(-mutr_prev * thicknesses[None, :])
    cumulative_att = np.hstack((np.ones((nnm, 1)), np.cumprod(att_factors, axis=1)))
    s0_attenuated_all = s0_base_all * cumulative_att

    ###################################################
    #### 4. PREPARE BANDED MATRIX
    ###################################################
    # ab shape: (Nnm, 5, num_coeffs)
    # Row 0: Top (j=i+2), Row 1: Upper (j=i+1), Row 2: Main (j=i), Row 3: Lower (j=i-1), Row 4: Bottom (j=i-2)
    ab = np.zeros((nnm, 5, num_coeffs), dtype=float)
    b_vec = np.zeros((nnm, num_coeffs), dtype=float)
    k = np.arange(num_finite_layers)

    ###################################################
    #### 5. FILL DIAGONALS
    ###################################################
    #### 5a. Surface Boundary (Row 0)
    # A[0,0] -> Main (Row 2, Col 0)
    # A[0,1] -> Upper (Row 1, Col 1)
    ab[:, 2, 0] = (zeta_all[:, 0] / delta_all[:, 0]) + boundary_constant
    ab[:, 1, 1] = (-zeta_all[:, 0] / delta_all[:, 0]) + boundary_constant
    b_vec[:, 0] = -(boundary_constant * s0_attenuated_all[:, 0]) - (
        zeta_all[:, 0] * s0_attenuated_all[:, 0] * mutr_all[:, 0]
    )

    #### 5b. Internal Interfaces
    zeta_k = zeta_all[:, k]
    delta_k = delta_all[:, k]
    mutr_k = mutr_all[:, k]
    s0_att_k = s0_attenuated_all[:, k]
    zeta_kp1 = zeta_all[:, k + 1]
    delta_kp1 = delta_all[:, k + 1]
    mutr_kp1 = mutr_all[:, k + 1]
    s0_att_kp1 = s0_attenuated_all[:, k + 1]

    dz_k = thicknesses[None, :]
    exp_dec_k = np.exp(-dz_k / delta_k)
    exp_gro_k = np.exp(dz_k / delta_k)

    src_k = s0_att_k * np.exp(-mutr_k * dz_k)
    src_kp1 = s0_att_kp1
    d_src_k = -s0_att_k * mutr_k * np.exp(-mutr_k * dz_k)
    d_src_kp1 = -s0_att_kp1 * mutr_kp1

    # Column Indices
    col_k_dec = 2 * k
    col_k_gro = 2 * k + 1
    col_kp1_dec = 2 * (k + 1)
    col_kp1_gro = 2 * (k + 1) + 1
    mask_fin = k < (num_finite_layers - 1)

    #### Equation 2k+1 (Fluence Continuity)
    # C_k_dec (Col 2k): j = i-1 -> Lower (Row 3)
    ab[:, 3, col_k_dec] = exp_dec_k

    # C_k_gro (Col 2k+1): j = i -> Main (Row 2)
    ab[:, 2, col_k_gro] = exp_gro_k

    # C_kp1_dec (Col 2k+2): j = i+1 -> Upper (Row 1)
    ab[:, 1, col_kp1_dec] = -1.0

    # C_kp1_gro (Col 2k+3): j = i+2 -> Top (Row 0)
    ab[:, 0, col_kp1_gro[mask_fin]] = -1.0

    b_vec[:, 2 * k + 1] = src_kp1 - src_k

    #### Equation 2k+2 (Flux Continuity)
    term_k_dec = -zeta_k * (-exp_dec_k / delta_k)
    term_k_gro = -zeta_k * (exp_gro_k / delta_k)
    term_kp1_dec = zeta_kp1 * (-1.0 / delta_kp1)
    term_kp1_gro = zeta_kp1 * (1.0 / delta_kp1)

    # C_k_dec (Col 2k): j = i-2 -> Bottom (Row 4)
    ab[:, 4, col_k_dec] = term_k_dec

    # C_k_gro (Col 2k+1): j = i-1 -> Lower (Row 3)
    ab[:, 3, col_k_gro] = term_k_gro

    # C_kp1_dec (Col 2k+2): j = i -> Main (Row 2)
    ab[:, 2, col_kp1_dec] = term_kp1_dec

    # C_kp1_gro (Col 2k+3): j = i+1 -> Upper (Row 1)
    ab[:, 1, col_kp1_gro[mask_fin]] = term_kp1_gro[:, mask_fin]

    b_vec[:, 2 * k + 2] = -zeta_kp1 * d_src_kp1 + zeta_k * d_src_k

    ###################################################
    #### 6. SOLVE
    ###################################################
    coefficients = np.zeros((nnm, num_coeffs))

    # Loop over wavelengths
    for i in range(nnm):
        coefficients[i, :] = solve_banded((2, 2), ab[i], b_vec[i], overwrite_ab=True)

    ###################################################
    #### 7. REFLECTANCE
    ###################################################
    phi_0 = s0_attenuated_all[:, 0] + coefficients[:, 0] + coefficients[:, 1]
    j_0 = -boundary_constant * phi_0
    return -(j_0 / phi_t)


def reflectance(
    mua: npt.ArrayLike,
    musp: npt.ArrayLike,
    thicknesses: npt.ArrayLike,
    refractive_index: float,
    *,
    outside_refractive_index: float = 1.0,
) -> np.ndarray:
    """Expose the original notebook implementation through the backend API."""
    if outside_refractive_index != 1.0:
        raise ValueError("the original notebook implementation only supports outside index 1.0")
    return getRd_MLDT(mua, musp, thicknesses, refractive_index)


__all__ = ["calculateBoundaryConstant_A", "getRd_MLDT", "reflectance"]
