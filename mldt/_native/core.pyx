# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Native MLDT forward assembly and pivoted banded solve.

See mldt/_system.py for the mathematical formulation and original notation.
This kernel uses the same split phi = H + P, with P(0) = 0, and homogeneous
endpoint unknowns [u0, v0, ..., u_last]. Current is J = -D*phi'; the surface
condition is D*phi'(0) = A*phi(0), with continuous fluence and current at each
interface. The final layer is semi-infinite, and unit illumination gives
reflectance A*u0.

The Python wrapper in mldt/cython.py validates inputs and computes the Fresnel
constant A. This kernel assembles and solves one wavelength at a time, reusing
native workspaces. LAPACK dgbsv requires seven storage rows for pivoted LU,
whereas the SciPy solve_banded input in _system.py uses five.
"""

import numpy as np

from libc.math cimport exp, expm1, fabs, sqrt
from libc.stdlib cimport free, malloc
from libc.string cimport memset
from scipy.linalg.cython_lapack cimport dgbsv


cdef inline void set_band(double *ab, int row, int col, double value) noexcept nogil:
    """Store matrix entry (row, col) in the column-major LAPACK band buffer."""
    # For kl = ku = 2, ldab = 2*kl + ku + 1 = 7; the diagonal is at row 4.
    # Rows 0 and 1 reserve fill-in space for pivoting. SciPy's input instead
    # puts the diagonal at row 2 and omits these two workspace rows.
    ab[4 + row - col + 7 * col] = value


cdef inline void endpoint_rates(
    double lam, double thickness, double *left, double *cross
) noexcept nogil:
    """Write a = lam*coth(lam*d) and b = lam/sinh(lam*d) to output pointers."""
    # H'(0) = -a*u + b*v and H'(d) = -b*u + a*v for u = H(0), v = H(d).
    cdef double x = lam * thickness
    cdef double x2
    cdef double decay
    cdef double denom
    # Series through x**4 has relative remainder O(x**6); both rates approach
    # 1/d at zero absorption. Decaying exponentials avoid thick-layer overflow.
    if fabs(x) < 1e-4:
        x2 = x * x
        left[0] = (1.0 + x2 / 3.0 - x2 * x2 / 45.0) / thickness
        cross[0] = (1.0 - x2 / 6.0 + 7.0 * x2 * x2 / 360.0) / thickness
    else:
        decay = exp(-x)
        denom = -expm1(-2.0 * x)
        left[0] = lam * (1.0 + decay * decay) / denom
        cross[0] = lam * 2.0 * decay / denom


cdef inline void source_terms(
    double lam, double mu, double distance, double *value, double *derivative
) noexcept nogil:
    """Write P/F and P'/F at local depth distance to output pointers."""
    # P/F = (exp(-mu*z) - exp(-lam*z))/(lam**2 - mu**2).
    # The shifted source preserves the original solution while making P(0) = 0.
    cdef double gap = fabs(lam - mu)
    cdef double az = gap * distance
    cdef double h
    cdef double base
    cdef double exp_base
    # Evaluate h = (1 - exp(-gap*z))/gap by series for small gap*z.
    # Resonance means lam == mu: h -> z, and P/F -> z*exp(-lam*z)/(2*lam).
    # The relative series remainder O((gap*z)**3/24) is below float64 precision.
    if az <= 1e-7:
        h = distance - gap * distance * distance / 2.0
        h += gap * gap * distance * distance * distance / 6.0
    else:
        h = -expm1(-az) / gap
    base = lam if lam < mu else mu
    exp_base = exp(-base * distance)
    value[0] = exp_base * h / (lam + mu)
    derivative[0] = exp_base * (exp(-az) - base * h) / (lam + mu)


def reflectance(
    double[:, ::1] mua,
    double[:, ::1] musp,
    double[::1] thicknesses,
    double boundary_constant,
) -> np.ndarray:
    """Return float64 reflectance (n_wavelengths,) for validated optical inputs.

    mua and musp are float64 arrays (n_wavelengths, n_layers) contiguous along
    the layer axis. thicknesses is a contiguous float64 vector of n_layers - 1
    positive thicknesses. boundary_constant is the scalar Fresnel constant A.
    The wrapper enforces shapes and physical domains before entering this
    kernel, whose bounds checks are disabled. Inputs are read without mutation.
    """
    # Derive workspace sizes and allocate the per-wavelength output first.
    cdef int nw = mua.shape[0]
    cdef int nl = mua.shape[1]
    cdef int nf = nl - 1
    cdef int ncoef = 2 * nf + 1
    cdef int ldab = 7
    cdef int kl = 2
    cdef int ku = 2
    cdef int nrhs = 1
    cdef int ldb = ncoef
    cdef int w, k, r1, r2, col, info
    cdef int failed = -1
    cdef double cumulative
    cdef double p_unit = 0.0
    cdef double dp_unit = 0.0
    cdef double mutr_value, diffusion_value, lam_value, forcing_value, slope_value
    cdef double *mutr = NULL
    cdef double *diffusion = NULL
    cdef double *lam = NULL
    cdef double *forcing = NULL
    cdef double *dp_left = NULL
    cdef double *p_right = NULL
    cdef double *dp_right = NULL
    cdef double *rate_left = NULL
    cdef double *rate_cross = NULL
    cdef double *ab = NULL
    cdef double *rhs = NULL
    cdef int *ipiv = NULL
    cdef object result_array = np.empty(nw, dtype=np.float64)
    cdef double[::1] result = result_array

    # With no finite layers, solve (D*lam + A)*u = D*P'(0) directly.
    # This also includes the zero-absorption limit without an artificial layer.
    if nl == 1:
        for w in range(nw):
            mutr_value = mua[w, 0] + musp[w, 0]
            diffusion_value = 1.0 / (3.0 * mutr_value)
            lam_value = sqrt(3.0 * mua[w, 0] * mutr_value)
            forcing_value = musp[w, 0] / diffusion_value
            slope_value = forcing_value / (lam_value + mutr_value)
            result[w] = (
                boundary_constant * diffusion_value * slope_value
                / (diffusion_value * lam_value + boundary_constant)
            )
        return result_array

    # All malloc buffers belong to this call and hold only one wavelength.
    # Layer buffers have nl or nf entries; ab has 7*ncoef entries, rhs and ipiv
    # have ncoef entries. result_array owns the output via NumPy, independently.
    mutr = <double *>malloc(nl * sizeof(double))
    diffusion = <double *>malloc(nl * sizeof(double))
    lam = <double *>malloc(nl * sizeof(double))
    forcing = <double *>malloc(nl * sizeof(double))
    dp_left = <double *>malloc(nl * sizeof(double))
    p_right = <double *>malloc(nf * sizeof(double))
    dp_right = <double *>malloc(nf * sizeof(double))
    rate_left = <double *>malloc(nf * sizeof(double))
    rate_cross = <double *>malloc(nf * sizeof(double))
    ab = <double *>malloc(ldab * ncoef * sizeof(double))
    rhs = <double *>malloc(ncoef * sizeof(double))
    ipiv = <int *>malloc(ncoef * sizeof(int))
    if (
        mutr == NULL or diffusion == NULL or lam == NULL or forcing == NULL
        or dp_left == NULL or p_right == NULL or dp_right == NULL
        or rate_left == NULL or rate_cross == NULL or ab == NULL
        or rhs == NULL or ipiv == NULL
    ):
        # free(NULL) is valid, so partial allocation failures use the same cleanup.
        free(mutr); free(diffusion); free(lam); free(forcing); free(dp_left)
        free(p_right); free(dp_right); free(rate_left); free(rate_cross)
        free(ab); free(rhs); free(ipiv)
        raise MemoryError("unable to allocate native MLDT workspace")

    # Only C arithmetic, memory operations, and LAPACK run without the GIL.
    # Wavelengths are processed sequentially; releasing the GIL permits other
    # Python threads to run but does not itself parallelize this loop.
    with nogil:
        # Build each wavelength's optical coefficients and source values.
        for w in range(nw):
            cumulative = 1.0
            for k in range(nl):
                mutr[k] = mua[w, k] + musp[w, k]
                diffusion[k] = 1.0 / (3.0 * mutr[k])
                lam[k] = sqrt(3.0 * mua[w, k] * mutr[k])
                forcing[k] = cumulative * musp[w, k] / diffusion[k]
                # P'(0) = F/(lam + mu) remains finite at resonance.
                dp_left[k] = forcing[k] / (lam[k] + mutr[k])
                if k < nf:
                    cumulative *= exp(-mutr[k] * thicknesses[k])
            for k in range(nf):
                source_terms(lam[k], mutr[k], thicknesses[k], &p_unit, &dp_unit)
                p_right[k] = forcing[k] * p_unit
                dp_right[k] = forcing[k] * dp_unit
                endpoint_rates(
                    lam[k], thicknesses[k], &rate_left[k], &rate_cross[k]
                )

            # Reset buffers: dgbsv overwrites ab with LU factors and rhs with
            # the solution. Surface row: (D*a + A)*u - D*b*v = D*P'(0).
            memset(ab, 0, ldab * ncoef * sizeof(double))
            memset(rhs, 0, ncoef * sizeof(double))
            set_band(ab, 0, 0, diffusion[0] * rate_left[0] + boundary_constant)
            set_band(ab, 0, 1, -diffusion[0] * rate_cross[0])
            rhs[0] = diffusion[0] * dp_left[0]
            for k in range(nf):
                r1 = 2 * k + 1
                r2 = r1 + 1
                col = 2 * k
                # Fluence continuity: v_k + P_k(d_k) = u_(k+1).
                set_band(ab, r1, col + 1, 1.0)
                set_band(ab, r1, col + 2, -1.0)
                rhs[r1] = -p_right[k]
                # Current continuity: -D_k*phi_k'(d_k) = -D_(k+1)*phi_(k+1)'(0).
                set_band(ab, r2, col, diffusion[k] * rate_cross[k])
                set_band(ab, r2, col + 1, -diffusion[k] * rate_left[k])
                if k + 1 < nf:
                    set_band(
                        ab, r2, col + 2, -diffusion[k + 1] * rate_left[k + 1]
                    )
                    set_band(
                        ab, r2, col + 3, diffusion[k + 1] * rate_cross[k + 1]
                    )
                else:
                    # The last layer has H'(0) = -lam*u and no growing mode.
                    set_band(ab, r2, col + 2, -diffusion[k + 1] * lam[k + 1])
                rhs[r2] = (
                    diffusion[k] * dp_right[k]
                    - diffusion[k + 1] * dp_left[k + 1]
                )
            # ipiv records pivot indices. info > 0 means a zero U pivot;
            # info < 0 means an invalid LAPACK argument. Either stops the loop.
            info = 0
            dgbsv(&ncoef, &kl, &ku, &nrhs, ab, &ldab, ipiv, rhs, &ldb, &info)
            if info != 0:
                # Record failure here; Python exceptions require the GIL below.
                failed = w
                break
            # P(0) = 0, so rhs[0] is surface fluence after the solve.
            result[w] = boundary_constant * rhs[0]

    # The GIL is reacquired here. Free every workspace on success or LAPACK
    # failure before raising; the current error message treats any nonzero info
    # as singularity. The partially filled result is never returned on failure.
    free(mutr); free(diffusion); free(lam); free(forcing); free(dp_left)
    free(p_right); free(dp_right); free(rate_left); free(rate_cross)
    free(ab); free(rhs); free(ipiv)
    if failed >= 0:
        raise np.linalg.LinAlgError(
            f"native MLDT banded system is singular at wavelength index {failed}"
        )
    return result_array


def backend_info() -> dict[str, str | bool]:
    """Return implementation details for the native backend."""
    return {"implementation": "native-cython-lapack-banded", "openmp": False}
