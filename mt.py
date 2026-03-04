"""
EAE 129 Midterm Project — Longitudinal Stability & Control (Aggie UAV)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 1) Raw data (from wind tunnel tables)
# -----------------------------
ALPHA_DEG = np.array([-5, 0, 5, 10, 15, 20, 25, 30], dtype=float)

DATA = {
    -5.0: {  # delta_e = -5 deg
        "CL": np.array([0.0580, 0.3800, 0.7020, 1.0240, 1.3460, 1.6680, 1.9900, 2.3120]),
        "CM": np.array([0.3385, 0.3100, 0.2815, 0.2530, 0.2245, 0.1960, 0.1675, 0.1390]),
    },
    0.0: {   # delta_e = 0 deg
        "CL": np.array([0.0780, 0.4000, 0.7220, 1.0440, 1.3660, 1.6880, 2.0100, 2.3320]),
        "CM": np.array([0.2785, 0.2500, 0.2215, 0.1930, 0.1645, 0.1360, 0.1075, 0.0790]),
    },
    5.0: {   # delta_e = +5 deg
        "CL": np.array([0.0980, 0.4200, 0.7420, 1.0640, 1.3860, 1.7080, 2.0300, 2.3520]),
        "CM": np.array([0.2185, 0.1900, 0.1615, 0.1330, 0.1045, 0.0760, 0.0475, 0.0190]),
    },
}


def build_long_dataframe() -> pd.DataFrame:
    """Return a tidy (long-format) DataFrame with columns: alpha_deg, de_deg, CL, CM."""
    rows = []
    for de, series in DATA.items():
        for a, cl, cm in zip(ALPHA_DEG, series["CL"], series["CM"]):
            rows.append({"alpha_deg": a, "de_deg": de, "CL": float(cl), "CM": float(cm)})
    return pd.DataFrame(rows)


df = build_long_dataframe()


# -----------------------------
# 2) Regression models for CL and CM
#    CL = CL0 + CL_alpha * alpha + CL_de * de
#    CM = CM0 + CM_alpha * alpha + CM_de * de
# -----------------------------
def fit_linear_model(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve y ≈ X b using least squares.
    Returns (b, residuals_vector) where residuals_vector = y - Xb.
    """
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    return b, resid


def design_matrix(alpha_deg: np.ndarray, de_deg: np.ndarray) -> np.ndarray:
    """Build [1, alpha, de] design matrix."""
    return np.column_stack([np.ones_like(alpha_deg, dtype=float), alpha_deg.astype(float), de_deg.astype(float)])


X = design_matrix(df["alpha_deg"].to_numpy(), df["de_deg"].to_numpy())

b_CL, r_CL = fit_linear_model(df["CL"].to_numpy(), X)
b_CM, r_CM = fit_linear_model(df["CM"].to_numpy(), X)

CL0, CL_alpha, CL_de = b_CL
CM0, CM_alpha, CM_de = b_CM

print("\n=== Estimated derivatives from global least-squares fit ===")
print(f"CL0      = {CL0:.4f}")
print(f"CL_alpha = {CL_alpha:.4f} per deg")
print(f"CL_de    = {CL_de:.4f} per deg")
print(f"CM0      = {CM0:.4f}")
print(f"CM_alpha = {CM_alpha:.4f} per deg")
print(f"CM_de    = {CM_de:.4f} per deg")

print("\nRMS residuals:")
print(f"  CL RMS = {np.sqrt(np.mean(r_CL**2)):.3e}")
print(f"  CM RMS = {np.sqrt(np.mean(r_CM**2)):.3e}")


# -----------------------------
# 3) Static margin from Cm vs CL slope
#    For small perturbations (linear):
#      Cm = Cm0 + Cm_alpha * alpha + ...
#      CL = CL0 + CL_alpha * alpha + ...
# -----------------------------
def static_margin_from_derivatives(CM_a: float, CL_a: float) -> float:
    if abs(CL_a) < 1e-12:
        raise ValueError("CL_alpha too small to compute static margin.")
    return -CM_a / CL_a


SM = static_margin_from_derivatives(CM_alpha, CL_alpha)
print(f"\nStatic Margin (from -CM_alpha/CL_alpha): SM = {SM:.3f}")


# -----------------------------
# 4) Elevator trim curve
#    Trim is CM = 0 => 0 = CM0 + CM_alpha*alpha + CM_de*de_trim
#    de_trim(alpha) = -(CM0 + CM_alpha*alpha) / CM_de
# -----------------------------
def elevator_trim_deg(alpha_deg: np.ndarray, CM0_: float, CM_a: float, CM_de_: float) -> np.ndarray:
    if abs(CM_de_) < 1e-12:
        raise ValueError("CM_de too small to compute trim deflection.")
    return -(CM0_ + CM_a * alpha_deg) / CM_de_


alpha_grid = np.linspace(ALPHA_DEG.min(), ALPHA_DEG.max(), 200)
de_trim = elevator_trim_deg(alpha_grid, CM0, CM_alpha, CM_de)


# -----------------------------
# 5) Neutral point estimate (geometric / tail-volume style)
#    This matches the “B-factor” approach in the reference, but reorganized.
# -----------------------------
def finite_wing_CLalpha(c_l_alpha_per_deg: float, AR: float, e: float) -> float:
    """
    Convert 2D airfoil slope (per deg) to finite-wing CL_alpha (per deg).
    Using: CL_a = c_l_a / (1 + (57.3*c_l_a)/(pi*e*AR))
    """
    return c_l_alpha_per_deg / (1.0 + (57.3 * c_l_alpha_per_deg) / (np.pi * e * AR))


def tail_factor_B(CL_aw: float, CL_ah: float, eta_h: float, Sh: float, Sw: float, d_eps_d_alpha: float) -> float:
    """
    B = (CL_ah / CL_aw) * eta_h * (Sh/Sw) * (1 - dε/dα)
    """
    return (CL_ah / CL_aw) * eta_h * (Sh / Sw) * (1.0 - d_eps_d_alpha)


def neutral_point_xbar(xbar_ac_w: float, xbar_ac_h: float, B: float) -> float:
    """
    xbar_np = (xbar_ac_w + B*xbar_ac_h) / (1 + B)
    """
    return (xbar_ac_w + B * xbar_ac_h) / (1.0 + B)


# --- Example geometry values

c_l_alpha_2d = 0.11      # per deg (airfoil)
eta_tail = 1.0           # tail efficiency
d_eps_d_alpha = 0.0      # downwash gradient

AR_w = 1.676
AR_h = 3.82
oswald_e = 1.0

S_w = 150.0              # in^2
S_h = 30.0               # in^2
xbar_ac_w = 0.25         # nondimensional 
xbar_ac_h = 4.0          # nondimensional 

CL_aw = finite_wing_CLalpha(c_l_alpha_2d, AR_w, oswald_e)
CL_ah = finite_wing_CLalpha(c_l_alpha_2d, AR_h, oswald_e)
B = tail_factor_B(CL_aw, CL_ah, eta_tail, S_h, S_w, d_eps_d_alpha)
xbar_np = neutral_point_xbar(xbar_ac_w, xbar_ac_h, B)

xbar_cg = 1.0  # per project convention / normalization
SM_geom = xbar_np - xbar_cg

print("\n=== Neutral point / static margin (geometry-style cross-check) ===")
print(f"CL_aw (finite) = {CL_aw:.4f} per deg")
print(f"CL_ah (finite) = {CL_ah:.4f} per deg")
print(f"B-factor       = {B:.4f}")
print(f"xbar_NP         = {xbar_np:.3f}")
print(f"xbar_CG         = {xbar_cg:.3f}")
print(f"SM_geom         = {SM_geom:.3f}")


# -----------------------------
# 6) Plotting functions
# -----------------------------
def plot_CL_vs_alpha():
    plt.figure(figsize=(8.5, 5.2))
    for de in sorted(DATA.keys()):
        plt.plot(ALPHA_DEG, DATA[de]["CL"], marker="o", label=f"de = {de:.0f} deg")
    plt.xlabel("Angle of Attack, alpha [deg]")
    plt.ylabel("Lift Coefficient, CL [-]")
    plt.title("CL vs alpha for selected elevator deflections")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_CM_vs_alpha():
    plt.figure(figsize=(8.5, 5.2))
    for de in sorted(DATA.keys()):
        plt.plot(ALPHA_DEG, DATA[de]["CM"], marker="o", label=f"de = {de:.0f} deg")
    plt.xlabel("Angle of Attack, alpha [deg]")
    plt.ylabel("Moment Coefficient about CG, CM [-]")
    plt.title("CM vs alpha for selected elevator deflections")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_CM_vs_CL():
    plt.figure(figsize=(8.5, 5.2))
    for de in sorted(DATA.keys()):
        plt.plot(DATA[de]["CL"], DATA[de]["CM"], marker="o", label=f"de = {de:.0f} deg")
    plt.xlabel("Lift Coefficient, CL [-]")
    plt.ylabel("Moment Coefficient, CM [-]")
    plt.title("CM vs CL for selected elevator deflections")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_trim_curve():
    plt.figure(figsize=(8.5, 5.2))
    plt.plot(alpha_grid, de_trim, linewidth=2)
    plt.xlabel("Trim angle of attack, alpha_trim [deg]")
    plt.ylabel("Elevator trim deflection, de_trim [deg]")
    plt.title("Elevator trim deflection vs trim angle of attack")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


if __name__ == "__main__":
    plot_CL_vs_alpha()
    plot_CM_vs_alpha()
    plot_CM_vs_CL()
    plot_trim_curve()
    plt.show()