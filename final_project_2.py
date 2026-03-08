import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

np.set_printoptions(suppress=True, precision=6)

# ============================================================
# EAE 129 Final Project
# XC-142A Longitudinal Dynamics
# ============================================================

# --------------------------------------------
# region flight condition and aircraft data 
# --------------------------------------------

u0 = 202.56          # trim flight speed [ft/s]
g = 32.2             # gravitational acceleration [ft/s^2]

# Stability derivatives from project statement
Xu = -0.22
Xw = 0.06
Zu = -0.15
Zw = -0.85

Mu = 0.01
Mw = -0.0095
Mq = -0.89
Mw_dot = -0.00127

# Control derivatives
Xde = 0.12
Zde = 4.58
Mde = 1.195

#fourth-order longitudinal state matrix
A = np.array([
    [Xu, Xw, 0.0, -g],
    [Zu, Zw, u0, 0.0],
    [Mu + Mw_dot * Zu, Mw + Mw_dot * Zw, Mq + Mw_dot * u0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
], dtype=float)

#control matrix
B = np.array([
    [Xde],
    [Zde],
    [Mde + Mw_dot * Zde],
    [0.0]
], dtype=float)

print("A matrix:")
print(A)
print("\nB matrix:")
print(B)

# endregion


# -----------------------------------------
# region eigenvalues and modal properties
# -----------------------------------------

C = np.eye(4)
D = np.zeros((4, 1))

ss_full = sig.StateSpace(A, B, C, D)

eigvals, _ = np.linalg.eig(A)

# magnitude of imaginary part:
eigvals_sorted = sorted(eigvals, key=lambda x: abs(np.imag(x)), reverse=True)

sp_eigs = np.array([eigvals_sorted[0], eigvals_sorted[1]])
lp_eigs = np.array([eigvals_sorted[2], eigvals_sorted[3]])

# Modal properties
def modal_properties(eigs: np.ndarray):
    coeffs = np.poly(eigs)
    nat_freq = np.sqrt(coeffs[2])
    damp_ratio = coeffs[1] / (2 * nat_freq)
    damp_freq = abs(np.imag(eigs[0]))
    time_const = -1 / np.real(eigs[0])
    return coeffs, nat_freq, damp_ratio, damp_freq, time_const

sp_coeffs, sp_natfreq, sp_dampratio, sp_dampfreq, sp_timeconst = modal_properties(sp_eigs)
lp_coeffs, lp_natfreq, lp_dampratio, lp_dampfreq, lp_timeconst = modal_properties(lp_eigs)

# endregion


# ------------------------------------------------------------
# region Free response and unit-step elevator response
# ------------------------------------------------------------

time = np.linspace(0, 200, 1000)

state_labels = [
    r"$\Delta u$, Forward Velocity Perturbation",
    r"$\Delta w$, Vertical Velocity Perturbation",
    r"$\Delta q$, Pitch Rate Perturbation",
    r"$\Delta \theta$, Pitch Angle Perturbation"
]

state_units = ["ft/s", "ft/s", "deg/s", "deg"]
state_colors = ["red", "orange", "blue", "green"]

# ---------- Free response ----------
u_free = np.zeros_like(time)
x0_free = np.array([0.0, 0.0, 0.0, 0.1])   # initial pitch-angle perturbation [rad]

tout_free, yout_free, xout_free = sig.lsim(ss_full, U=u_free, T=time, X0=x0_free)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Free Response of XC-142A Longitudinal System")

for i, ax in enumerate(axes.flat):
    if i < 2:
        ax.plot(time, xout_free[:, i], c=state_colors[i], label=state_labels[i])
    else:
        ax.plot(time, np.rad2deg(xout_free[:, i]), c=state_colors[i], label=state_labels[i])

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Response [{state_units[i]}]")
    ax.set_title(state_labels[i])
    ax.grid(True)

plt.tight_layout()
plt.show()

# ---------- Step response ----------
u_step = np.ones_like(time)     # unit step elevator input
x0_step = np.zeros(4)

tout_step, yout_step, xout_step = sig.lsim(ss_full, U=u_step, T=time, X0=x0_step)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Step Response of XC-142A Longitudinal System")

for i, ax in enumerate(axes.flat):
    if i < 2:
        ax.plot(time, xout_step[:, i], c=state_colors[i], label=state_labels[i])
    else:
        ax.plot(time, np.rad2deg(xout_step[:, i]), c=state_colors[i], label=state_labels[i])

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Response [{state_units[i]}]")
    ax.set_title(state_labels[i])
    ax.grid(True)

plt.tight_layout()
plt.show()

# endregion


# ----------------------
# region approximations
# ----------------------

# Short-period approximation 
Zalpha = -568.776
Malpha = -4.824
Malpha_dot = -0.474

A_sp = np.array([
    [Zalpha / u0, 1.0],
    [Malpha + Malpha_dot * (Zalpha / u0), Mq + Malpha_dot]
], dtype=float)

# phugoid approximation 
A_lp = np.array([
    [Xu, -g],
    [-Zu / u0, 0.0]
], dtype=float)

B_approx = np.zeros((2, 1))
C_approx = np.array([[1.0, 0.0]])
D_approx = np.array([[0.0]])

print("\nShort-Period Approximation A matrix:")
print(A_sp)

print("\nPhugoid Approximation A matrix:")
print(A_lp)

spapprox_ss = sig.StateSpace(A_sp, B_approx, C_approx, D_approx)
lpapprox_ss = sig.StateSpace(A_lp, B_approx, C_approx, D_approx)

approx_sp_eigs, _ = np.linalg.eig(A_sp)
approx_lp_eigs, _ = np.linalg.eig(A_lp)

approx_sp_coeffs, approx_sp_natfreq, approx_sp_dampratio, approx_sp_dampfreq, approx_sp_timeconst = modal_properties(approx_sp_eigs)
approx_lp_coeffs, approx_lp_natfreq, approx_lp_dampratio, approx_lp_dampfreq, approx_lp_timeconst = modal_properties(approx_lp_eigs)

# endregion


# ---------------
# region outputs
# ---------------

print("\n============================================================")
print("EXACT FOURTH-ORDER LONGITUDINAL SYSTEM")
print("============================================================")
print(f"Exact eigenvalues: {eigvals}")

print("\nShort-Period Mode:")
print(f"  Eigenvalues       : {sp_eigs}")
print(f"  Characteristic eq.: {sp_coeffs}")
print(f"  Natural frequency : {sp_natfreq:.4f} rad/s")
print(f"  Damping ratio     : {sp_dampratio:.4f}")
print(f"  Damped frequency  : {sp_dampfreq:.4f} rad/s")
print(f"  Time constant     : {sp_timeconst:.4f} s")

print("\nPhugoid Mode:")
print(f"  Eigenvalues       : {lp_eigs}")
print(f"  Characteristic eq.: {lp_coeffs}")
print(f"  Natural frequency : {lp_natfreq:.4f} rad/s")
print(f"  Damping ratio     : {lp_dampratio:.4f}")
print(f"  Damped frequency  : {lp_dampfreq:.4f} rad/s")
print(f"  Time constant     : {lp_timeconst:.4f} s")

print("\n============================================================")
print("REDUCED-ORDER APPROXIMATIONS")
print("============================================================")

print("\nShort-Period Approximation:")
print(f"  Eigenvalues       : {approx_sp_eigs}")
print(f"  Characteristic eq.: {approx_sp_coeffs}")
print(f"  Natural frequency : {approx_sp_natfreq:.4f} rad/s")
print(f"  Damping ratio     : {approx_sp_dampratio:.4f}")
print(f"  Damped frequency  : {approx_sp_dampfreq:.4f} rad/s")
print(f"  Time constant     : {approx_sp_timeconst:.4f} s")

print("\nPhugoid Approximation:")
print(f"  Eigenvalues       : {approx_lp_eigs}")
print(f"  Characteristic eq.: {approx_lp_coeffs}")
print(f"  Natural frequency : {approx_lp_natfreq:.4f} rad/s")
print(f"  Damping ratio     : {approx_lp_dampratio:.4f}")
print(f"  Damped frequency  : {approx_lp_dampfreq:.4f} rad/s")
print(f"  Time constant     : {approx_lp_timeconst:.4f} s")

print("\n============================================================")
print("COMPARISON")
print("============================================================")

print("\nShort-Period Mode Comparison:")
print(f"  Exact Eigenvalues              : {sp_eigs}")
print(f"  Approximate Eigenvalues        : {approx_sp_eigs}")
print(f"  Exact Natural Frequency        : {sp_natfreq:.4f} rad/s")
print(f"  Approximate Natural Frequency  : {approx_sp_natfreq:.4f} rad/s")
print(f"  Exact Damping Ratio            : {sp_dampratio:.4f}")
print(f"  Approximate Damping Ratio      : {approx_sp_dampratio:.4f}")
print(f"  Exact Damped Frequency         : {sp_dampfreq:.4f} rad/s")
print(f"  Approximate Damped Frequency   : {approx_sp_dampfreq:.4f} rad/s")
print(f"  Exact Time Constant            : {sp_timeconst:.4f} s")
print(f"  Approximate Time Constant      : {approx_sp_timeconst:.4f} s")

print("\nPhugoid Mode Comparison:")
print(f"  Exact Eigenvalues              : {lp_eigs}")
print(f"  Approximate Eigenvalues        : {approx_lp_eigs}")
print(f"  Exact Natural Frequency        : {lp_natfreq:.4f} rad/s")
print(f"  Approximate Natural Frequency  : {approx_lp_natfreq:.4f} rad/s")
print(f"  Exact Damping Ratio            : {lp_dampratio:.4f}")
print(f"  Approximate Damping Ratio      : {approx_lp_dampratio:.4f}")
print(f"  Exact Damped Frequency         : {lp_dampfreq:.4f} rad/s")
print(f"  Approximate Damped Frequency   : {approx_lp_dampfreq:.4f} rad/s")
print(f"  Exact Time Constant            : {lp_timeconst:.4f} s")
print(f"  Approximate Time Constant      : {approx_lp_timeconst:.4f} s")

# endregion