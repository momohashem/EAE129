import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

np.set_printoptions(suppress=True, precision=6)

# ============================================================
# EAE 129 Final Project
# Fighter Aircraft Longitudinal Dynamics
# Sea level, Mach 0.257
# ============================================================

# -----------------------------
# Given flight condition / data
# -----------------------------
u0 = 286.92          # trim flight speed [ft/s]
g = 32.174           # gravitational acceleration [ft/s^2]
cbar = 9.55          # mean aerodynamic chord [ft]
Iy = 58611.0         # pitch-axis inertia [slug-ft^2]
W = 16300.0          # weight [lb]
m = W / g            # mass [slug]
rho = 0.002377       # sea-level air density [slug/ft^3]

Q = 0.5 * rho * u0**2  # dynamic pressure [lb/ft^2]

S = 196.1            # wing reference area [ft^2]
S_t = 50.0           # tail area [ft^2]
tau_e = 0.5          # elevator efficiency factor
eta_t = 0.95         # dynamic pressure ratio at tail

# -----------------------------
# Aerodynamic coefficients
# -----------------------------
C_D0 = 0.263
C_Da = 0.45
C_Du = 0.0

C_L0 = 0.735
C_Lu = 0.0
C_La = 3.44
C_La_t = 3.0

C_ma = -0.64
C_mu = 0.0
C_mq = -5.8
C_ma_dot = -1.6
C_mde = -1.46

# Elevator force derivative
C_Zde = -C_La_t * tau_e * eta_t * (S_t / S)

# -----------------------------
# Dimensional stability derivatives
# -----------------------------
X_u = -(C_Du + 2 * C_D0) * (Q * S) / (m * u0)
X_w = -((C_Da - C_L0) * Q * S) / (m * u0)
X_de = 0.0

Z_u = -(C_Lu + 2 * C_L0) * (Q * S) / (m * u0)
Z_w = -((C_La + C_D0) * Q * S) / (m * u0)
Z_de = -C_Zde * Q * S / m

M_u = C_mu * (Q * S * cbar) / (Iy * u0)
M_w = C_ma * (Q * S * cbar) / (Iy * u0)
M_w_dot = C_ma_dot * (cbar / (2 * u0)) * (Q * S * cbar) / (Iy * u0)
M_q = C_mq * (Q * S * cbar**2) / (Iy * 2 * u0)
M_de = C_mde * (Q * S * cbar) / Iy

# -----------------------------
# Full fourth-order model
# -----------------------------
A = np.array([
    [X_u, X_w, 0.0, -g],
    [Z_u, Z_w, u0, 0.0],
    [M_u + M_w_dot * Z_u, M_w + M_w_dot * Z_w, M_q + M_w_dot * u0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
], dtype=float)

B = np.array([
    [X_de],
    [Z_de],
    [M_de + M_w_dot * Z_de],
    [0.0]
], dtype=float)

print("A matrix:")
print(A)
print("\nB matrix:")
print(B)

# -----------------------------
# Modal analysis
# -----------------------------
def modal_properties(pair):
    sigma = np.real(pair[0])
    wd = abs(np.imag(pair[0]))
    wn = np.sqrt(sigma**2 + wd**2)
    zeta = -sigma / wn
    tau = -1.0 / sigma
    return wn, wd, zeta, tau

eigvals = np.linalg.eigvals(A)
eigvals_sorted = sorted(eigvals, key=lambda z: abs(np.imag(z)), reverse=True)

sp_eigs = np.array([eigvals_sorted[0], eigvals_sorted[1]])
ph_eigs = np.array([eigvals_sorted[2], eigvals_sorted[3]])

sp_wn, sp_wd, sp_zeta, sp_tau = modal_properties(sp_eigs)
ph_wn, ph_wd, ph_zeta, ph_tau = modal_properties(ph_eigs)

print("\nShort-period eigenvalues:", sp_eigs)
print(f"Short-period: wn={sp_wn:.4f}, wd={sp_wd:.4f}, zeta={sp_zeta:.4f}, tau={sp_tau:.4f}")

print("\nPhugoid eigenvalues:", ph_eigs)
print(f"Phugoid: wn={ph_wn:.4f}, wd={ph_wd:.4f}, zeta={ph_zeta:.4f}, tau={ph_tau:.4f}")

# -----------------------------
# State-space system
# -----------------------------
C = np.eye(4)
D = np.zeros((4, 1))
sys = sig.StateSpace(A, B, C, D)

time = np.linspace(0, 200, 1000)
labels = [
    r"$\Delta u$, Forward Velocity Perturbation",
    r"$\Delta w$, Vertical Velocity Perturbation",
    r"$\Delta q$, Pitch Rate Perturbation",
    r"$\Delta \theta$, Pitch Angle Perturbation",
]
units = ["ft/s", "ft/s", "deg/s", "deg"]
colors = ["red", "orange", "blue", "green"]

# -----------------------------
# Free response
# -----------------------------
u_free = np.zeros_like(time)
x0_free = np.array([0.0, 0.0, 0.0, 0.1])  # initial pitch-angle perturbation [rad]
t_free, y_free, x_free = sig.lsim(sys, U=u_free, T=time, X0=x0_free)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Free Response of Fighter-Aircraft Longitudinal System")
for i, ax in enumerate(axes.flat):
    if i < 2:
        ax.plot(time, x_free[:, i], color=colors[i], linewidth=1.5)
    else:
        ax.plot(time, np.rad2deg(x_free[:, i]), color=colors[i], linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Response [{units[i]}]")
    ax.set_title(labels[i])
    ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("free_response_fighter.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Step response
# -----------------------------
u_step = np.ones_like(time)
x0_step = np.zeros(4)
t_step, y_step, x_step = sig.lsim(sys, U=u_step, T=time, X0=x0_step)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Step Response of Fighter-Aircraft Longitudinal System")
for i, ax in enumerate(axes.flat):
    if i < 2:
        ax.plot(time, x_step[:, i], color=colors[i], linewidth=1.5)
    else:
        ax.plot(time, np.rad2deg(x_step[:, i]), color=colors[i], linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Response [{units[i]}]")
    ax.set_title(labels[i])
    ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("step_response_fighter.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Eigenvalue location plot
# -----------------------------
fig = plt.figure(figsize=(8, 6))
plt.scatter(np.real(sp_eigs), np.imag(sp_eigs), marker='x', s=100, c='black', label='Short-Period')
plt.scatter(np.real(ph_eigs), np.imag(ph_eigs), marker='o', s=70, facecolors='none', edgecolors='blue', label='Phugoid')
plt.axhline(0, color='black', linewidth=0.7)
plt.axvline(0, color='black', linewidth=0.7)
plt.xlabel("Real Axis")
plt.ylabel("Imaginary Axis")
plt.title("Eigenvalue Locations of Fighter-Aircraft Longitudinal System")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("eigenvalue_locations_fighter.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Reduced-order approximations
# -----------------------------
A_sp = np.array([
    [Z_w, u0],
    [M_w + M_w_dot * Z_w, M_q + M_w_dot * u0]
], dtype=float)

A_ph = np.array([
    [X_u, -g],
    [-Z_u / u0, 0.0]
], dtype=float)

print("\nA_sp:")
print(A_sp)
print("eig(A_sp):", np.linalg.eigvals(A_sp))

print("\nA_ph:")
print(A_ph)
print("eig(A_ph):", np.linalg.eigvals(A_ph))
