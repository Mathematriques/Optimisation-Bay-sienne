import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

# Loi sur deux copies de [0, 15]
# pas différent pour caputrer plus facilement les erreurs
x1 = np.linspace(0.0, 15.0, 700)
x2 = np.linspace(0.0, 15.0, 710)
x = np.r_[x1, x2]

dim_1 = len(x1)
dim_2 = len(x2)

μ1 = np.zeros_like(x1)
μ2 = np.zeros_like(x2)
μ = np.r_[μ1, μ2]


def noyau_simple(u, v, l=0.3, σ=5.0):
	diff = (u[..., np.newaxis] - v[np.newaxis, ...]) / l
	return σ**2 * np.exp(-0.5 * diff**2)


def noyau(u, v):
	u1, u2 = u[:dim_1], u[dim_1:]
	v1, v2 = v[:dim_2], v[dim_2:]
	# 0.9 = grande corrélation entre les deux copies de [0, 15]
	K11, K12 = noyau_simple(u1, v1), noyau_simple(u1, v2) * 0.9
	K21, K22 = noyau_simple(u2, v1) * 0.9, noyau_simple(u2, v2)
	return np.block([[K11, K12], [K21, K22]])


def fonction(x):
	return 3 + np.sin(1.5 * x) - 0.02 * x * x


# Densitée initiale et corrélation
Σ = noyau(x, x)
σ = np.sqrt(np.diag(Σ))
σ1, σ2 = σ[:dim_1], σ[dim_1:]

sample12 = rng.multivariate_normal(mean=μ, cov=Σ)
sample1, sample2 = sample12[:dim_1], sample12[dim_1:]

fig, axs = plt.subplots(2)

axs[0].fill_between(x1, μ1 - σ1, μ1 + σ1, alpha=0.2)
axs[0].plot(x1, sample1, color="tab:red", linewidth=0.5)
axs[0].plot(x1, μ1)

axs[1].fill_between(x2, μ2 - σ2, μ2 + σ2, alpha=0.2)
axs[1].plot(x2, sample2, color="tab:red", linewidth=0.5)
axs[1].plot(x2, μ2)

plt.show()


# Points de mesures sur chaque copie
xm1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
xm2 = np.array([8.5, 9.0])
xm = np.r_[xm1, xm2]

ym1 = fonction(xm1)
ym2 = fonction(xm2)
ym = np.r_[ym1, ym2]  # pas de bruit

# Densitée mise à jour par des mesures
μm = np.r_[np.interp(xm1, x1, μ1), np.interp(xm2, x2, μ2)]
Σm = noyau(xm, x)
Σmm = noyau(xm, xm) + 0.1 * np.eye(len(xm))
Σmm_inv = np.linalg.inv(Σmm)
μ_post = μ + Σm.T @ Σmm_inv @ (ym - μm)
Σ_post = Σ - Σm.T @ Σmm_inv @ Σm
σ_post = np.sqrt(np.diag(Σ_post))

μ1_post, μ2_post = μ_post[:dim_1], μ_post[dim_1:]
σ1_post, σ2_post = σ_post[:dim_1], σ_post[dim_1:]

fig, axs = plt.subplots(2)

axs[0].fill_between(x1, μ1_post - σ1_post, μ1_post + σ1_post, alpha=0.2)
axs[0].plot(x1, μ1_post)
axs[0].plot(xm1, ym1, color="tab:brown", marker="o", linestyle="None", markersize=3)

axs[1].fill_between(x2, μ2_post - σ2_post, μ2_post + σ2_post, alpha=0.2)
axs[1].plot(x2, μ2_post)
axs[1].plot(xm2, ym2, color="tab:brown", marker="o", linestyle="None", markersize=3)

plt.show()
