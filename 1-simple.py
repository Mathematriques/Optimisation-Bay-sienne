import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng()

# /!\ ça devrait être n échantillons dans R => (n, 1)
# Mais c'est plus pratique pour les graphiques simples
x = np.arange(0.0, 10.0, 0.01)
μ = 0.0 * x  # prior

# Covariance entre chaque éléments
def noyau(u, v, l=0.3, σ=1.0):
	# u, v => (n, 1) −> (n,)
	diff = (u[..., np.newaxis] - v[np.newaxis, ...]) / l  # (n, n, 1) -> (n, n)
	return σ**2 * np.exp(-0.5 * diff**2)


def fonction(x):
	return 3 + np.sin(1.5 * x) - 0.02 * x * x


# Mesures
xm = np.arange(0.1, 9.5, 0.5)
z = np.zeros_like(xm)
C = np.diag(xm / x.max()) ** 2  # cov du bruit hétérosédastique
ym = fonction(xm) + rng.multivariate_normal(mean=z, cov=C)

# −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−
# variable latente Z = F(x), mesure Y = Z + ε
# P(Z = z | Y = y)
#       = P(Z = z, Y = y) / P(Y = y)
#       = P(Y = y | Z = z) × P(Z = z) / P(Y = y)
#       = P(ε = y - z) × P(Z = z) / P(Y = y)
# −−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−−

# F = N [μ | Σ  Σm.T]
# Z     [m | Σm Σmm ]

fig, axs = plt.subplots(2)

# Densitée initiale
Σ = noyau(x, x)
σ = np.sqrt(np.diag(Σ))

axs[0].grid()
axs[0].set_title("prior")
axs[0].fill_between(x, μ - σ, μ + σ, alpha=0.2)
axs[0].plot(x, μ)
axs[0].plot(x, rng.multivariate_normal(mean=μ, cov=Σ), color="tab:red", linewidth=0.5)
axs[0].plot(x, rng.multivariate_normal(mean=μ, cov=Σ), color="tab:red", linewidth=0.5)
axs[0].plot(x, rng.multivariate_normal(mean=μ, cov=Σ), color="tab:red", linewidth=0.5)
axs[0].plot(x, fonction(x), color="k", linestyle="--")

# Densitée mise à jour par des mesures
μm = np.interp(xm, x, μ)
Σm = noyau(xm, x)
Σmm = noyau(xm, xm) + C
Σmm_inv = np.linalg.inv(Σmm)
μ_post = μ + Σm.T @ Σmm_inv @ (ym - μm)
Σ_post = Σ - Σm.T @ Σmm_inv @ Σm
σ_post = np.sqrt(np.diag(Σ_post))

axs[1].grid()
axs[1].set_title("posterior")
axs[1].plot(xm, ym, color="tab:brown", marker="o", linestyle="None", markersize=3)
axs[1].fill_between(x, μ_post - σ_post, μ_post + σ_post, alpha=0.2)
axs[1].plot(x, μ_post)
axs[1].plot(x, rng.multivariate_normal(mean=μ_post, cov=Σ_post), color="tab:red", linewidth=0.5)
axs[1].plot(x, rng.multivariate_normal(mean=μ_post, cov=Σ_post), color="tab:red", linewidth=0.5)
axs[1].plot(x, rng.multivariate_normal(mean=μ_post, cov=Σ_post), color="tab:red", linewidth=0.5)
axs[1].plot(x, fonction(x), color="k", linestyle="--")

plt.show()
