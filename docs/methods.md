# Kinematic methods and interpretation

This package measures empirical velocity observables in a documented particle snapshot. Its public example is a prescribed toy distribution, not an evolved galaxy, an equilibrium solution, or a reproduction of an Auriga analysis. The LMC/direct-detection application provides scientific motivation; actual simulation interpretation requires independently established input provenance and geometry. See [Smith-Orlik et al.](https://arxiv.org/abs/2302.04281).

## Spatial selection

Positions must be physical, galaxy-centered Cartesian coordinates in kpc. The default selection is the inclusive radial shell `6 <= r <= 10`. With `--cone-axis ax ay az`, it additionally requires

$$\hat{\boldsymbol r}\cdot\hat{\boldsymbol a}\geq\cos\alpha,$$

where `--half-angle` gives $\alpha$ in degrees. The axis is normalized internally, and the boundary allows eight float64 machine epsilons in cosine for roundoff. Equal inner and outer radii are permitted and select particles exactly on that radius. Origins are excluded because their spherical direction is undefined. A shell or cone is a geometric selection; it does not by itself establish a solar-neighborhood analogue.

## Host-frame spherical velocities and anisotropy

The spherical basis uses polar angle $\theta$ from $+z$ and azimuth $\phi$ from $+x$ toward $+y$:

$$\boldsymbol e_r=\boldsymbol r/r,\qquad
\boldsymbol e_\phi=(-\sin\phi,\cos\phi,0),\qquad
\boldsymbol e_\theta=\boldsymbol e_\phi\times\boldsymbol e_r.$$

Projection onto this orthonormal basis preserves each particle's speed. At the poles the code chooses $\phi=0$; individual tangential components there depend on that convention. Anisotropy is calculated **before** any observer boost. For each component, with its own selected total weight $W_c$,

$$\bar v_{k,c}=W_c^{-1}\sum_{i\in c}w_i v_{k,i},\qquad
Q_{k,c}=\sum_{i\in c}w_i(v_{k,i}-\bar v_{k,c})^2,$$

$$\widehat\beta_c=1-\frac{Q_{\theta,c}+Q_{\phi,c}}{2Q_{r,c}}.$$

Centering removes the component's mean spherical streaming motion. The common variance normalization cancels in the ratio; the streaming pipeline merges centered weighted moments across chunks. Fewer than two particles or zero radial dispersion gives an undefined result (`null` in JSON and an empty CSV field). The standalone unweighted `anisotropy` function raises on undefined input. This finite-sample ratio is not generally unbiased, and $\beta=0$ alone does not establish isotropy. The pooled shell estimate can include spatial variations in streaming and dispersion; it is not a resolved $\beta(r)$ profile.

## Observer-frame speeds

For phase $p$, particle $i$ has observer-frame speed

$$s_i(p)=\left|\boldsymbol v_i-\boldsymbol v_{\rm obs}(p)\right|.$$

Local Galactic axes are $U$ toward the Galactic center, $V$ along Galactic rotation, and $W$ toward the north Galactic pole. The implemented approximation is

$$\boldsymbol v_{\rm obs,local}(p)=(11.1,\ v_c+12.24,\ 7.25)
+29.8\,[\boldsymbol e_1\cos\lambda+\boldsymbol e_2\sin\lambda]
\quad\mathrm{km\,s^{-1}},$$

$$\lambda=2\pi(p-0.218),\quad
\boldsymbol e_1=(0.9931,0.1170,-0.01032),\quad
\boldsymbol e_2=(-0.067,0.4927,-0.8676).$$

Phase is fractional year measured from January 1, with periodic wrapping. The orbital velocity basis, signs, and phase convention follow [Savage, Freese & Gondolo, equations 13–15](https://arxiv.org/abs/astro-ph/0607121); these velocity vectors must not be interchanged with orbital-position basis vectors. The peculiar velocity follows [Schönrich, Binney & Dehnen](https://arxiv.org/abs/0912.3693). The adopted rounded coefficients and circular orbit are an approximation, not a precision ephemeris; daily rotation and gravitational focusing are not modeled. The circular speed is a user parameter, defaulting to 220 km/s.

For an observer basis matrix $B$, each row gives a local unit axis in processed simulation coordinates. Row-vector conversion is $\boldsymbol v_{\rm obs}=\boldsymbol v_{\rm obs,local}B$. The code requires a right-handed orthonormal matrix and defaults to identity. `--basis` supplies its nine entries in row-major order. Choosing this basis is a physical input, not an orientation inferred from the particles.

## Annual observables and normalization

The calculation samples $T$ equally spaced phases $p_j=j/T$, $j=0,\ldots,T-1$; the default is $T=12$. Their arithmetic mean approximates an annual mean. This is an observer-motion calculation on one fixed snapshot, not a sequence of evolving galaxy snapshots. Increase the phase count to assess convergence for the application.

Let $W=\sum_{i\in\mathrm{selected}}w_i$ include **both** components. For component $c$ and a speed bin $b$ of width $\Delta s_b$,

$$\bar f_{c,b}=\frac{1}{TW\Delta s_b}
\sum_j\sum_{i\in c}w_i\,\mathbf1[s_i(p_j)\in b].$$

The strict tail and mean inverse speed are evaluated directly from particle speeds:

$$\bar P_c(>v_{\min})=\frac{1}{TW}\sum_j\sum_{i\in c}
w_i\,\mathbf1[s_i(p_j)>v_{\min}],$$

$$\bar\eta_c(v_{\min})=\frac{1}{TW}\sum_j
\sum_{\substack{i\in c\\s_i(p_j)>v_{\min}}}\frac{w_i}{s_i(p_j)}.$$

The density and inverse-speed observable have units $(\mathrm{km\,s^{-1}})^{-1}=\mathrm{s\,km^{-1}}$; tail probability is dimensionless. MW and LMC contributions add to the total because they use the same denominator. They are not independently normalized conditional distributions. The tagged fraction after selection need not equal the generator's whole-snapshot fraction.

Each phase contributes its own histogram, tail, and inverse-speed sum **before averaging**. Averaging a particle's speed first changes these nonlinear observables. Thresholds use strict `>`: a speed exactly at the threshold contributes nothing. At $v_{\min}=0$, zero-speed particles remain in $W$ but contribute no inverse-speed term; no arbitrary speed floor is introduced. This is an explicit empirical convention for discrete samples.

Bins follow NumPy's left-inclusive, right-exclusive convention, except that the final right edge is included. Speeds above `--speed-max` remain in the denominator and are reported as `histogram_overflow_fraction`; the displayed histogram is never renormalized to hide overflow. Tail and inverse-speed estimates still include those particles. `--speed-max` must be a positive integer multiple of `--bin-width`.

Mean inverse speed is a useful kinematic ingredient in direct-detection calculations; a predicted detector rate additionally needs density, particle and nuclear physics, and detector response. Those are outside this package. For context, see [Freese, Lisanti & Savage](https://arxiv.org/abs/1209.3339).

## Synthetic demonstration

`write_synthetic` uses NumPy's local seeded generator and assigns an exact tagged count, rounding `N * lmc_fraction` to the nearest integer with half ties upward. Positions are isotropic and uniform in volume between 4 and 14 kpc, independently of the component. Velocity components are independent Gaussians:

| Component | Mean `(vx, vy, vz)` (km/s) | Component standard deviations (km/s) |
| --- | --- | --- |
| Host/MW | `(0, 0, 0)` | `(155, 155, 155)` |
| Tagged/LMC | `(0, -300, 80)` | `(70, 50, 50)` |

The default uses 20,000 particles, seed 42, a tagged fraction of 0.05, and unit weights. All parameters and the generator source hash are saved in the snapshot. These prescribed distributions have no gravitational potential, dynamical evolution, equilibrium constraint, or escape-speed truncation. A visually prominent tagged tail illustrates mixture weighting and observer boosts; it does not measure the real LMC's contribution.

## Numerical and presentation scope

Input is read in chunks. Kernels retain all phase speeds for the current chunk, so memory includes an array of shape `(phase_count, selected_chunk_size)` rather than the entire snapshot. Tail accumulation sorts speeds per phase and component; it does not approximate the tail using histogram bins.

The figure renderer reads saved tables, checks component additivity, and exports PDF, SVG, and 600 dpi PNG with a descriptive caption. Logarithmic panels omit zero values rather than replacing them by artificial positive floors; the tables retain exact zeros. No uncertainty bands are supplied: annual phases reuse the same particles and are not independent realizations. Sampling uncertainty, simulation convergence, halo-to-halo variation, and observational systematics require separate analysis.
