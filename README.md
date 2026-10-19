# Bayesian Estimation of a New Keynesian DSGE Model with Sequential Monte Carlo

**A Python-based replication study of the New Keynesian DSGE model and Bayesian Sequential Monte Carlo (SMC) estimation approach discussed in Herbst & Schorfheide (2016).**

This project combines **macroeconomic modeling**, **state-space methods**, **Kalman filtering**, and **particle-based Bayesian inference** to estimate structural parameters from US macroeconomic data. The resulting posterior estimates are compared with published reference results.

**At a glance:**

| | |
|---|---|
| **Model** | Small New Keynesian Dynamic Stochastic General Equilibrium (DSGE) model |
| **Estimation** | Bayesian Sequential Monte Carlo with likelihood tempering |
| **Implementation** | Python |
| **Estimated parameters** | 13 |
| **SMC configuration** | 5,000 particles; 400 tempering stages |
| **Sample** | US quarterly data, 1983 Q1–2002 Q4 |
| **Observed series** | Output gap, inflation, nominal interest rate (as labeled in the report) |

> **Research context:** This is an independent replication exercise based on Herbst & Schorfheide (2016), not an official implementation accompanying the book.

## Research objective

The aim is to approximate the posterior distribution of structural DSGE parameters, conditional on observed macroeconomic data:

$$
p(\theta \mid y_{1:T}) \propto p(\theta)\,p(y_{1:T}\mid\theta).
$$

Evaluating the likelihood for a candidate parameter vector requires solving the DSGE model and running a Kalman filter. Sequential Monte Carlo provides a particle-based approach to exploring the posterior through a series of tempered distributions.

## Economic model

The report implements a small **New Keynesian DSGE model** featuring:

- An **intertemporal IS (Euler) equation** governing aggregate demand.
- A **New Keynesian Phillips curve** linking inflation and economic activity.
- A **Taylor-style monetary policy rule** with interest-rate smoothing.
- **Autoregressive structural shocks** affecting the model's dynamics.

The parameter vector contains 13 structural, persistence, and volatility parameters:

$$
\theta = (\tau, \kappa, \psi_1, \psi_2, r_A, \pi_A, \gamma_Q,
\rho_R, \rho_g, \rho_z, \sigma_R, \sigma_g, \sigma_z).
$$

The solved model is expressed in state-space form, enabling likelihood evaluation using a **Kalman filter**.

## Estimation workflow

The implementation described in the report follows this sequence:

1. **Initialize particles:** Draw parameter vectors from their prior distributions.
2. **Solve the DSGE system:** Use a `gensys`-style solution to obtain a linear rational-expectations state transition.
3. **Construct the state-space representation:** Map latent states to observed macroeconomic variables and specify structural and measurement uncertainty.
4. **Evaluate likelihoods:** Apply the Kalman filter to compute the data likelihood for each parameter vector.
5. **Temper the likelihood:** Move gradually from the prior toward the posterior via intermediate distributions.
6. **Reweight and resample:** Update particle weights, calculate effective sample size (ESS), and use **systematic resampling** when ESS falls below its threshold.
7. **Mutate particles:** Apply **Random-Walk Metropolis–Hastings (RWMH)** steps, using a proposal covariance informed by the particle cloud.
8. **Summarize the posterior:** Calculate posterior means, credible intervals, diagnostic plots, and parameter relationships.

At tempering stage $t$, the target distribution is:

$$
\pi_{\phi_t}(\theta) \propto p(\theta)\,p(y_{1:T}\mid\theta)^{\phi_t},
\qquad 0=\phi_0<\cdots<\phi_K=1.
$$

### Methodology demonstrated

`Python` · `Bayesian econometrics` · `DSGE modeling` · `Sequential Monte Carlo` · `Kalman filtering` · `State-space models` · `Metropolis–Hastings` · `Posterior diagnostics`

## Results

The report presents an SMC run using **5,000 particles** and **400 tempering stages**. The posterior estimates are compared with the reference values in **Herbst & Schorfheide (2016), Table 4.1**.

### Posterior estimates vs. published reference

The interval column reports the **5th–95th percentile range** of the estimated posterior distribution.

| Parameter | SMC posterior mean | SMC 5%–95% interval | Herbst & Schorfheide mean |
|:--|--:|:--:|--:|
| $\tau$ | 2.3606 | [1.5860, 3.2720] | 2.83 |
| $\kappa$ | 0.8465 | [0.6102, 0.9886] | 0.78 |
| $\psi_1$ | 1.9211 | [1.5551, 2.3035] | 1.80 |
| $\psi_2$ | 0.5894 | [0.2042, 1.1155] | 0.63 |
| $r_A$ | 0.4417 | [0.0510, 0.9379] | 0.42 |
| $\pi_A$ | 3.3814 | [2.7574, 4.0092] | 3.30 |
| $\gamma_Q$ | 0.5988 | [0.3681, 0.8195] | 0.52 |
| $\rho_R$ | 0.7727 | [0.7106, 0.8280] | 0.77 |
| $\rho_g$ | 0.9779 | [0.9456, 0.9984] | 0.98 |
| $\rho_z$ | 0.9240 | [0.8831, 0.9634] | 0.88 |
| $\sigma_R$ | 0.2180 | [0.1788, 0.2661] | 0.22 |
| $\sigma_g$ | 0.6551 | [0.5669, 0.7558] | 0.71 |
| $\sigma_z$ | 0.1997 | [0.1651, 0.2401] | 0.31 |

**Interpretation:** Several estimates are close to the published reference means (for example, $\rho_R$, $\rho_g$, and $\sigma_R$). Differences remain for some parameters, notably $\tau$ and $\sigma_z$. The comparison is presented as a replication diagnostic, **not** as evidence of exact numerical agreement.

### Prior and posterior distributions

The comparison below illustrates how the observed data update the prior distribution across model parameters.

![Prior and posterior distributions for the 13 parameters](figures/prior_vs_posterior.png)

### SMC diagnostics

The effective sample size shows when the particle distribution becomes concentrated and when resampling occurs. The mutation acceptance rate summarizes the behavior of the RWMH step across tempering stages.

| Effective sample size (ESS) | RWMH mutation acceptance rate |
|:--:|:--:|
| ![ESS by stage](figures/effective_sample_size.png) | ![Mutation acceptance by stage](figures/mutation_acceptance_rate.png) |

<details>
<summary><strong>Additional posterior diagnostics and historical decomposition</strong></summary>

**Parameter trace plots across SMC stages**

![SMC parameter trace plots](figures/smc_trace_plots.png)

**Posterior pairwise joint distributions**

![Pairwise posterior joint densities](figures/posterior_joint_distributions.png)

**Posterior parameter correlations**

![Posterior correlation matrix](figures/posterior_correlations.png)

**Historical decomposition of the three observables**

![Historical decomposition](figures/historical_decomposition.png)

</details>

## Technical documentation

The [technical overview (PDF)](docs/DSGE_SMC_overview.pdf) explains the model equations, likelihood construction, SMC algorithm, and posterior diagnostics in greater detail. It also includes the corresponding figures and comparison with Herbst & Schorfheide (2016).

### Implementation components described in the report

| Component | Purpose |
|---|---|
| `model_solution(theta)` | Solve the linear rational-expectations DSGE model |
| `sysmat(T1, T0, theta)` | Build the state-space matrices |
| `kalman_loglik(...)` | Evaluate the log-likelihood of the observed data |
| `log_prior(theta)` | Evaluate the parameter prior density |
| `log_posterior_tempered(theta, phi)` | Combine prior and tempered likelihood |
| SMC loop | Reweight, monitor ESS, resample, and mutate particles |

**Reproducing the computation:** This README is based on the accompanying research overview. The PDF documents the estimation pipeline and results, but it does **not** contain complete executable source files, a dependency list, or data-loading instructions. Upload the Python implementation and any shareable data separately; then add the exact installation and run commands here.

## Reference

Herbst, E. P., & Schorfheide, F. (2016). *Bayesian Estimation of DSGE Models*. Princeton University Press.

---

*Report author: Georg Mayrpeter.*
