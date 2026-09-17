# Changelog

## 0.1.0

Unreleased, and it runs. heddle is written in twill's `mode systems`, which
landed in twill 1.6; `twill test tests` passes all 8 suites under twill 1.7.1.
This entry used to say nothing here had executed, which is no longer true.
`README.md`'s State table says which piece each suite covers, and which two
pieces still have no test.

Changed, on moving the pin from twill 1.12.0 to 1.13.0:

- A pin-currency bump, not a behaviour change. `spool.toml`, the CI workflow
  and the README's install line move from 1.12.0 to 1.13.0. The suites pass on
  1.13.0, with the one documented arm64 tolerance difference in `nuts_test`
  that `docs/needs.md` records and CI does not hit on linux.

Changed, on moving the pin from twill 1.7.1 to 1.12.0:

- The three transforms that returned a value and its log Jacobian return a
  tuple, `(Tensor, Tensor)`, and `Simplex`, `Ordered` and `CholFactor` are
  gone. `docs/needs.md` entry 19 called them "tuples with names", the same two
  fields declared three times because a function could return one value;
  twill 1.12 returns two. A caller writes `let (y, log_jac) = tr.simplex(x, k)`
  and cannot read a field that is not there. `DrawResult`, `Subtree`,
  `AdviResult` and `LogDensity` stay, for the reason the entry gives: they have
  more than two parts, or parts a reader wants by name.
- The assertions are `std/test`. twill 1.11 ships the ones every satellite
  harness copied by hand, so `tests/harness.tw` is deleted and every suite
  imports `std/test` as `t`. `near_grad` was heddle's own and lives in
  `tests/dist_test.tw`, the one suite that calls it. `report` returns the
  status instead of calling `exit`, and prints its summary in the shape
  `twill test` reads, so the runner shows the counts beside each file, where
  before it showed none.
- `diag.sorted_copy` is twill 1.9's `sort`, which returns a new array and
  takes no cutoff constant. The insertion sort it replaces was kept to avoid
  `std/stats.tw`'s constant, which is entry 27, now closed. The comparison is
  passed, because a coordinate read out of a tensor is a rank-0 tensor at
  runtime and `sort`'s own order refuses it; the entry records the finding.
- The recursion limit heddle's entry 9 asked to have stated is stated: twill
  1.12 refuses a call nested more than 10,000 deep with a twill error naming
  the function, and NUTS at `max_depth` 10 uses eleven frames.

Written:

- A model as `fn(Tensor) -> Tensor`, with the gradient of the log posterior
  supplied by twill's `grad`. No tracing layer and no second representation of
  the user's program.
- Distributions with log density, sampler, and the reparameterised form where
  one exists: normal, half-normal, lognormal, student t, exponential, gamma,
  beta, Dirichlet, categorical, multinomial, and a multivariate normal
  parameterised by its Cholesky factor.
- A differentiable `lgamma` over tensors, because a shape parameter is a
  parameter and `std/stats.tw`'s `F64` version is a constant to `grad`.
- Transforms for constrained parameters with their log Jacobians derived in the
  source: log, shifted log, logit, interval, stick-breaking simplex, ordered
  vectors, and a positive-diagonal Cholesky factor.
- Random walk Metropolis with a Robbins-Monro proposal scale.
- Static Hamiltonian Monte Carlo, as the reference the tree is checked against.
- NUTS with multinomial sampling, the generalised U-turn criterion and its two
  cross-boundary checks, dual averaging, and diagonal mass matrix adaptation on
  Stan's three-stage warmup schedule.
- Mean-field ADVI using the reparameterisation trick.
- Split rank-normalised R-hat, effective sample size by Geyer's initial
  monotone positive sequence, Monte Carlo standard error, divergence counts and
  tree depth saturation, with warnings written as sentences.
- Tests in the harness spool and loom share, and `examples/eight_schools.tw` in
  both parameterisations.

- A Laplace approximation: Newton to the mode, the Gaussian from `hessian`, the
  log evidence, sampling and the delta method.

Not written, and why: see the last section of `README.md`.

`docs/needs.md` is the list of language features this source asked twill for. It
now records which arrived and which are still open, and the open ones are 19,
22, 23, 25, 26 and 27.
