<p align="center">
  <img alt="twill" src="https://raw.githubusercontent.com/twill-lang/twill/main/assets/twill-mark.png" width="120">
</p>

<h1 align="center">heddle</h1>

<p align="center">
  <b>Probabilistic programming and Bayesian inference for <a href="https://github.com/twill-lang/twill">twill</a>.</b><br>
  Written in twill.
</p>

<p align="center">
  <img alt="heddle" src="https://img.shields.io/badge/heddle-v0.1.0-7FE3C4?style=flat-square&labelColor=12332C">
  <img alt="written in twill" src="https://img.shields.io/badge/written%20in-twill-D2F0E4?style=flat-square&labelColor=12332C">
  <img alt="status" src="https://img.shields.io/badge/tests-passing-4FB79B?style=flat-square&labelColor=12332C">
  <img alt="MIT" src="https://img.shields.io/badge/license-MIT-D2F0E4?style=flat-square&labelColor=12332C">
</p>

---

## It runs

`heddle` is written in twill, in `.tw` files, using `mode systems`. That subset
did not exist when this library was written, so for a long time none of the code
here executed and this section said so. twill 1.6 is the release that closed it:
the 8 test suites under `tests/` pass, and CI runs them against a released
twill on every push rather than gating on the prose in this file.

**Seven of the eight pass anywhere; the eighth passes where CI runs it.**
`tests/nuts_test.tw` is 27/27 on linux/amd64 (which is what CI is) and 26/27
on arm64, where the half-normal posterior mean comes back 0.8299 against a
tolerance of 0.7979 ± 0.03. Go's `math.Exp` differs by one ULP between the two
architectures and a seeded NUTS run amplifies it: one trajectory diverges early,
the sampler takes a different path, and the answer moves by a thousand times the
input difference. The sampler is behaving correctly and so is the test; what is
wrong is a tolerance that assumes an architecture. `docs/needs.md` has the
measurement and twill's `docs/CORRECTNESS.md` section 4 has the ULP.

```bash
twill test tests
```

You need twill 1.12.0 or newer.

### That command takes about sixteen minutes. It is not hung.

This is the thing about heddle worth knowing before anything else. `twill test
tests` prints nothing about a suite until that suite finishes, and two of the
eight do real Monte Carlo work: `tests/nuts_test.tw` runs NUTS on five targets
and `tests/diag_test.tw` builds long chains to check R-hat and ESS against. A
first-time reader watches a silent terminal for a quarter of an hour and
concludes the compiler has locked up. It has not.

Measured on Windows 11 against the twill 1.7.1 release binary, `time` around the
whole run:

| Invocation | Suites | Wall clock |
| --- | --- | --- |
| `twill test tests` | 8 | 16m01s |
| `twill test tests/advi_test.tw tests/dist_test.tw tests/hmc_test.tw tests/laplace_test.tw tests/model_test.tw tests/transform_test.tw` | 6 | 10.2s |
| `twill test tests --filter diag` | 1 | 1m16.8s |
| `twill test tests --filter nuts` | 1 | 11m56.9s |
| `twill test tests --filter laplace` | 1 | 0.2s |

The six fast suites are the second row and they are the ones to run while you
work: ten seconds for the whole of the library except the two suites above.
`twill test` takes explicit file paths, which is how that row is written.

`--filter <substring>` runs the suites whose path contains the substring, and it
takes one substring: passing `--filter` twice keeps only the last one, so it
selects a suite rather than a set. Explicit paths are the way to name a set.

Nothing here is optimised and nothing here is parallel. `src/nuts.tw`'s
`run_chains` runs its chains one after another because twill has no way to run
them at once, which is `docs/needs.md` entry 26 and the largest performance item
in this repository. There is also no progress output, which is entry 25, and
that entry used to blame the language for having no clock. It has one, and has
had since 1.6.0-rc1; what is missing is heddle deciding what to print and how
often.

`docs/needs.md` is still worth reading -- it is the list of what this library
asked the language for, and it now records which of those arrived and which are
still open.

## Why this package is the argument for twill

Every probabilistic programming system needs one thing: the gradient of the log
posterior. Hamiltonian Monte Carlo is built on it, variational inference is
built on it, and neither is possible without it.

Systems built on numerical frameworks that do not differentiate arbitrary code
obtain that gradient by constructing a second representation of the user's
model. A trace. A graph. A tape recorded through overloaded operators. A set of
`sample` and `observe` effects interpreted by a handler. That second
representation is the framework's real language, and it is where the friction
lives:

- Control flow the tracer cannot see through is silently mishandled.
- A model that calls a library function the tracer does not recognise stops
  differentiating and reports a zero, which looks like a parameter with no
  effect rather than like an error.
- A parameter used twice is recorded twice unless the framework deduplicates.
- The user learns two languages: the host, and the subset the tracer accepts.

twill differentiates twill. `grad(logp)` is the gradient of the log posterior
for any `logp` a user can write, including one with a `while` loop, one that
calls `std/nn`, and one that calls another package. There is nothing to trace
because there is no second representation.

The consequence is that heddle is small. A model is a function:

```rust
fn logp(q: Tensor) -> Tensor
```

One argument, the unconstrained parameter vector. One result, the log of the
unnormalised posterior. That is the entire interface. There is no model object,
no registry, no graph, no DSL. Composition of models is composition of
functions. Reparameterising a model is writing a function that calls the first
one.

The reparameterisation trick, which is the whole of variational inference, is
four lines here rather than a subsystem:

```rust
let objective = fn(m: Tensor, w: Tensor) -> Tensor {
  let theta: Tensor = m + exp(w) * eps
  logp(theta) + entropy(w, d)
}
let g = grads(objective)(mu_t, om_t)
```

`eps` is drawn outside, so the randomness is not on the path the gradient takes.
`grads` returns the gradient with respect to both variational parameters. No
part of that chain was written by hand, and no distribution had to know its own
reparameterisation for it to work.

## State

Every "runs" below names the test or example that exercises it. A piece with
no test under `tests/` and no caller in `examples/` says so instead, because
"it compiles" is not a claim about behaviour.

| Piece | State |
| --- | --- |
| A model as `fn(Tensor) -> Tensor`, gradient from `grad` | runs; `tests/model_test.tw` checks the value and the gradient together |
| Distributions: normal, half-normal, lognormal, student t | runs; `tests/dist_test.tw` |
| Distributions: exponential | written, untested. Nothing under `tests/` or `examples/` calls `exponential_log_prob` |
| Distributions: gamma, beta, dirichlet, categorical, multinomial | runs; `tests/dist_test.tw`, the densities and the gamma, Dirichlet and multinomial samplers |
| Multivariate normal, Cholesky parameterised, differentiable in the factor | runs; `tests/dist_test.tw` checks the gradient reaches the factor |
| Reparameterised forms where one exists, and a plain statement where none does | written, untested. No `_reparam` function is called from `tests/` or `examples/`; ADVI writes the trick out inline rather than calling one |
| Transforms: log, logit, interval, stick-breaking, ordered, Cholesky | runs; `tests/transform_test.tw` |
| Every transform's log Jacobian, derived in the source | runs; `tests/transform_test.tw` checks each against a numerical determinant of the forward map |
| Random walk Metropolis, with a Robbins-Monro proposal scale | runs; exercised through `tests/nuts_test.tw`, which uses it as the reference sampler |
| Static HMC, as a reference the tree can be checked against | runs; `tests/hmc_test.tw` |
| NUTS: multinomial sampling, generalised U-turn, dual averaging, diagonal mass | runs; `tests/nuts_test.tw` |
| Mean-field ADVI with the reparameterisation trick | runs; `tests/advi_test.tw` |
| Laplace approximation: Newton to the mode, Gaussian from the Hessian | runs; `tests/laplace_test.tw`, and `examples/logistic_laplace.tw` end to end |
| Diagnostics: split rank-normalised R-hat, ESS, MCSE, divergences | runs; `tests/diag_test.tw` |
| Dense mass matrix, Riemannian HMC | **not in v0.1** |
| Discrete parameters and their marginalisation | **not in v0.1** |
| Anything running end to end | yes, for Laplace: `twill run examples/logistic_laplace.tw` exits 0 in about half a second. The NUTS example is the slow one; see below |

## What a NUTS run looks like

Eight schools, the standard first hierarchical model, in its non-centred
parameterisation. `examples/eight_schools.tw` is this program complete, with the
centred version beside it for comparison.

**Read this before you run it.** `examples/eight_schools.tw` is four NUTS chains
of 1000 warmup and 2000 sampling draws each, run one after another, and it does
not finish quickly. I ran `twill run examples/eight_schools.tw` under twill
1.7.1 on Windows 11 and killed it at a 180 second timeout, still running. I did
not measure how long it actually takes, so I will not put a number on it. It is
not hung; it is a NUTS run with no progress output, which is `docs/needs.md`
entry 25: heddle's own gap rather than a missing clock. heddle's CI checks this file and does not run it, on purpose.

The transcript below is illustrative and is **not** captured output. I have not
run this example to completion, so the figures in it are what the summary table
looks like rather than numbers heddle produced. The only pasted output in this
README that I measured is the Laplace example under Getting started.

```rust
mode systems

import "twill_modules/heddle/src/dist.tw" as dist
import "twill_modules/heddle/src/transform.tw" as tr
import "twill_modules/heddle/src/model.tw" as model
import "twill_modules/heddle/src/nuts.tw" as nuts
import "twill_modules/heddle/src/diag.tw" as diag
import "twill_modules/heddle/src/chain.tw" as chain

let Y: Tensor = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0]
let SIGMA: Tensor = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0]

# Where each block sits in the flat parameter vector. Written down rather than
# counted by hand, because a hand-counted offset is right until a block is
# inserted and then silently reads the wrong parameter.
let L = model.new_layout()
model.declare(L, "mu", 1)
model.declare(L, "log_tau", 1)
model.declare(L, "z", 8)

# The model. An ordinary twill function. `grad` of it is the gradient of the
# log posterior, and heddle contains no backward pass for any of this.
fn logp(q: Tensor) -> Tensor {
  let mu: Tensor = model.scalar_block(L, q, "mu")
  let log_tau: Tensor = model.scalar_block(L, q, "log_tau")
  let z: Tensor = model.block(L, q, "z")

  let tau: Tensor = exp(log_tau)
  let theta: Tensor = mu + tau * z

  dist.normal_log_prob(mu, scalar(0.0), scalar(5.0))
    + dist.half_normal_log_prob(tau, scalar(5.0))
    + tr.positive_log_jac(log_tau)          # tau is sampled on the log scale
    + dist.normal_log_prob(z, scalar(0.0), scalar(1.0))
    + dist.normal_log_prob(Y, theta, SIGMA)
}

let cfg = nuts.nuts_config(1000, 2000, 20260807)
let chains = nuts.run_chains(logp, inits, cfg)

let s = diag.summarise(chains, model.coordinate_names(L))
let w = diag.warnings(chains, s, cfg.max_depth)
```

Output:

```
  parameter        mean      sd      5%     50%     95%    mcse     ess   rhat
  mu             7.9214  5.0687 -0.3183  7.8402 16.3915  0.0642  6241.2  1.0001
  log_tau        1.0296  0.8917 -0.6412  1.1447  2.3016  0.0138  4162.7  1.0004
  z[1]           0.3121  0.9749 -1.2802  0.3204  1.8975  0.0121  6479.1  1.0000
  ...
  divergent: 0 of 8000
```

Run the centred version of the same model and the last line changes:

```
  divergent: 137 of 8000

  heddle: 137 of 8000 transitions diverged. The posterior is biased, not merely
  noisy: divergences happen in a specific region, so that region is
  under-sampled. Raise target_accept above 0.9, or reparameterise.
```

Two parameterisations of an identical posterior. One of them samples and one of
them does not, and the divergence count is the only thing in the output that
says which is which.

## Diagnostics are output, not options

A Markov chain Monte Carlo run has no failure mode that looks like a failure.
There is no exception, no error code, no missing result. A chain that never left
its starting neighbourhood returns a smooth set of draws with a small standard
error and a wrong answer. The output of a broken run and the output of a good
run are the same shape, the same size, and the same type.

So heddle treats the diagnostics as the result and the draws as the by-product.
`summarise` returns R-hat, effective sample size and Monte Carlo standard error
beside every estimate, and `warnings` returns sentences rather than flags,
because a flag gets filtered out of a log and a sentence gets read.

Four checks, from three directions:

**R-hat**, split and rank-normalised. Split, because four chains all drifting
slowly in the same direction agree with each other at every moment, so an
unsplit R-hat reports 1.00 for a run that has not converged; cutting each chain
in half turns the drift into a disagreement between its own halves.
Rank-normalised, because a variance ratio is undefined for a heavy-tailed
posterior and badly estimated for a skewed one, and rank normalisation makes the
comparison exactly normal by construction and invariant under any monotone
reparameterisation. The threshold is 1.01, not the 1.1 from the 1992 paper,
which is far too loose to be useful.

**Effective sample size**, by Geyer's initial monotone positive sequence. The
truncation rule is the hard part and the usual shortcut, stopping at the first
negative autocorrelation, terminates far too early on a chain with a slow
oscillation and overstates the answer by a factor of several. Geyer's rule
instead sums consecutive pairs and stops where the estimate stops obeying a
property the true value provably has.

**Divergent transitions.** These come from a different direction entirely: they
are the sampler reporting that its integrator failed, and they fail in a
specific region, so a run with divergences has systematically under-sampled a
specific part of the posterior. R-hat can be 1.00, the trace can look perfect,
and the answer can still be wrong. The count is the only thing that says so.

**Tree depth saturation.** Not a correctness problem. It means the trajectory
had not turned after a thousand leapfrog steps, which is a message about the
model's geometry rather than about the sampler, and raising the cap makes each
draw slower without fixing it.

R-hat is a necessary condition for convergence and never a sufficient one. Four
chains that all found the same wrong mode agree perfectly. That is why heddle
reports all four and not the one that is easiest to compute.

## The Jacobian, said once and loudly

A sampler moves in unconstrained real space. Half the parameters anyone cares
about are not in unconstrained real space: a scale is positive, a probability is
in the unit interval, a mixture weight vector is on the simplex.

The fix is to sample an unconstrained x and define the constrained parameter as
y = f(x). The density then changes, and the log density the sampler must be
handed is

```
log p_y(f(x)) + log|det J_f(x)|
```

Drop the second term and nothing breaks. No error, no warning, no divergence, no
failed diagnostic. The sampler runs happily and returns a posterior that is
biased toward whichever end of the space the transform compresses. For the log
transform that means a systematically small scale parameter, which is the single
most common silent defect in hand-rolled Bayesian code, and it looks like a
tighter fit, which is why it survives review.

`src/transform.tw` therefore gives every transform in three parts, and derives
each Jacobian in the source rather than asserting it:

| Transform | Support | log&#124;det J&#124; |
| --- | --- | --- |
| `positive` | (0, inf) | `sum(x)` |
| `lower` / `upper` | half-line | `sum(x)` |
| `unit` | (0, 1) | `sum(log sigmoid(x) + log(1 - sigmoid(x)))`, computed stably |
| `interval` | (a, b) | the unit term plus `n log(b - a)` |
| `simplex` | the K-simplex from K-1 coordinates | the logit terms plus `sum log r_k`, the remaining stick |
| `ordered` | increasing vectors | `sum` of all but the first coordinate |
| `chol_factor` | positive-diagonal Cholesky factors | `sum` of the diagonal coordinates |

`tests/transform_test.tw` checks every one of them against a numerical
determinant of the forward map, and `tests/nuts_test.tw` runs a model with and
without the Jacobian to show the size of the bias it removes.

The simplex entry is the one that gets left out, because `log r_k` has no
counterpart in any single-parameter transform. Leaving it out pushes the
posterior toward the simplex corners.

## NUTS

The centrepiece. The trajectory is built by repeated doubling and stops when it
starts to double back on itself.

**The termination condition is the part people get subtly wrong**, so
`src/nuts.tw` spells it out. A sub-trajectory has turned when the momentum at
either of its ends points back toward the other:

```
p_sharp_minus . rho < 0    or    p_sharp_plus . rho < 0
```

Three things about that are easy to get wrong and each yields a sampler that
works well enough to ship:

1. **The metric must be applied.** `p_sharp` is `M^-1 p`, not `p`. With an
   identity mass matrix the two are the same, so a test on a standard normal
   passes; the error only appears once mass adaptation does something, as
   trajectories that stop too early in the wide directions.
2. **`rho` is the sum of the momenta over the sub-trajectory**, not the
   difference of the endpoint positions. They agree under a Euclidean metric and
   are different objects otherwise, so `rho` is carried up the tree through
   every merge.
3. **Both ends must be checked.** A one-ended criterion is not symmetric under
   reversing the trajectory, and an asymmetric stopping rule breaks detailed
   balance. The `or` is not defensive.

There are also two extra checks per merge that straddle the boundary between the
two halves. The main check misses a U-turn that happens exactly there, because
neither half saw it internally and the summed momentum over the whole can still
point forward. Omitting them gives a sampler that is correct on a Gaussian and
runs long wasteful trajectories on anything with curvature, which reads as a low
effective sample size and gets blamed on the model.

And the property that decides the design: **a sub-tree that fails invalidates
the entire doubling, not the part of itself that was fine.** Keeping the valid
prefix is tempting and wrong, because whether a state came before or after the
failure depends on the random direction the doubling took, so keeping it makes
the selection depend on that direction and the sampler stops targeting the
posterior.

Two departures from the 2011 paper, both current practice:

- **Multinomial rather than slice sampling** for choosing the state. It uses the
  whole trajectory instead of an energy slice of it, gets more effective sample
  size per gradient evaluation, and avoids a stall mode where one high-energy
  state leaves the slice nearly empty.
- **Biased progressive sampling at the top level.** Within the tree the halves
  are exchangeable and the rule is `w2/(w1+w2)`; at the top level, where the new
  half is freshly explored ground, it is `min(1, w2/w1)`. The bias pushes the
  draw toward the far end of the trajectory and remains valid because the
  doubling is symmetric in direction.

Warmup is Stan's three-stage schedule: a fast interval that adapts the step size
only, doubling windows that estimate the diagonal metric, and a final fast
interval that retunes the step size for the metric it ended with. Dual averaging
freezes its **averaged** iterate, not its current one, because the current one
is still jittering by design.

## Variational inference, and what it costs

`src/advi.tw` is mean-field ADVI: independent normals on each unconstrained
coordinate, fitted by maximising the evidence lower bound with the
reparameterisation trick and adagrad.

It is fast and it is the right tool for iterating on a model. It is not a
substitute for sampling, and the reasons are stated in the source rather than
buried:

- Mean field assumes the coordinates are independent. KL(q || p) penalises q for
  putting mass where p has none and not the reverse, so the fit **shrinks into**
  the posterior and systematically understates the variance. Means come out
  reasonable; credible intervals come out too narrow, often by a factor of two
  on a correlated posterior. `tests/advi_test.tw` asserts this rather than
  describing it.
- There is no diagnostic worth the name. R-hat and effective sample size are
  about chains and there are none. Running R-hat on draws from the fitted
  approximation gives 1.00, because they are independent by construction, and
  that number is not a check.
- The ELBO going flat says the optimiser stopped, not that the answer is right.

The use that survives all of that is `advi_init` and `advi_inv_mass`: the fitted
mean is a good starting point for NUTS and the fitted scales are a good initial
metric, so warmup is spent adapting rather than travelling.

## Repository layout

```
src/vec.tw          flat vectors, and the single seam into the tensor half
src/model.tw        a model is a function; parameter layout by name
src/dist.tw         densities, samplers, reparameterised forms
src/transform.tw    constrained parameters and their log Jacobians
src/adapt.tw        dual averaging, Welford, the warmup schedule
src/hmc.tw          the leapfrog, the Hamiltonian, static HMC
src/nuts.tw         tree doubling, the U-turn criterion, the run loop
src/rwm.tw          random walk Metropolis, the reference sampler
src/advi.tw         mean-field ADVI
src/laplace.tw      Newton to the mode, the Gaussian from the Hessian
src/diag.tw         R-hat, ESS, MCSE, divergences, the summary
src/chain.tw        what a sampler returns
tests/              tests, named as sentences
examples/           logistic regression by Laplace; eight schools, both
                    parameterisations
docs/needs.md       what the language asked for, and what arrived
```

## Getting started

Get a twill. The releases carry a single static binary per platform, named
`twill-v1.12.0-<os>-<arch>`, with assets `linux-amd64`, `linux-arm64`,
`darwin-amd64`, `darwin-arm64` and `windows-amd64.exe`:

```bash
curl -fsSL -o twill   https://github.com/twill-lang/twill/releases/download/v1.12.0/twill-v1.12.0-linux-amd64
chmod +x twill
./twill --version        # Twill 1.12.0
```

Then run the fast example. It is Bayesian logistic regression by Laplace
approximation, and it finishes in under a second:

```bash
twill run examples/logistic_laplace.tw
```

That is the whole of the output, captured from that command under twill 1.7.1:

```
Bayesian logistic regression by Laplace approximation
converged: true in 7 Newton steps

posterior mode, plus or minus one marginal standard deviation:
  intercept  -0.101148 +- 1.117369
  weight 1   2.01338 +- 1.185424
  weight 2   1.497331 +- 1.156241

log evidence: -3.089811
  the Laplace approximation to log p(data), for comparing models

predicted P(class one) by sampling 4000 posterior weights, mean +- sd:
  deep in class zero  0.049732 +- 0.109556
  deep in class one   0.940989 +- 0.123488
  on the boundary     0.480165 +- 0.224906
  the boundary point's wider band is the uncertainty a point estimate hides

the same predictions by the delta method, analytic, no sampling:
  deep in class zero  0.009771 +- 0.021463
  deep in class one   0.988065 +- 0.025824
  on the boundary     0.474735 +- 0.278629
```

The other example, `examples/eight_schools.tw`, is a NUTS run and does not
finish quickly. Read the warning above it before starting it.

To use heddle from your own project:

```
spool add heddle https://github.com/twill-lang/heddle
```

spool vendors into `twill_modules/`, and twill's import is a path, so the import
lines are the long ones in the example above and they resolve relative to the
project root. That is twill's rule rather than heddle's; see spool's README.

## Dependencies

twill, and nothing else. No third-party twill packages. `std/random` for the
seeded generator, `std/linalg` for the Cholesky and the triangular solves in
`src/dist.tw` and `src/laplace.tw`, and the tensor builtins are the whole
surface heddle builds on.

Two standard library functions are deliberately **not** used, and heddle carries
its own of each:

- `std/stats.tw`'s `lgamma`. A shape parameter is a parameter, so its log gamma
  has to be differentiable, and an `F64` function is a constant to `grad`.
  `src/dist.tw` carries a tensor Lanczos approximation for that reason, and
  `tests/dist_test.tw` checks it against the known factorials and checks that
  its gradient is the digamma.
- `std/nn.tw`'s `softplus`, which is the naive `log(1.0 + exp(x))` and overflows.
  `src/transform.tw` carries the stable form, and
  `tests/transform_test.tw` checks it does not overflow at a large argument.
  `docs/needs.md` entry 23 is the request to fix the one in `std/nn.tw`.

## What was left out, and why

- **A dense mass matrix.** It captures correlation, costs a Cholesky per
  adaptation window, and needs more draws than a window has for any dimension
  worth using it on. The diagonal captures scale, which is the part that varies
  by orders of magnitude. Where correlation is the actual problem, no metric
  fixes it and a reparameterisation does, which is the lesson of the eight
  schools example.
- **Riemannian HMC.** It needs the Hessian of the log posterior along the
  trajectory. twill has `hessian`, so this is possible rather than blocked, and
  it is out of v0.1 because a soft-absolute-value metric has its own tuning
  problem and heddle has not earned the right to add one.
- **Marginal likelihood and model comparison.** Bridge sampling needs a second
  reference density and a careful iteration; the shortcuts (the harmonic mean
  estimator in particular) are worse than not answering.
- **Discrete parameters.** No gradient exists, so HMC cannot move them. The
  correct answer is to marginalise them out inside the model, which the user can
  already do with `logsumexp`, and the wrong answer is a Gibbs step bolted onto
  NUTS that quietly breaks the invariance the whole sampler rests on.
- **A `Distribution` type with `log_prob` and `sample` as fields.** twill has
  function values in struct fields, so this is possible rather than blocked; I
  checked by compiling a struct with an `fn(F64) -> F64` field under twill
  1.7.1 and calling through it. It is left out because it would buy nothing: no
  code here dispatches over a distribution at runtime, because the model writer
  names it at the call site. `docs/needs.md` entry 4 is updated to say so.

## Contributing

The most useful contribution right now is not code. It is a correction to
[`docs/needs.md`](docs/needs.md): a feature listed there that the language
already has, a workaround that is worse than described, or a missing entry found
by reading the source.

After that, the U-turn criterion and the two extra boundary checks in
`src/nuts.tw` are the part most worth arguing with, followed by the Jacobian
derivations in `src/transform.tw`. Both are places where being right matters and
being wrong is invisible.

## License

MIT. See [LICENSE](LICENSE).
