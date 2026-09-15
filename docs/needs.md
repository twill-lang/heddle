# What heddle needs from twill

heddle is written in twill and it runs. `twill test tests` passes 7 of its 8
suites on any machine and the eighth on some of them: `tests/nuts_test.tw` is
27/27 on linux/amd64 and 26/27 on arm64, where the half-normal posterior mean
comes back 0.8299 against a tolerance of 0.7979 ± 0.03. That is not a heddle
defect and not a twill one. Go's `math.Exp` differs by one ULP between the two
architectures, a seeded NUTS run amplifies it: one trajectory diverges early
and the sampler takes a different path, and the answer moves by a thousand
times the input difference. Measured on 2026-09-01 and written up in twill's
`docs/CORRECTNESS.md` section 4. **A test that pins a number produced by an
iterative float method pins the architecture it was written on**, and the
tolerance to fix this one is heddle's decision, not a language change.

This file started as the reason none of it ran: the language and
runtime features the source uses that twill did not provide, with the file and
function that needs each one, and what heddle did in the meantime.

Most of it is now history. Entries 1 to 18 and 21 are delivered, and the proof
is that the suites pass rather than an announcement in a changelog: every one of
those features is on the path a passing test takes. Entry 20 was delivered and
adopted. Entry 24 was delivered by `twill test`, and heddle's CI uses it. Entry
4 was delivered and heddle has chosen not to use it, which is a different thing
and is said as such below. What is still open is entries 19, 22, 23, 25, 26 and
27, and they are the useful part of this file today.

Each delivered entry keeps its original text, because the argument it made is
what got it built and the wording of the ask is worth preserving. The
**Delivered** line at the head of each says what arrived and where heddle uses
it. Where an entry's own **Status:** line contradicts the Delivered line, the
Delivered line is the current one.

It is a work queue for the language, not a complaint. Every entry was reached by
writing real code and hitting the wall, which is the only way a list like this is
worth anything. Where heddle has a workaround the workaround is described,
because how ugly a workaround is measures how badly the feature is wanted.

The baseline is milestone 1 of `docs/self-hosting.md` in the twill repository:
`mode systems`, `I64` with bitwise operations, `Str` with length and byte
indexing, `Arr[T]`, `Dict[Str, V]`, `struct`, and `read_file`. Everything below
is on top of that.

Entries that restate something already in twill's own `docs/needs.md` say so and
give the NEEDS number. Those are not new work; they are evidence that a second
independent package hit the same wall, which is the only argument for priority
this list can make.

## Delivered: heddle could not run at all without these, and now it does

### 1. `mode systems` itself

**Delivered.** twill 1.6. Every `.tw` file in this repository begins `mode
systems` and CI gates on it.

**Used by:** every file
**Status:** designed in `docs/self-hosting.md`, not implemented. Duplicates twill
NEEDS-1.

Nothing else on this list matters until this does.

### 2. Tensors, `grad`, `grads` and `value_and_grad` inside a systems-mode file

**Delivered.** `Tensor` is a legal annotation in systems mode and the
differentiation builtins are callable there. `src/model.tw`'s
`value_and_gradient` is the seam, and `tests/model_test.tw`'s
`the_value_and_gradient_of_a_model_are_both_right` is the check. This is the
entry heddle existed to file.

**Needs:** `Tensor` as a type name in an annotation, and the differentiation
builtins callable from systems mode
**Used by:** `src/model.tw` (`value_and_gradient`, `log_density`),
`src/dist.tw` (every density), `src/transform.tw` (every transform),
`src/advi.tw` (`advi_fit`)
**Status:** not stated anywhere. `docs/self-hosting.md` section 1.3 says the
shape machinery is inactive in systems mode and says nothing about whether a
tensor is a value there at all.

This is the entry heddle exists to file. The whole library is a thin layer over
`grad`, and the layer has to be written in systems mode because it needs `Arr`,
`struct` and `Dict`, and the thing it wraps is a numeric-mode builtin. If those
two halves cannot meet in one file, heddle cannot be written in any form.

What heddle assumes: `Tensor` is a legal annotation, the differentiation
builtins take and return ordinary values, and a systems-mode function may be
passed to `grad` (see entry 3). The shape checker staying quiet about tensors in
systems mode is fine and expected.

loom's `docs/needs.md` entry 2 asks the adjacent question for a parameter tree.
heddle does not need `Tree`: its parameters are one flat rank-1 tensor by
design, because a sampler's state is a point in R^n. The two entries are the
same seam seen from two sides.

### 3. Function values as parameters, and a function type in an annotation

**Delivered.** `fn(Tensor) -> Tensor` is a parameter type and a function value
is passed into a systems-mode function. `src/nuts.tw`, `src/hmc.tw`,
`src/rwm.tw`, `src/advi.tw` and `src/laplace.tw` all take the model this way.
Closures capturing the environment work too, which `src/adapt.tw`'s
`initial_step_search` needs.

**Needs:** `fn(Tensor) -> Tensor` as a parameter type, and a function value
passed into a systems-mode function
**Used by:** `src/nuts.tw` (`nuts_draw`, `run_chain`, `build_tree`),
`src/hmc.tw` (`leapfrog`, `hmc_draw`), `src/rwm.tw` (`rwm_draw`),
`src/advi.tw` (`advi_fit`), `src/model.tw` (`value_and_gradient`),
`src/adapt.tw` (`initial_step_search`, which takes `fn(F64) -> F64`)
**Status:** functions are values in numeric twill; whether a systems-mode
function may take one, and how the type is spelled, is not stated. Duplicates
loom's entry 3.

Not a convenience. The model **is** the function, so a sampler that cannot take
one has no way to be handed a model. There is no alternative design here: a
struct holding the model would need entry 4, and a global would make two models
in one program impossible.

`src/adapt.tw` also passes a closure that captures `probe` and `h_probe`, so the
requirement is a real closure and not a top-level function pointer.

### 4. Function values in struct fields

**Delivered.** twill takes a function as a struct field type. I checked under
twill 1.7.1 by declaring `struct Dist { name: Str, log_prob: fn(F64) -> F64 }`,
building one and calling through the field; it runs. heddle still has no such
struct, and that is now a decision rather than a limit, for the reason this
entry already gives: nothing here dispatches over a distribution at runtime.
`README.md` said this was blocked; that has been corrected.

**Needs:** a struct field whose type is a function
**Used by:** would be used by a `Distribution` struct in `src/dist.tw`, and by a
`Target` struct in `src/model.tw`
**Status:** same as entry 3 and probably the same work. Duplicates loom's entry
3, second half.

heddle's workaround is to have no such struct: every distribution is three free
functions (`normal_log_prob`, `normal_sample`, `normal_reparam`) and the model
is passed as a bare parameter through every sampler function.

The workaround is defensible and was going to be the design anyway, so this is a
low priority. Bundling would only pay off if something dispatched over a
distribution at runtime, and nothing here does, because the model writer names
the distribution at the call site. Recorded so that the absence reads as a
decision rather than an omission.

### 5. `tensor_of_arr(Arr[F64]) -> Tensor`

**Delivered.** Spelled `tensor(xs)`, which accepts a systems-mode `Arr[F64]`.
`src/vec.tw`'s `to_tensor` is one line because of it.

**Needs:** a rank-1 tensor built from a systems-mode `Arr[F64]`
**Used by:** `src/vec.tw` (`to_tensor`), and therefore every gradient evaluation
in `src/hmc.tw`, `src/nuts.tw`, `src/rwm.tw` and `src/advi.tw`
**Status:** `tensor(list)` builds a tensor from a numeric-mode heterogeneous
list. An `Arr[F64]` is not that list and there is no conversion.

Called twice per gradient evaluation, so at tree depth 10 it is called about a
thousand times per NUTS draw. It should be a copy of a contiguous buffer and
nothing more.

heddle could avoid it by writing the samplers on tensors instead. That was
considered and rejected: a leapfrog written on tensors records a tape node per
half step, and the tape then holds a thousand-position NUTS trajectory alive.
The tape is for the density, not for the integrator. `src/vec.tw`'s header
argues this at length.

### 6. `arr_of_tensor(Tensor) -> Arr[F64]` and `item` into an `F64`

**Delivered.** Both. `arr_of_tensor` is a builtin and `item` yields a
systems-mode `F64`. `src/vec.tw`'s `of_tensor` and `scalar_of` are one line
each.

**Needs:** the other direction of entry 5, and a rank-0 tensor as a systems-mode
`F64`
**Used by:** `src/vec.tw` (`of_tensor`, `scalar_of`)
**Status:** indexing a rank-1 tensor yields a rank-0 tensor and `item` yields
its scalar in numeric mode. Neither is spelled for a systems-mode `F64`.

`scalar_of` is separate from `of_tensor` on purpose: a log density returns one
number and unwrapping it through a one-element array would be a lie about the
shape. Both are needed.

### 7. `concat_tensors(Arr[Tensor], axis) -> Tensor`

**Delivered.** `concat(pieces, axis)` takes a systems-mode `Arr[Tensor]` and
the result stays on the tape. `src/transform.tw` uses it in `simplex`,
`ordered` and `chol_factor`, and `tests/transform_test.tw` checks each of their
Jacobians against a numerical determinant, which only passes if the gradient
survived the concat.

**Needs:** `concat` over a systems-mode `Arr[Tensor]`
**Used by:** `src/transform.tw` (`simplex`, `simplex_inverse`, `ordered`,
`ordered_inverse`, `chol_factor`), `tests/dist_test.tw`
**Status:** `concat(list, axis)` takes a numeric-mode list. Related to twill
NEEDS-72 (nested containers), which asks the general question.

Every transform that builds a vector one component at a time needs this, and
they build it one component at a time because each component depends on the ones
before it. Stick breaking is inherently sequential; there is no vectorised form
that does not compute the running remainder.

The result must stay on the tape, since the transform sits inside the
differentiated density. A conversion that severed the gradient would be worse
than nothing, because it would be silent.

### 8. `continue` in a `while` loop

**Delivered.** twill has `continue`. `src/dist.tw` uses it at three sites,
including the `v <= 0.0` redraw in the Marsaglia-Tsang loop in `gamma_sample`,
so the flag workaround this entry describes was never written.

**Used by:** `src/dist.tw` (`gamma_sample`, the Marsaglia-Tsang rejection loop;
`multinomial_sample`, the exhausted-trials case)
**Status:** not implemented. Duplicates twill NEEDS-12, and loom's entry 14.

`gamma_sample` is a rejection sampler and rejection is what `continue` is for.
Without it the loop body has to be restructured around a flag, which is
mechanical and makes the accept condition read as its own negation, in a
function whose correctness is the reason the beta, the Dirichlet and the
student t are correct.

### 9. Recursion, and a depth that reaches 10

**Delivered, and now the number is stated.** `src/nuts.tw`'s `build_tree`
recurses and `tests/nuts_test.tw` passes, so the depth reaches at least
`max_depth`. The recursive formulation was kept.

twill 1.12 set the limit: a call nested more than 10,000 deep is refused with
a twill error that names the function and the line, where before it was a Go
stack overflow with neither. Eleven frames is three orders of magnitude under
it. The twill changelog is worth reading on what the limit does not promise:
it is a diagnostic and not a guarantee, and a runaway call nested inside deep
enough expression nesting can still overflow the host before the counter
reaches 10,000. `build_tree` is not that shape, and the depth it reaches is
bounded by `max_depth` rather than by a missing base case.

**Used by:** `src/nuts.tw` (`build_tree` calls itself), `src/dist.tw`
(`gamma_sample` calls itself once for the shape-below-one boost)
**Status:** twill NEEDS-30 asks for a recursion depth and a guard on it, without
naming a number.

NUTS is naturally recursive and the depth is bounded by `max_depth`, so eleven
frames is the worst case and the requirement is modest. Recorded because
NEEDS-30 leaves the limit unstated and a limit below eleven would make the
centrepiece of this repository unwritable.

The iterative formulation exists and is genuinely harder to read: it needs an
explicit stack of partially built sub-trees and the merge order becomes
implicit. Given that the termination condition is the part everyone gets wrong,
readability here is worth more than it usually is.

## Delivered: features the source assumed exist, and does

### 10. `F64` scalar math in systems mode

**Delivered.** All of them, and `f64_pow` takes a negative exponent:
`src/adapt.tw`'s `m^-kappa` and `src/advi.tw`'s `n^-1/2` both run.

**Needs:** `f64_sqrt`, `f64_log`, `f64_exp`, `f64_pow`, `f64_min`, `f64_of_i64`,
`i64_of_f64`
**Used by:** every file except `src/chain.tw`
**Status:** `std/stats.tw` and `std/random.tw` already call all of these, so
they are assumed rather than proposed. Duplicates twill NEEDS-40 and NEEDS-68.

Recorded only to note that `f64_pow` is used with a negative exponent in
`src/adapt.tw` (`dual_avg_update`, the `m^-kappa` averaging weight) and in
`src/advi.tw` (the `n^-1/2` step decay). A `pow` restricted to positive
exponents would break both.

### 11. Struct field mutation through a parameter

**Delivered.** Structs have reference semantics and every accumulator in
`src/adapt.tw` mutates through its parameter.

**Used by:** `src/adapt.tw` (`dual_avg_update`, `welford_add`, `next_window`),
`src/nuts.tw` (`merge` writes to `out`), `tests/harness.tw` (`check` and the
whole counter)
**Status:** `docs/self-hosting.md` says structs have reference semantics.
Duplicates twill NEEDS-42 and NEEDS-67, and loom's entry 10.

Every adapter here is a small mutable accumulator and the alternative is
returning a fresh struct from every update, which for `welford_add` inside the
warmup loop means an allocation per draw per coordinate.

### 12. `Arr[T]` element assignment

**Delivered.** Delivered and used throughout `src/vec.tw`, `src/hmc.tw`,
`src/rwm.tw`, `src/advi.tw` and `src/diag.tw`.

**Used by:** `src/vec.tw` (`copy` and the arithmetic), `src/hmc.tw`
(`leapfrog`, the position update), `src/rwm.tw` (`rwm_draw`, the proposal),
`src/advi.tw` (the adagrad update), `src/diag.tw` (`sorted_copy`,
`rank_normalise`)
**Status:** duplicates twill NEEDS-43.

### 13. Integer division and modulo on `I64`

**Delivered.** `/` on two `I64` is integer division.
`tests/transform_test.tw`'s `the_cholesky_size_counts_the_free_entries` is the
check that `chol_size` is not off by a half.

**Used by:** `src/diag.tw` (`split_chains`, the halving), `src/transform.tw`
(`chol_size`), `src/rwm.tw` (the periodic shape refresh)
**Status:** duplicates twill NEEDS-24 and NEEDS-44.

`chol_size` is `n + n * (n - 1) / 2` and is wrong by a half if `/` is float
division, in a way that produces a parameter vector one element short and a
shape error a long way from here.

### 14. `Arr[Arr[F64]]` and `Arr[Chain]`, nested

**Delivered.** Nested containers work. `src/diag.tw` takes `Arr[Arr[F64]]`
throughout and `tests/diag_test.tw` passes, which requires the chains to have
stayed apart.

**Used by:** `src/chain.tw` (`Chain.draws`, `columns`), `src/nuts.tw`
(`run_chains`), `src/diag.tw` (every diagnostic takes `Arr[Arr[F64]]`),
`src/advi.tw` (`advi_draws`)
**Status:** duplicates twill NEEDS-72.

R-hat is a comparison of within-chain against between-chain variance, so the
chains have to stay apart. Flattening them into one array first is the mistake
that makes R-hat report 1.00 always, and there is no representation that avoids
nesting.

### 15. `Bool` in an annotation, and an `Arr[Bool]`

**Delivered.** Delivered. `src/chain.tw`'s `Chain.divergent` is an `Arr[Bool]`.

**Used by:** `src/chain.tw` (`Chain.divergent`), `src/nuts.tw` (`Subtree.ok`,
`Subtree.divergent`), `src/vec.tw` (`all_finite`, `is_finite`)
**Status:** duplicates twill NEEDS-14.

### 16. `str()` on an `I64` and an `F64`

**Delivered.** Both. `src/diag.tw`'s `warnings` renders an R-hat into a
sentence and `tests/diag_test.tw` reads the summary's labels back.

**Used by:** `src/diag.tw` (`warnings`), `src/model.tw` (`declare`,
`coordinate_names`), `tests/harness.tw`, `examples/eight_schools.tw`
**Status:** duplicates twill NEEDS-45 for `I64`. The `F64` case is NEEDS-29 and
NEEDS-89, which are about a round-trip rendering.

`src/diag.tw` prints an R-hat inside a warning sentence. A rendering that gives
`1.0100000000000002` there makes the warning look like a bug in the diagnostic,
so the shortest round-tripping form matters more here than in most callers.

### 17. `Str` concatenation with `+`

**Delivered.** Delivered, and used in `src/diag.tw`, `src/model.tw` and every
test file.

**Used by:** `src/diag.tw` (`warnings`), `src/model.tw` (`declare`),
`tests/harness.tw`, `examples/eight_schools.tw`
**Status:** duplicates twill NEEDS-35 and NEEDS-99.

### 18. `exit(I64)`

**Delivered.** Delivered, and used for a while by `tests/harness.tw`'s
`report`, so a red suite failed CI. That file is gone: the suites report
through `std/test`, whose `report` returns the status rather than raising it,
and `twill test` reads the failure count off the summary line; see entry 24.

**Used by:** `tests/harness.tw` (`report`)
**Status:** duplicates twill NEEDS-28.

Without it a failing test prints and returns zero, so CI passes on a red suite,
which is worse than having no CI.

## Not blocking, but the source is worse without them

### 19. `Res[T, E]`, or multiple return values

**Used by:** `src/model.tw` (`declare` returns a `Str` that is empty on success),
`src/transform.tw` (`Simplex`, `Ordered` and `CholFactor` each exist only to
return a value and its Jacobian together), `src/hmc.tw` (`DrawResult`),
`src/nuts.tw` (`Subtree`), `src/advi.tw` (`AdviResult`), `src/model.tw`
(`LogDensity`)
**Status:** the language half is done -- `Res[T, E]` and `Opt[T]` are checked
types (twill 1.6) and twill 1.7 closed NEEDS-4, so a declaration here can take
type parameters too. Multiple return values arrived in twill 1.12 as tuples,
and the three structs the correction below said they would remove are gone:
`simplex`, `ordered` and `chol_factor` return `(Tensor, Tensor)`, the value
and its log Jacobian in that order, and a caller writes
`let (y, log_jac) = tr.simplex(x, k)`. `LogDensity` stays, as the correction
said it should, and so do `DrawResult`, `Subtree` and `AdviResult`: a tuple
holds parts a reader takes in order, and those have parts a reader wants by
name. The error-convention half, `model.declare` returning a `Str`, is still
heddle's to adopt.

**One correction, because this entry asked for the wrong thing.** It says the
two-value returns need generics. Three of the four do not: `Simplex`, `Ordered`
and `CholFactor` are the same *concrete* struct three times -- two `Tensor`
fields, a value and its log Jacobian -- so what removes them is one shared
struct, or the tuple this entry also asks for, and a type parameter would be
ceremony over a type that never varies. `LogDensity` is the one that differs
(`F64` and `Arr[F64]`), and unifying it with the others would take two
parameters to say something no reader wanted said. Generics arriving therefore
does not resolve this entry; multiple return values still would. Duplicates
twill NEEDS-10, and loom's entry 4 and weft's entry 13.

Two separate complaints in one entry, because they have one fix.

The error convention: every fallible function returns a `Str` that is empty on
success, which is spool's convention and has spool's problem, that the compiler
does not make anyone read it. `model.declare` is the case that matters: a
duplicate block name is refused and a caller who ignores the return gets a model
whose parameter vector is one block short.

The two-value returns: `Simplex`, `Ordered` and `CholFactor` are the same
struct three times, a value and its log Jacobian. They are not types anyone
wanted; they are tuples with names. `LogDensity` is the fourth.

### 20. Sum types and `match`

**Used by:** would be used by `src/nuts.tw` (`Subtree.ok` and
`Subtree.divergent` are a two-bit encoding of a three-case outcome: valid,
turned, diverged), `src/dist.tw` (the family of distributions)
**Status:** done, and the paragraph below it describes code that no longer
exists. The language half landed in twill 1.6 -- `enum` with payloads, `match`
and enforced exhaustiveness -- with nested patterns, literal patterns and guards
added in 1.7. heddle adopted it at the same time: `src/nuts.tw` declares
`enum Outcome { Valid, Turned, Diverged }` and `Subtree.outcome` is one of them,
so the two Bools and the "not ok, not divergent means turned" convention are
gone, and `can_extend` is the single place that decides. `src/dist.tw` turned
out not to want an enum at all: it is a flat set of `<family>_log_prob`,
`_sample` and `_reparam` functions with no tag to dispatch on, and every caller
names the family statically. Duplicates twill NEEDS-3 and loom's entry 5.

*Previously, and now describing nothing:* "`Subtree` carries two Bools where one
enum with three cases would do, and the combination \"not ok, not divergent\"
means \"turned\" by convention rather than by construction."

`Subtree` carries two Bools where one enum with three cases would do, and the
combination "not ok, not divergent" means "turned" by convention rather than by
construction. Nothing forces a reader of `merge` to handle the third case, and
the merge is the function where a missed case is a biased posterior.

### 21. A tensor `logsumexp` over a systems-mode axis argument

**Delivered.** The axis argument is an `I64` in systems mode.
`tests/dist_test.tw`'s categorical and multinomial cases pass, which is the
check.

**Used by:** `src/dist.tw` (`categorical_log_prob`,
`categorical_log_prob_many`, `multinomial_log_prob`)
**Status:** `logsumexp(t, axis)` exists in numeric mode and defaults to the last
axis. Whether the axis argument is an `I64` in systems mode is unstated, and
falls under entry 2.

### 22. A differentiable `lgamma` in the standard library

**Needs:** `lgamma` over a `Tensor`, differentiable
**Used by:** `src/dist.tw` (`lgamma`, `lbeta`, and through them the gamma, beta,
Dirichlet, student t and multinomial densities)
**Status:** still open as of twill 1.7.1. `std/stats.tw` has
`lgamma(x: F64) -> F64` and nothing in `std` has a tensor `lgamma`; I checked by
grepping `std/` for `fn lgamma` and `fn digamma`. Not differentiable and
therefore not usable here.

heddle carries its own Lanczos approximation over tensors. It is about
twenty lines and it is correct, so this is not a wall. It is duplication of a
constant table that already exists in `std/stats.tw`, and two transcriptions of
nine Lanczos coefficients in one ecosystem will eventually disagree.

The right home is `std/num.tw`, beside the other tensor functions, for the same
reason `std/stats.tw` explains at its head: the tensor version and the `Arr`
version are complements rather than duplicates, and this one is missing.

A digamma would also be welcome and is not needed: `grad(lgamma)` is the
digamma, exactly, which is the point.

### 23. A stable `softplus` in `std/nn`

**Needs:** `std/nn.softplus` computed as `max(x,0) + log(1 + exp(-|x|))`
**Used by:** `src/transform.tw` (`softplus`, `log_sigmoid`,
`log_one_minus_sigmoid`, and through them every unit-interval and simplex
Jacobian)
**Status:** still open as of twill 1.7.1. `std/nn.tw` line 654 is still
`fn softplus(x) = log(1.0 + exp(x))`, which is `+inf` by x = 710 and has lost
all precision by x = -37.

Both ends are reached by a real sampler: an unconstrained scale wanders to
plus or minus 40 routinely during warmup, and an infinity in a Jacobian term
poisons the whole trajectory and shows up as a divergence with no cause.

heddle carries its own. Changing the one in `std/nn.tw` is still worth doing,
but the last sentence of this entry used to say it was "a strict improvement
with no behaviour change for any argument where the current one is finite", and
that is not true. Measured against the twill binary on 2026-09-01:

| x | `log(1 + exp(x))` | `maximum(x,0) + log(1 + exp(-abs(x)))` |
| --- | --- | --- |
| 800 | `+Inf` | 800 |
| 30 | 30 | 30 |
| 0 | 0.693147 | 0.693147 |
| **grad at 0** | **0.5** | **1** |

The values agree everywhere the current one is finite. The **gradient at exactly
zero does not**, because the rewrite is a sum of two kinked terms whose kinks
cancel everywhere except at the kink itself: `grad(maximum(x,0))` is 1 at zero
and `grad(abs(x))` is 0, so the two subgradient conventions add to 1 where the
true derivative is `sigmoid(0) = 0.5`.

Softplus is smooth, so this is the rewrite importing a non-differentiable point
into a function that does not have one. It is one input out of the reals and the
sampler's chance of landing on exactly 0.0 is negligible, but a language whose
whole claim is that `grad` is built in should decide that deliberately rather
than discover it. The alternatives, a `where` threshold or a `log1p` builtin,
each cost something else: `where` evaluates both branches, so the overflowing
one poisons the gradient with a NaN even when its value is discarded, and
`log1p` is a new builtin rather than a change to a standard-library line.

Still open, and now open on a decision rather than on nobody having written it.

### 24. A test runner

**Delivered.** `twill test [path ...]` collects every `*_test.tw` under the
paths given, and it has a `--filter <substring>`. heddle's CI runs `./twill
test tests` rather than a hand-maintained file list, so a new test file is
picked up the moment it is added. That was the whole of the ask.

The assertions came later, in twill 1.11, as `std/test`, and
`tests/harness.tw` is deleted: every suite imports `std/test` as `t` with the
same `check`, `equal_str`, `equal_i64`, `near` and `report` it called before.
`near_grad` was heddle's own and lives in `tests/dist_test.tw`, the one suite
that calls it, rather than in a shared helper, because a helper that imports
`std/test` gets its own counter and a failure it recorded would never reach
the suite's `report`. `report` prints its summary in the shape the runner
reads, so `twill test` now shows the counts beside each file where the copy's
`dist: 35 passed, 0 failed` gave it nothing to parse.

**Needs:** a `twill test` that collects `tests/*_test.tw`
**Used by:** everything in `tests/`
**Status:** none. Duplicates twill NEEDS-80's neighbourhood, and the same gap
spool, loom and weft all record.

Every test file is a program that calls its cases at the bottom and ends with
`report`, which exits non-zero on a failure. That is enough for CI on the day
`mode systems` runs, and it means a new test file is invisible to CI until
someone adds it to the workflow by hand.

### 25. A timer

**Needs:** a monotonic millisecond clock
**Used by:** would be used by `src/nuts.tw` (`run_chain`) to report draws per
second and to estimate the remaining time
**Status: delivered, and this entry was wrong about it.** `mono_ns()` returns a
monotonic nanosecond count and `clock_now_ms()` a wall-clock millisecond one;
both were checked against the binary on 2026-09-01, and `mono_ns` has been in
the language since 1.6.0-rc1, before the release this entry says it is missing
from. loom's entry 16, which this one calls a duplicate, has said "DELIVERED in
twill 1.7" the whole time, so the two cross-referenced each other and
disagreed.

What remains is heddle's, not the language's: `run_chain` reports nothing while
it runs. The number this entry argues for is the ratio of gradient evaluations
to draws rather than elapsed time, and heddle already counts the evaluations,
so what is missing is where the line is printed and how often, which is an
output-format decision and not a clock.

A NUTS run takes minutes to hours and reports nothing while it runs. The useful
number is not elapsed time but the ratio of gradient evaluations to draws, which
tells a user whether the trajectories are long because the posterior is hard or
because the step size is wrong. heddle can count the evaluations and cannot say
how long they took.

### 26. Parallel chains

**Needs:** some way to run four `run_chain` calls at once
**Used by:** `src/nuts.tw` (`run_chains`)
**Status:** still open as of twill 1.7.1. Not designed anywhere, and there is
no process, thread or spawn builtin. `docs/self-hosting.md` mentions threads
only as something the native core has.

This is the reason `tests/nuts_test.tw` and `tests/diag_test.tw` dominate the
test suite's wall clock. `README.md` carries the measured numbers.

Four chains are the standard because R-hat needs several, and they are perfectly
independent: separate seeds, separate state, no shared mutable anything. This is
the most embarrassingly parallel workload in statistics and heddle runs them one
after another, so a wall-clock run takes four times longer than it needs to.

Not blocking, and the largest single performance item on this list.

### 27. Sorting an `Arr[F64]` from a shared place

**Used by:** `src/diag.tw` (`sorted_copy`)
**Status: delivered in twill 1.9.0, and adopted.** `sort` is a builtin that
returns a new array, orders `F64` values, and has no cutoff constant to
inherit, so `sorted_copy` is one line and the insertion sort below is gone.

*What the entry said while it was open:* `std/stats.tw` has `sorted`, and
using it would pull in that module's `SORT_CUTOFF` constant, tuned for a
different case. Related to twill NEEDS-23, which asks for `Arr[Str]`.

heddle writes an insertion sort, which is correct and quadratic, on arrays of a
few thousand elements, once per run. Acceptable and slightly embarrassing. A
generic sort taking a comparison function would remove it, and that is loom's
entry 11.
