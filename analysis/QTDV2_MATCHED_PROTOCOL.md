# QTDv2 matched-comparator protocol

## Purpose

This protocol tests an empirical question, not a claim of quantum advantage:
whether a trainable quantum feature ranker improves held-out rare-failure ranking
over an equally trainable classical nonlinear feature ranker on the same data,
splits, label budget, and ranking budget.

## Locked design

- Dataset: the existing deterministic 256-state geometric benchmark. Results
  are **within-benchmark** only and do not establish safety validation or
  external generalization.
- Outer evaluation: ten stratified, label-disjoint 50/50 splits (seeds 42--51).
  No test labels are used by feature tuning, model fitting, or ranking.
- Inner tuning: a deterministic stratified partition of each outer-training
  split. Only this inner partition is used to select feature parameters.
- Methods:
  1. raw-feature logistic ranker;
  2. QTDv2, a two-layer, four-qubit input-encoded statevector feature map with
     trainable rotation offsets and a logistic ranking head;
  3. a matched classical trigonometric feature map with the same number of
     trainable offsets, the same SPSA optimizer, inner validation rule, and
     logistic ranking head.
- Budget: every ranker evaluates exactly the same first k candidates at
  k={5,10,20,30,50}; no environment call occurs during training or tuning.
- Primary outcome: paired held-out discovery-AUC difference at k=50 between
  QTDv2 and the matched classical feature map.
- Secondary outcomes: average precision, recall@k, precision@k, and raw
  logistic-ranker comparisons.

## Decision rule

QTDv2 is called an **empirical improvement on this benchmark** only if all of
the following hold on the locked ten outer splits:

1. the paired bootstrap 95% CI for mean discovery-AUC difference versus the
   matched classical feature map has lower bound above zero;
2. at least eight of ten paired discovery-AUC differences are positive; and
3. mean average-precision difference is positive.

This is not a quantum-advantage claim. A statevector simulation and a
256-state benchmark cannot establish computational advantage over classical
computation.

## Reporting safeguards

- Persist every split's state IDs, inner-fit/inner-tune IDs, selected feature
  parameters, candidate ranking, and all metrics.
- Report all methods and all seeds; do not promote a seed or configuration
  selected after inspecting test results.
- Do not add any QTDv2 result to the manuscript unless this protocol is run
  unchanged, its raw output is retained, and the result is reported with its
  matched-classical comparator.
