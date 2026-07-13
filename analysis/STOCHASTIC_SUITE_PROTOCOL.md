# Stochastic encounter-suite protocol

## Scope

This is a synthetic, aviation-inspired stress-ranking suite. It is not an
operational encounter model, certification artifact, or replacement for a
validated collision-avoidance simulator. Its purpose is to move evaluation
beyond one deterministic 256-state grid while preserving known data-generating
mechanisms, rare outcomes, and reproducible held-out regimes.

## Locked suite

Each scenario has six continuous variables: relative heading, relative altitude,
speed, lateral offset, response delay, and wind. A hidden stochastic encounter
oracle produces Bernoulli failures from a nonlinear risk surface. Four regimes
are fixed in source:

1. `nominal` -- baseline encounter distribution;
2. `crosswind_shift` -- wind/heading distribution shift;
3. `late_response` -- delayed-response shift;
4. `altitude_shift` -- altered vertical-separation regime.

Each regime has a disjoint 512-scenario training pool and 512-scenario test
pool, fixed by its seed. Training labels are observed failure counts from 32
independent rollouts per scenario. Test labels are observed counts from 256
independent rollouts per scenario, never used by training or tuning.

## Evaluation commitment

- First report all four fixed regimes; do not retain only a favorable one.
- Treat each test scenario's empirical failure rate as the ranking relevance
  value; report risk-mass recall@k, precision@k, and normalized discovery AUC.
- Compare QTD only with matched classical transforms under equal labels,
  rollouts, parameter tuning, and candidate budgets.
- Keep task definitions, seeds, and all raw scenario records immutable after
  the first materialization. A later task expansion must use new names and
  seeds, not replace these records.
