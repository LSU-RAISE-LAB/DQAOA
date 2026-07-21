# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-07-20

### Added

- Brute-force, CPLEX MIQP, monolithic QAOA, abstract distributed QAOA, and explicit TeleGate-based distributed QAOA modes.
- QUBO canonicalization from dense quadratic, linear, and constant inputs.
- Automatic contiguous and graph-aware QPU allocation, plus manual allocation support.
- Parameterized circuit reuse and transpile-once/bind-many execution.
- SPSA-gradient Adam optimization, depth progression, warm starts, random restarts, batching, and parallel restarts.
- Distribution-level metrics, exact-reference comparisons, convergence reporting, and validation helpers.
- Plotting and circuit-figure utilities.
