# Solver modes

| Mode | Purpose | Required dependencies | Main caveat |
|---|---|---|---|
| `bruteforce` | Exact enumeration baseline | NumPy | Exponential in the number of variables |
| `miqp_cplex` | Classical MIQP baseline | DOcplex plus a compatible CPLEX runtime | CPLEX is separately licensed/installed |
| `monolithic_qaoa` | Standard single-QPU QAOA | Qiskit and Qiskit Aer | All logical qubits are placed on one QPU |
| `abstract_distributed_qaoa` | Distributed allocation with abstract remote CX accounting | Qiskit and Qiskit Aer | Cross-QPU CX is represented by a direct simulator CX |
| `telegate_explicit_qaoa` | Explicit TeleGate-style remote CX simulation | Qiskit and Qiskit Aer | Adds communication qubits, measurements, resets, and feedforward |

## Choosing a mode

- Use `bruteforce` to verify small instances.
- Use `monolithic_qaoa` as the single-QPU quantum baseline.
- Use `abstract_distributed_qaoa` to isolate allocation and cross-QPU counts without explicit TeleGate overhead.
- Use `telegate_explicit_qaoa` to study the circuit overhead of the implemented remote-gate model.
- Use `miqp_cplex` when a local CPLEX installation is available and a scalable classical baseline is needed.

## Allocation

Distributed modes evaluate contiguous and graph-aware assignments. A manual assignment can also be supplied through `manual_assignment`. Candidates are ranked first by the number of cross-QPU quadratic terms and then by the estimated remote-CX count.
