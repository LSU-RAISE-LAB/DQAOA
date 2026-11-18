# DQAOA-QUBO Solver Studio

Streamlit GUI for classical references, monolithic QAOA and distributed QAOA.
All quantum circuits run on the classical Qiskit Aer simulator. Reported runtime
is simulation and software time, not execution time on networked quantum hardware.

## Windows: double-click installation and launch

Extract the full archive, then open the `gui` folder:

1. Double-click **Install Requirements.cmd** once. It finds Python 3.12 or
   installs it for your account through WinGet, creates a private `.venv`,
   installs dependencies and verifies imports.
2. Double-click **Open App.cmd**. The interface opens in your default browser
   when the local server is ready. No terminal commands or code editor are needed.
3. On later visits, only use **Open App.cmd**.

Keep the launcher window open during use. Stop any active solver using the GUI,
then press Ctrl+C in the launcher window to stop the server. Closing the browser
alone does not stop the server. Setup requires internet; if WinGet is unavailable,
it opens the Python download page and asks you to install Python 3.12 first.
The private environment does not modify your existing Conda environment.
Optional licensed CPLEX is not installed automatically.

The `.cmd`/PowerShell launchers target Windows and have not been executed on
Windows in this Linux validation environment. Browser-launch orchestration is
checked with mocks; the GUI was previously tested with Streamlit AppTest.

## Manual installation (alternative)


Python 3.12 was used for the release checks. From the repository root:

```bash
cd gui
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then install and launch:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).
An existing compatible Conda environment can be used instead of creating a venv.
Dependencies in requirements.txt are pinned to the versions used for these checks;
they are not a claim about the environment used for historical paper results.

## Run a built-in case

1. Choose **Built-in example** as the input source; the initial Manual editor
   otherwise contains zeros.
2. Select the 6-variable example and the desired solver modes.
3. Use two QPUs with capacities `[3, 3]` for distributed QAOA.
4. Choose a preset, inspect the resolved configuration, and click Run.
5. Wait for **Show results** to become enabled, then click it.

The status panel checks for completion every five seconds without refreshing
all input controls. Stop current run terminates the active solver process.
Result files and process logs are written under `outputs/` and ignored by Git.

The built-in 12-variable case can use `[6, 6]`; the 15-variable case can use
`[5, 5, 5]`. Capacities refer to logical data variables. The explicit distributed
simulation uses two shared, reusable communication ancillas for serial remote
operations; it does not model a physical network with only two communication
qubits in total. Network latency, entanglement-generation delays and noise are
not modeled.

## Presets and references

- Fast demo: depth 1, 256 training shots, 128 final shots, 4 optimizer iterations,
  2 random starts, no warm-start perturbations.
- Balanced research: depth 2, 512 training shots, 256 final shots, 20 iterations,
  4 random starts, 1 warm-start perturbation.
- Accuracy check: depth 2, 512 training shots, 512 final shots, 8 iterations,
  12 random starts, 3 warm-start perturbations.

Presets are convenience settings, not guarantees of solution quality or exact
reproductions of the manuscript experiments. Record the seed, inputs, resolved
settings and software version when reporting results. More depth, shots and
restarts can substantially increase distributed simulation time.

`select_by_exact_when_available=False` is set in both:

- `dqaoa_streamlit_demo/solver_backend.py` (MultiStartConfig default)
- `dqaoa_streamlit_demo/job_runner.py` (explicit call argument)

A classical reference may be computed for comparison, but is not used to rank
QAOA candidates with this option disabled. To omit automatic reference
computation entirely, uncheck **Compute brute-force exact baseline for small n**
and deselect classical solver modes. Classical references can be run afterward.

This release preserves the existing non-reference ranking: best sampled cost,
then sampled probability mass at that cost, then incumbent-bitstring frequency,
followed by existing mean-cost/distribution-gap/objective tie-breakers. Training
still minimizes mean cost. Disabling reference selection does not alter these
rules. The existing final sampling is not an additional independent evaluation
after candidate selection; publication validation should include that check.

## Optional CPLEX

```bash
python -m pip install -r requirements-optional-cplex.txt
```

The MIQP mode additionally requires a compatible installed and licensed IBM CPLEX
runtime. It is not needed for brute force or either QAOA mode.

## Files and backend version

- `app.py`: Streamlit interface.
- `dqaoa_streamlit_demo/job_runner.py`: background process entry point.
- `dqaoa_streamlit_demo/solver_backend.py`: bundled solver implementation.
- `examples/`: built-in CSV inputs (constant offsets are in app.py: 0.5, 0.25,
  and 0.35 for the 6-, 12-, and 15-variable cases respectively).
- `.streamlit/config.toml`: display configuration.

The bundled solver starts from repository commit
`1a6cbfebb0b05e1b409d7704512e621935f54715`, with a local parameter-binding fix
for circuits whose cost parameters disappear (including all-zero QUBOs).
The app imports this bundled file, not the solver at the repository root.
Keep these copies synchronized when making future solver changes.

## Validation scope

Functional checks in Python 3.12 with the pinned dependencies included the
all-zero manual input and the built-in 6-variable example through Run, background
execution, Show results and dashboard rendering. See the accompanying release
notes for earlier small-case algorithm checks. These checks are not a circuit
equivalence proof or a statistical benchmark.

Installer fix: a missing Python 3.12 runtime reported by the Python launcher
now falls through to installation instead of aborting setup. Native probe stderr
is captured separately from PowerShell errors.

## Results display

Mean-cost and p_best metrics are omitted from the dashboard tables, summary
cards, plots and downloadable result JSON/CSV. Filtering is applied to presentation
copies only; training, candidate ranking and internal solver output files remain
unchanged. Individual elite-bitstring probabilities are still shown.
