# QUBO Alignment Command Log

Every shell command executed for the QUBO alignment task is listed below with raw stdout as captured.

## 1
Command:
```powershell
Get-Location
```
Stdout:
```text

Path                                                             
----                                                             
<REPO_ROOT>


```

## 2
Command:
```powershell
Test-Path -LiteralPath 'outputs\qubo_alignment.json'
```
Stdout:
```text
False
```

## 3
Command:
```powershell
Get-Content -LiteralPath 'outputs\qaoa_results_k50.json' | Select-Object -First 45
```
Stdout:
```text
{
  "generated_at": "2026-06-27T03:51:48.899690+00:00",
  "method": "QAOA",
  "requested_config": {
    "reps": 2,
    "shots": 1024,
    "k": 50,
    "seed": 42,
    "maxiter": 35
  },
  "actual_config": {
    "reps": 2,
    "shots": 1024,
    "k": 50,
    "seed": 42,
    "maxiter": 35
  },
  "reason_for_cap": null,
  "backend": "Qiskit StatevectorSampler",
  "used_qiskit": true,
  "reps": 2,
  "shots": 1024,
  "k": 50,
  "seed": 42,
  "maxiter": 35,
  "elapsed_seconds": 331.151,
  "unique_failures": 5,
  "total_unique_failures": 5,
  "cumulative_failures": [
    0,
    0,
    0,
    1,
    2,
    2,
    2,
    2,
    2,
    3,
    3,
    3,
    4,
    4,
    4,
    4,
```

## 4
Command:
```powershell
rg -n "def build_qubo_matrix|def _build_surrogate_qubo_matrix|load_empirical_failure_probabilities|FAILURE_STATES_PATH|failure_states_path.exists|return _build_surrogate|return qubo" phase2_qaoa\qubo_encoder.py
```
Stdout:
```text
30:FAILURE_STATES_PATH = ROOT / "outputs" / "failure_states.json"
149:def load_empirical_failure_probabilities(
150:    failure_states_path: Path = FAILURE_STATES_PATH,
156:    if not failure_states_path.exists():
204:def _build_surrogate_qubo_matrix() -> np.ndarray:
234:    return qubo
237:def build_qubo_matrix() -> np.ndarray:
243:    empirical_probabilities = load_empirical_failure_probabilities()
245:        return _build_surrogate_qubo_matrix()
268:    return qubo
313:    return qubo_to_hamiltonian(qubo), qubo
```

## 5
Command:
```powershell
rg -n "build_qubo_matrix|get_hamiltonian|QAOAExplorer|self.qubo|hamiltonian|qaoa_ansatz|actual_config|output" phase2_qaoa\qaoa_runner.py
```
Stdout:
```text
31:    from qiskit.circuit.library import qaoa_ansatz
38:    qaoa_ansatz = None
51:    get_hamiltonian,
54:OUT = ROOT / "outputs"
137:    def draw(self, output: str = "text", fold: int = 80) -> str:
138:        del output, fold
152:class QAOAExplorer:
167:        self.hamiltonian, self.qubo = get_hamiltonian()
175:            self.circuit = qaoa_ansatz(
176:                cost_operator=self.hamiltonian,
203:    def _hamiltonian_cost(self, bitstring: str) -> float:
206:        for label, coeff in zip(self.hamiltonian.paulis, self.hamiltonian.coeffs):
222:                - 0.15 * self._hamiltonian_cost(state["bitstring"])
251:            expected_cost += (count / total_shots) * self._hamiltonian_cost(bitstring)
290:    def run(self, k_iterations: int = 50, output_path: str | Path | None = OUT / "qaoa_results.json") -> Dict[str, Any]:
314:            "circuit_text": str(self.circuit.draw(output="text", fold=80)),
317:        if output_path is not None:
318:            out_path = Path(output_path)
332:    parser.add_argument("--output", type=str, default=str(OUT / "qaoa_results.json"))
338:    text = str(circuit.draw(output="text", fold=80))
344:    explorer = QAOAExplorer(reps=args.reps, shots=args.shots, seed=args.seed)
360:    results = explorer.run(k_iterations=args.k, output_path=args.output)
362:    print(f"Results written to   : {args.output}")
```

## 6
Command:
```powershell
& '.\.venv\Scripts\python.exe' -c "import sklearn; print(sklearn.__version__)"
```
Stdout:
```text
```

## 7
Command:
```powershell
Get-Item -LiteralPath 'outputs\failure_states.json','outputs\qaoa_results_k50.json' | Format-List Name,Length,LastWriteTime
```
Stdout:
```text


Name          : failure_states.json
Length        : 11656
LastWriteTime : 19-05-2026 11:18:07

Name          : qaoa_results_k50.json
Length        : 3436
LastWriteTime : 27-06-2026 09:21:48


```

## 8
Command:
```powershell
Get-Content -LiteralPath 'phase2_qaoa\qubo_encoder.py' | Select-Object -Skip 140 -First 140
```
Stdout:
```text
    return (
        PARAM_GRID["int_heading"].index(float(params["int_heading"])),
        PARAM_GRID["int_altitude"].index(float(params["int_altitude"])),
        PARAM_GRID["int_speed"].index(float(params["int_speed"])),
        PARAM_GRID["int_x_offset"].index(float(params["int_x_offset"])),
    )


def load_empirical_failure_probabilities(
    failure_states_path: Path = FAILURE_STATES_PATH,
) -> Dict[str, float]:
    """
    Load empirical failure frequencies exported by phase1/evaluate.py and map
    them onto the 8-bit Phase 2 state encoding.
    """
    if not failure_states_path.exists():
        return {}

    payload = json.loads(failure_states_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("failure_states", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    counts: Dict[Tuple[int, int, int, int], int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("failure_type") not in {"separation_loss", "near_miss"}:
            continue
        if "indices" in row and len(row["indices"]) == 4:
            indices = tuple(int(value) for value in row["indices"])
        else:
            params = row.get("params")
            if not isinstance(params, dict):
                continue
            indices = _state_indices_from_params(params)
        counts[indices] = counts.get(indices, 0) + 1

    total_failures = sum(counts.values())
    if total_failures <= 0:
        return {}

    return {
        encode_state_indices(indices): count / total_failures
        for indices, count in counts.items()
    }


def _apply_one_hot_regularization(qubo: np.ndarray) -> None:
    # Adds a penalty when both bits of the same parameter's 2-bit encoding are set.
    # We write 0.12 to both Q[first,second] and Q[second,first] here.
    # qubo_to_hamiltonian() symmetrizes via (Q[i,j]+Q[j,i])/2 = 0.12 (correct).
    # Net one-hot penalty in the Hamiltonian coupling: 0.12.
    for param_index in range(4):
        first = param_index * 2
        second = first + 1
        qubo[first, second] += 0.12
        qubo[second, first] += 0.12


def _build_surrogate_qubo_matrix() -> np.ndarray:
    """
    Legacy 8x8 QUBO matrix based on the smooth hand-shaped surrogate.
    Kept as a fallback until empirical data has been generated.
    """
    qubo = np.zeros((N_QUBITS, N_QUBITS), dtype=float)
    bit_scores = np.zeros(N_QUBITS, dtype=float)
    pair_scores = np.zeros((N_QUBITS, N_QUBITS), dtype=float)
    states = enumerate_parameter_states()

    for state in states:
        score = estimate_failure_score(state)
        bits = encode_state_indices(_state_indices_from_params(state))[::-1]
        active = [index for index, bit in enumerate(bits) if bit == "1"]
        for bit_index in active:
            bit_scores[bit_index] += score
        for left in range(len(active)):
            for right in range(left + 1, len(active)):
                i = active[left]
                j = active[right]
                pair_scores[i, j] += score

    total_states = float(len(states))
    qubo[np.diag_indices(N_QUBITS)] = -(bit_scores / total_states)

    for i in range(N_QUBITS):
        for j in range(i + 1, N_QUBITS):
            qubo[i, j] = qubo[j, i] = -(pair_scores[i, j] / total_states) * 0.35

    _apply_one_hot_regularization(qubo)
    return qubo


def build_qubo_matrix() -> np.ndarray:
    """
    Builds an 8x8 QUBO matrix from empirical failure frequencies measured in
    Phase 1. If no empirical file exists yet, falls back to the original
    surrogate-risk formulation so the repo still runs end-to-end.
    """
    empirical_probabilities = load_empirical_failure_probabilities()
    if not empirical_probabilities:
        return _build_surrogate_qubo_matrix()

    qubo = np.zeros((N_QUBITS, N_QUBITS), dtype=float)
    bit_scores = np.zeros(N_QUBITS, dtype=float)
    pair_scores = np.zeros((N_QUBITS, N_QUBITS), dtype=float)

    for bitstring, probability in empirical_probabilities.items():
        bits = bitstring[::-1]
        active = [index for index, bit in enumerate(bits) if bit == "1"]
        for bit_index in active:
            bit_scores[bit_index] += probability
        for left in range(len(active)):
            for right in range(left + 1, len(active)):
                i = active[left]
                j = active[right]
                pair_scores[i, j] += probability

    qubo[np.diag_indices(N_QUBITS)] = -bit_scores
    for i in range(N_QUBITS):
        for j in range(i + 1, N_QUBITS):
            qubo[i, j] = qubo[j, i] = -pair_scores[i, j] * 0.35

    _apply_one_hot_regularization(qubo)
    return qubo


def qubo_to_hamiltonian(qubo: np.ndarray):
```

## 9
Command:
```powershell
Get-Content -LiteralPath 'phase2_qaoa\qaoa_runner.py' | Select-Object -Skip 150 -First 130
```
Stdout:
```text

class QAOAExplorer:
    """
    Runs a Qiskit-first QAOA exploration loop with a deterministic local fallback.

    Classical optimizer: COBYLA with maxiter=35 per iteration.
    """

    def __init__(self, reps: int = 2, shots: int = 1024, seed: int = 42) -> None:
        self.reps = reps
        # shots=1024 is the paper's reported value.
        # The portal benchmark (app.py:run_assurance_benchmark) uses shots=256, k=10
        # for speed. For paper-replicating results, use: --shots 1024 --k 50
        self.shots = shots
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.hamiltonian, self.qubo = get_hamiltonian()
        self.all_failures: List[Dict[str, Any]] = []
        self.iteration_counts: List[int] = []
        self._known_keys: set[str] = set()
        self._evaluated_states = self._build_state_catalog()

        if QISKIT_AVAILABLE:
            algorithm_globals.random_seed = seed
            self.circuit = qaoa_ansatz(
                cost_operator=self.hamiltonian,
                reps=reps,
                insert_barriers=True,
                flatten=True,
            )
            self.sampler = StatevectorSampler(seed=seed)
            self.backend_name = "Qiskit StatevectorSampler"
        else:
            self.circuit = FallbackCircuit(num_qubits=N_QUBITS, reps=reps)
            self.sampler = None
            self.backend_name = "Deterministic fallback sampler"

    def _build_state_catalog(self) -> List[Dict[str, Any]]:
        catalog: List[Dict[str, Any]] = []
        for state in enumerate_parameter_states():
            heading_index = PARAM_GRID["int_heading"].index(state["int_heading"])
            altitude_index = PARAM_GRID["int_altitude"].index(state["int_altitude"])
            speed_index = PARAM_GRID["int_speed"].index(state["int_speed"])
            offset_index = PARAM_GRID["int_x_offset"].index(state["int_x_offset"])
            bitstring = encode_state_indices(
                [heading_index, altitude_index, speed_index, offset_index]
            )
            evaluation = evaluate_state(state)
            evaluation["bitstring"] = bitstring
            catalog.append(evaluation)
        return catalog

    def _hamiltonian_cost(self, bitstring: str) -> float:
        z_values = [1 - 2 * int(bit) for bit in bitstring]
        total = 0.0
        for label, coeff in zip(self.hamiltonian.paulis, self.hamiltonian.coeffs):
            term = 1.0
            for index, pauli in enumerate(reversed(str(label))):
                if pauli == "Z":
                    term *= z_values[index]
            total += float(coeff.real) * term
        return total

    def _fallback_counts(self, params: np.ndarray) -> Dict[str, int]:
        logits: List[float] = []
        for index, state in enumerate(self._evaluated_states):
            jitter = 0.03 * math.sin(float(np.sum(params)) + index)
            score = (
                1.6 * state["surrogate_score"]
                + 0.7 * float(state["failure"])
                + jitter
                - 0.15 * self._hamiltonian_cost(state["bitstring"])
            )
            logits.append(score)
        centered = np.array(logits) - max(logits)
        probs = np.exp(centered)
        probs = probs / probs.sum()
        sampled = self.rng.multinomial(self.shots, probs)
        return {
            state["bitstring"]: int(count)
            for state, count in zip(self._evaluated_states, sampled)
            if count > 0
        }

    def _qiskit_counts(self, params: np.ndarray) -> Dict[str, int]:
        bound = self.circuit.assign_parameters(params)
        measured = bound.measure_all(inplace=False)
        result = self.sampler.run([measured], shots=self.shots).result()
        return result[0].data.meas.get_counts()

    def _sample_counts(self, params: np.ndarray) -> Dict[str, int]:
        if QISKIT_AVAILABLE:
            return self._qiskit_counts(params)
        return self._fallback_counts(params)

    def _evaluate_circuit(self, params: np.ndarray) -> float:
        counts = self._sample_counts(params)
        total_shots = max(sum(counts.values()), 1)
        expected_cost = 0.0
        for bitstring, count in counts.items():
            expected_cost += (count / total_shots) * self._hamiltonian_cost(bitstring)
        return expected_cost

    def _optimize_params(self, iteration: int) -> np.ndarray:
        if QISKIT_AVAILABLE and COBYLA is not None:
            init = self.rng.uniform(0, 2 * math.pi, self.circuit.num_parameters)
            # COBYLA maxiter=35: classical outer-loop optimization budget per QAOA iteration.
            # The paper previously claimed 50 evaluations; the correct value is 35.
            result = COBYLA(maxiter=35).minimize(fun=self._evaluate_circuit, x0=init)
            return np.array(result.x, dtype=float)

```

## 10
Command:
```powershell
& '.\.venv\Scripts\python.exe' -m pip install scikit-learn
```
Stdout:
```text
Collecting scikit-learn
  Using cached scikit_learn-1.9.0-cp312-cp312-win_amd64.whl.metadata (11 kB)
Requirement already satisfied: numpy>=1.24.1 in <REPO_ROOT>\.venv\lib\site-packages (from scikit-learn) (2.4.4)
Requirement already satisfied: scipy>=1.10.0 in <REPO_ROOT>\.venv\lib\site-packages (from scikit-learn) (1.17.1)
Collecting joblib>=1.4.0 (from scikit-learn)
  Using cached joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
Collecting narwhals>=2.0.1 (from scikit-learn)
  Using cached narwhals-2.22.1-py3-none-any.whl.metadata (15 kB)
Collecting threadpoolctl>=3.5.0 (from scikit-learn)
  Using cached threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Using cached scikit_learn-1.9.0-cp312-cp312-win_amd64.whl (8.2 MB)
Using cached joblib-1.5.3-py3-none-any.whl (309 kB)
Using cached narwhals-2.22.1-py3-none-any.whl (454 kB)
Using cached threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Installing collected packages: threadpoolctl, narwhals, joblib, scikit-learn
Successfully installed joblib-1.5.3 narwhals-2.22.1 scikit-learn-1.9.0 threadpoolctl-3.6.0

[notice] A new release of pip is available: 25.0.1 -> 26.1.2
[notice] To update, run: <REPO_ROOT>\.venv\Scripts\python.exe -m pip install --upgrade pip
```

## 11
Command:
```powershell
& '.\.venv\Scripts\python.exe' -m py_compile analysis\qubo_alignment.py
```
Stdout:
```text
```

## 12
Command:
```powershell
& '.\.venv\Scripts\python.exe' -c "from phase2_qaoa.qubo_encoder import build_qubo_matrix, _build_surrogate_qubo_matrix, load_empirical_failure_probabilities; import numpy as np; empirical=load_empirical_failure_probabilities(); current=build_qubo_matrix(); surrogate=_build_surrogate_qubo_matrix(); print('empirical_probability_count', len(empirical)); print('current_equals_surrogate', bool(np.allclose(current, surrogate))); print('current_is_empirical_by_nonempty_probabilities', bool(empirical)); print('current_matrix_shape', current.shape)"
```
Stdout:
```text
empirical_probability_count 12
current_equals_surrogate False
current_is_empirical_by_nonempty_probabilities True
current_matrix_shape (8, 8)
```

## 13
Command:
```powershell
& '.\.venv\Scripts\python.exe' analysis\qubo_alignment.py
```
Stdout:
```text
wrote outputs\qubo_alignment.json
state_count=256
geometric_failure_count=18
policy_failure_count=12
empirical_probability_count=12
qubo_type,label_set,r,auc,hamiltonian_auc,auc_match
empirical_build_qubo_matrix,geometric_labels,-0.190582082674,0.753151260504,0.526610644258,False
empirical_build_qubo_matrix,policy_labels,-0.282699056948,0.844091530055,0.585040983607,False
surrogate_fallback,geometric_labels,-0.030936522311,0.557189542484,0.537815126050,False
surrogate_fallback,policy_labels,-0.073508087721,0.596482240437,0.584016393443,False
```

## 14
Command:
```powershell
Get-Content -LiteralPath 'phase2_qaoa\qubo_encoder.py' | Select-Object -Skip 270 -First 45
```
Stdout:
```text
def qubo_to_hamiltonian(qubo: np.ndarray) -> SparsePauliOp:
    """Converts a QUBO matrix into a diagonal Ising Hamiltonian."""
    num_qubits = qubo.shape[0]
    terms: List[Tuple[str, complex]] = []
    constant = 0.0

    for i in range(num_qubits):
        constant += qubo[i, i] / 2.0
        linear_coeff = -qubo[i, i] / 2.0
        if abs(linear_coeff) > 1e-12:
            label = "I" * (num_qubits - i - 1) + "Z" + "I" * i
            terms.append((label, complex(linear_coeff)))

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            q_value = (qubo[i, j] + qubo[j, i]) / 2.0
            if abs(q_value) <= 1e-12:
                continue
            constant += q_value / 4.0
            linear_i = -q_value / 4.0
            linear_j = -q_value / 4.0
            zz_coeff = q_value / 4.0

            label_i = "I" * (num_qubits - i - 1) + "Z" + "I" * i
            label_j = "I" * (num_qubits - j - 1) + "Z" + "I" * j
            zz_chars = list("I" * num_qubits)
            zz_chars[num_qubits - i - 1] = "Z"
            zz_chars[num_qubits - j - 1] = "Z"
            terms.extend(
                [
                    (label_i, complex(linear_i)),
                    (label_j, complex(linear_j)),
                    ("".join(zz_chars), complex(zz_coeff)),
                ]
            )

    terms.append(("I" * num_qubits, complex(constant)))
    return SparsePauliOp.from_list(terms)


def get_hamiltonian() -> Tuple[SparsePauliOp, np.ndarray]:
    qubo = build_qubo_matrix()
    return qubo_to_hamiltonian(qubo), qubo


```

## 15
Command:
```powershell
& '.\.venv\Scripts\python.exe' -c "... direct QUBO-vector Hamiltonian delta check ..."
```
Stdout:
```text
max_centered_abs 1.8297115384615403
constant 0.5699038461538461
```

## 16
Command:
```powershell
& '.\.venv\Scripts\python.exe' -c "... upper-triangular objective vs xTQx Hamiltonian delta check ..."
```
Stdout:
```text
upper max_centered_abs 1.8067145002298446e-15 constant 3.0357660829594124e-17
xtqx max_centered_abs 1.8297115384615403 constant 0.5699038461538461
```

## 17
Command:
```powershell
& '.\.venv\Scripts\python.exe' -c "... compute xtqx and upper-triangular metrics ..."
```
Stdout:
```text
empirical geometric xtqx_r -0.190582082674 xtqx_auc 0.753151260504 upper_r -0.19118633136 upper_auc 0.749883286648
empirical policy xtqx_r -0.282699056948 xtqx_auc 0.844091530055 upper_r -0.275889663663 upper_auc 0.844091530055
surrogate geometric xtqx_r -0.030936522311 xtqx_auc 0.557189542484 upper_r -0.034595694474 upper_auc 0.557306255836
surrogate policy xtqx_r -0.073508087721 xtqx_auc 0.596482240437 upper_r -0.082016244998 upper_auc 0.595969945355
```

## 18
Command:
```powershell
& '.\.venv\Scripts\python.exe' -m py_compile analysis\qubo_alignment.py
```
Stdout:
```text
```

## 19
Command:
```powershell
Test-Path -LiteralPath 'outputs\qubo_alignment_corrected.json'
```
Stdout:
```text
False
```

## 20
Command:
```powershell
& '.\.venv\Scripts\python.exe' analysis\qubo_alignment.py --output outputs\qubo_alignment_corrected.json
```
Stdout:
```text
wrote outputs\qubo_alignment_corrected.json
state_count=256
geometric_failure_count=18
policy_failure_count=12
empirical_probability_count=12
qubo_type,label_set,xtqx_r,xtqx_auc,hamiltonian_auc,xtqx_hamiltonian_auc_match,upper_hamiltonian_auc_match
empirical_build_qubo_matrix,geometric_labels,-0.190582082674,0.753151260504,0.749883286648,False,False
empirical_build_qubo_matrix,policy_labels,-0.282699056948,0.844091530055,0.844091530055,True,True
surrogate_fallback,geometric_labels,-0.030936522311,0.557189542484,0.557306255836,False,True
surrogate_fallback,policy_labels,-0.073508087721,0.596482240437,0.595628415301,False,False
```

## 21
Command:
```powershell
& '.\.venv\Scripts\python.exe' -c "import json; p='outputs/qubo_alignment_corrected.json'; data=json.load(open(p)); ..."
```
Stdout:
```text
empirical_build_qubo_matrix geometric_labels upper_auc 0.7502334267040149 ham_auc 0.7498832866479925 centered_max 2.4771851236948805e-15
empirical_build_qubo_matrix policy_labels upper_auc 0.8440915300546448 ham_auc 0.8440915300546448 centered_max 2.4771851236948805e-15
surrogate_fallback geometric_labels upper_auc 0.5573062558356676 ham_auc 0.5573062558356676 centered_max 2.297858084365778e-15
surrogate_fallback policy_labels upper_auc 0.5957991803278689 ham_auc 0.5956284153005464 centered_max 2.297858084365778e-15
```

## 22
Command:
```powershell
& '.\.venv\Scripts\python.exe' -c "import numpy as np; from phase2_qaoa.qubo_encoder import build_qubo_matrix, qubo_to_hamiltonian, enumerate_parameter_states, encode_state_indices, PARAM_GRID, PARAM_ORDER; ..."
```
Stdout:
```text
<class 'qiskit.quantum_info.operators.symplectic.sparse_pauli_op.SparsePauliOp'>
74 74
IIIIIIZI (0.07692307692307693+0j)
```

## 23
Command:
```powershell
& '.\.venv\Scripts\python.exe' -c "... rounded upper/Hamiltonian AUC check ..."
```
Stdout:
```text
emp geo 0.7498832866479925 0.7498832866479925
emp pol 0.8440915300546448 0.8440915300546448
surr geo 0.5575396825396826 0.5575396825396826
surr pol 0.5961407103825136 0.5961407103825136
```

## 24
Command:
```powershell
Test-Path -LiteralPath 'outputs\qubo_alignment_final.json'
```
Stdout:
```text
False
```

## 25
Command:
```powershell
& '.\.venv\Scripts\python.exe' -m py_compile analysis\qubo_alignment.py
```
Stdout:
```text
```

## 26
Command:
```powershell
& '.\.venv\Scripts\python.exe' analysis\qubo_alignment.py --output outputs\qubo_alignment_final.json
```
Stdout:
```text
wrote outputs\qubo_alignment_final.json
state_count=256
geometric_failure_count=18
policy_failure_count=12
empirical_probability_count=12
qubo_type,label_set,xtqx_r,xtqx_auc,hamiltonian_auc,xtqx_hamiltonian_auc_match,upper_hamiltonian_auc_match
empirical_build_qubo_matrix,geometric_labels,-0.190582082674,0.753151260504,0.749883286648,False,True
empirical_build_qubo_matrix,policy_labels,-0.282699056948,0.844091530055,0.844091530055,True,True
surrogate_fallback,geometric_labels,-0.030936522311,0.557189542484,0.557539682540,False,True
surrogate_fallback,policy_labels,-0.073508087721,0.596482240437,0.596140710383,False,True
```
