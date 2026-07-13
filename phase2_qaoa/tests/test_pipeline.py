from phase2_qaoa.monte_carlo import run_monte_carlo
from phase2_qaoa.qaoa_runner import QAOAExplorer, evaluate_state


def test_evaluate_state_flags_conflict() -> None:
    result = evaluate_state(
        {
            "int_heading": 90.0,
            "int_altitude": 31000.0,
            "int_speed": 6.0,
            "int_x_offset": 0.0,
        }
    )
    assert result["failure"] is True


def test_full_pipeline_returns_results_without_writing_outputs() -> None:
    qaoa = QAOAExplorer(reps=1, shots=64, seed=7).run(k_iterations=10, output_path=None)
    mc = run_monte_carlo(k_iterations=10, seed=7, output_path=None)

    assert qaoa["total_unique_failures"] >= 1
    assert mc["k_iterations"] == 10
