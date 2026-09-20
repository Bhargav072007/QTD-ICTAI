import numpy as np
import pytest
from phase3_quantum_tree.pipeline import run_quantum_tree_pipeline
from run_positive_control import make_oracle, sign_test


def test_oracle_permutation_preserves_multiset_and_gate():
    args=dict(scores=np.zeros(12), teacher_probs=np.linspace(0,1,12),
              labels=np.array([0,1]*6), gated=np.array([False]*6+[True]*6),seed=42)
    for strength in (1., .5, .25):
        signal=make_oracle(strength,False)(**args)
        shuffled=make_oracle(strength,True)(**args)
        assert np.array_equal(np.sort(signal[args['gated']]),np.sort(shuffled[args['gated']]))
        assert np.all(shuffled[~args['gated']]==0)
        if strength==1: assert np.array_equal(signal[args['gated']],args['labels'][args['gated']])


def test_sign_test_excludes_ties():
    value=sign_test([1]*8+[-1,0])
    assert value['positive_count']==8 and value['zero_count']==1
    assert value['sign_test_p']==round(20/512,6)


def test_qtd_total_calls_and_no_duplicate_reporting():
    r=run_quantum_tree_pipeline(seed=42,shots=64,k_iterations=50,output_path=None)
    c=r['resources']
    assert c['training_label_acquisition_calls']==256
    assert c['search_simulator_calls']==50
    assert c['reporting_simulator_calls']==0
    assert c['simulator_calls_total_this_run']==306
    assert c['simulator_unique_states_this_run']==256
    assert c['sampler_calls']==64
    assert c['circuit_shots']==64*64


def test_qaoa_refuses_overwrite_before_running(tmp_path,monkeypatch):
    from phase2_qaoa.qaoa_runner import QAOAExplorer
    p=tmp_path/'result.json';p.write_text('historical')
    runner=QAOAExplorer()
    monkeypatch.setattr(runner,'run_iteration',lambda _: pytest.fail('should preflight output'))
    with pytest.raises(FileExistsError):runner.run(1,p)
    assert p.read_text()=='historical'


def test_diagnostic_auc_does_not_split_numerical_ties():
    from analysis.qubo_diagnostics import rank_auc
    assert rank_auc(np.array([1., 1. + 1e-15]),np.array([1,0])) == .5
    assert rank_auc(np.array([1., 1. + 1e-6]),np.array([1,0])) == 0.
