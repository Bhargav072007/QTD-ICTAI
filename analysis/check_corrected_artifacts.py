"""Read-only checks of corrected results derived from records and code, not paper constants."""
from pathlib import Path
import sys,argparse,json,hashlib
from math import comb
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from phase2_qaoa.qaoa_runner import QAOAExplorer,evaluate_state
from phase2_qaoa.qubo_encoder import objective_value,decode_bitstring,enumerate_parameter_states,encode_state_indices,_state_indices_from_params


def auc(curve):
    return sum((a+b)/2 for a,b in zip([0]+curve[:-1],curve))


def verify_paired(node, path='root'):
    checked=0
    if isinstance(node,dict):
        if isinstance(node.get('per_seed'),list) and node['per_seed'] and all(isinstance(x,(int,float)) for x in node['per_seed']) and 'aggregate' in node:
            a=np.array(node['per_seed'],float);agg=node['aggregate']
            assert abs(a.mean()-agg['mean'])<1e-6,(path,'mean')
            assert abs(a.std()-agg['std'])<1e-6,(path,'population SD')
            if 'bootstrap_mean_95_ci' in node:
                ci=node['bootstrap_mean_95_ci'];rng=np.random.default_rng(ci.get('seed',0))
                boot=rng.choice(a,size=(ci.get('resamples',10000),len(a)),replace=True).mean(1)
                assert np.allclose(np.quantile(boot,[.025,.975]),[ci['ci_95_low'],ci['ci_95_high']],atol=1e-6,rtol=0),(path,'bootstrap')
            if 'exact_tests' in node:
                tests=node['exact_tests'];pos=int(sum(a>0));neg=int(sum(a<0));n=pos+neg
                p=min(1.,2*sum(comb(n,i) for i in range(min(pos,neg)+1))/2**n) if n else None
                assert [tests[k] for k in ['positive_count','negative_count','zero_count']]==[pos,neg,len(a)-n],path
                if p is None: assert tests['sign_test_p'] is None,path
                else: assert abs(tests['sign_test_p']-p)<1e-6,path
            checked+=1
        for key,value in node.items():checked+=verify_paired(value,path+'.'+key)
    elif isinstance(node,list):
        for i,value in enumerate(node):checked+=verify_paired(value,path+f'[{i}]')
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    args=parser.parse_args();d=args.data_dir
    q=QAOAExplorer();diag=q.hamiltonian.to_matrix().diagonal().real
    for n in range(256):
        word=f'{n:08b}';x=np.array([int(c) for c in word[::-1]])
        assert abs(q._hamiltonian_cost(word)-diag[n])<1e-12,word
        assert abs(objective_value(q.qubo,x)-diag[n])<1e-12,word
        assert encode_state_indices(_state_indices_from_params(decode_bitstring(word)))==word
    payload=json.loads((d/'qaoa_multiseed.json').read_text())
    rows=payload['per_seed'];assert [r['seed'] for r in rows]==list(range(42,52))
    for row in rows:
        part=json.loads((d/'qaoa_multiseed_parts'/f"qaoa_seed_{row['seed']}.json").read_text())
        assert part['deterministic'] and part['record']==row
        assert part['code_commit']==payload['code_commit']
        for name,digest in part['provenance']['sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        curve=row['cumulative_failures'];assert len(curve)==payload['config']['k']
        assert curve==sorted(curve) and curve[-1]==row['unique_failures']
        assert auc(curve)==row['auc']
        assert next((i for i,v in enumerate(curve,1) if v),None)==row['first_failure']
        c=row['resources'];shots=payload['config']['shots']
        assert c['optimizer_evaluations']==sum(c['optimizer_nfev_per_iteration'])
        assert c['optimizer_shots']==shots*c['optimizer_evaluations']
        assert c['sampling_shots']==shots*c['sampling_calls']
        assert c['circuit_shots_total']==c['optimizer_shots']+c['sampling_shots']
        assert c['simulator_calls_total_this_run']==c['setup_catalog_calls']+c['search_simulator_calls']+c['reporting_simulator_calls']
        assert c['search_simulator_calls']==row['total_env_evaluations']
        assert c['search_unique_states']==row['unique_env_evaluations']
    for key,stat in payload['aggregate'].items():
        if isinstance(stat,dict) and 'mean' in stat:
            values=[r[key] for r in rows if r[key] is not None]
            assert np.allclose([np.mean(values),np.std(values)],[stat['mean'],stat['std']],atol=1e-6,rtol=0),key
    diagnostic=json.loads((d/'qubo_diagnostics.json').read_text())
    states=enumerate_parameter_states();labels=np.array([int(evaluate_state(s)['failure']) for s in states])
    energies=np.array([q._hamiltonian_cost(encode_state_indices(_state_indices_from_params(s))) for s in states])
    polynomial=np.array([objective_value(q.qubo,np.array([int(c) for c in encode_state_indices(_state_indices_from_params(s))[::-1]])) for s in states])
    assert np.allclose(energies,polynomial,rtol=0,atol=1e-12)
    cum=np.cumsum(labels[np.argsort(polynomial,kind='stable')])
    recorded=diagnostic['variants']['with_penalty']['upper_triangular_hamiltonian']
    assert cum[:50].tolist()==recorded['exact_enumeration_cumulative_failures_k50']
    for budget,count in recorded['failures_in_lowest_energy'].items():assert cum[int(budget)-1]==count
    count=0
    for name in ('mechanism_controls.json','positive_control.json'):
        count+=verify_paired(json.loads((ROOT/'outputs'/name).read_text()),name)
    print(f'ALL PASS: 256 objective/encoding states; {len(rows)} duplicate-verified seeds; resource accounting; comparator; {count} paired statistical records')


if __name__=='__main__':main()
