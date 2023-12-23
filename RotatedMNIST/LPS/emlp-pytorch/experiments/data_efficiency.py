""" Data efficiency experiment for regression datasets. """
import copy
from oil.tuning.study import train_trial, Study
from oil.tuning.args import argupdated_config
from emlp_pytorch.datasets import O5Synthetic, ParticleInteraction, Inertia
from emlp_pytorch.groups import SO, O, O13, SO13, SO13p
from emlp_pytorch.nn import MLP, EMLP
from .train_regression import makeTrainer


if __name__ == "__main__":
    Trial = train_trial(makeTrainer)
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    name = 'data_efficiency_nobn'
    config_spec['ndata'] = 30000+5000+1000
    config_spec['log_level'] = 'warning'
    # Run MLP baseline on datasets
    config_spec.update({
        # [O5Synthetic,Inertia,ParticleInteraction],
        'dataset': ParticleInteraction,
        'network': MLP, 'aug': [False, True],
        'num_epochs': (lambda cfg: min(int(30*30000/cfg['split']['train']), 1000)),
        'split': {'train': [30, 100, 300, 1000, 3000, 10000, 30000], 'test': 5000, 'val': 1000},
    })
    config_spec = argupdated_config(config_spec)
    name = f"{name}_{config_spec['dataset']}"
    thestudy = Study(Trial, {}, study_name=name,
                     base_log_dir=config_spec['trainer_config'].get('log_dir', None))
    thestudy.run(num_trials=-3, new_config_spec=config_spec, ordered=True)

    # Now run the EMLP (with appropriate group) on the datasets
    config_spec['network'] = EMLP
    config_spec['aug'] = False
    groups = {O5Synthetic: [SO(5), O(5)], Inertia: [SO(3), O(
        3)], ParticleInteraction: [SO13p(), SO13(), O13()]}
    config_spec['net_config']['group'] = groups[config_spec['dataset']]
    thestudy.run(num_trials=-3, new_config_spec=config_spec, ordered=True)
    print(thestudy.results_df())
