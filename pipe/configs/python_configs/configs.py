import json

import ml_collections
from ml_collections import ConfigDict


def empty_config():
    return ml_collections.ConfigDict()


def _convert_dict_to_python_config(dd, prefix="config"):
    print(f"{prefix} = ConfigDict()")

    for i in dd:
        if isinstance(dd[i], str):
            print(f"{prefix}.{i} = '{dd[i]}'")
        elif isinstance(dd[i], dict):
            _convert_dict_to_python_config(dd[i], prefix=f"{prefix}.{i}")
        else:
            print(f"{prefix}.{i} = {dd[i]}")


def _convert_json_to_to_python_config(path):
    with open(path, "r") as f:
        dd = json.load(f)
    _convert_dict_to_python_config(dd)


def get_all_options_config():
    """Simply converting all options to ml_collections.ConfigDict, Autogenerated"""
    config = ml_collections.ConfigDict()
    config.logdir = 'logs/'
    config.out_dir = 'results/'
    config.out_filename = 'wrn_16x4_p4_msnag_clone_ga_no_wd'
    config.distributed_backend = 'mpi'
    config.data_propagator = 'auto'
    config.statistics = 'cv'
    config.model = 'wrn_16x4_p4'
    config.dataset = 'cifar10'
    config.trainer = ml_collections.ConfigDict()
    config.trainer.type = 'cv'
    config.trainer.args = ml_collections.ConfigDict()
    config.trainer.args.max_grad_norm = 0.25
    config.trainer.args.always_calc_grad_norm = True
    config.bs_train = 128
    config.bs_test = 200
    config.num_data_workers = 6
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.type = 'sgd1'
    config.optimizer.args = ml_collections.ConfigDict()
    config.optimizer.args.lr = 0.1
    config.optimizer.args.weight_decay = 0.0005
    config.optimizer.args.momentum = 0.9
    config.optimizer.args.nesterov = False
    config.lr_scheduler = ml_collections.ConfigDict()
    config.lr_scheduler.type = 'get_multi_step_lr_schedule_with_warmup'
    config.lr_scheduler.preproc_args = ml_collections.ConfigDict()
    config.lr_scheduler.preproc_args.num_training_steps = 'epochs_to_steps'
    config.lr_scheduler.args = ml_collections.ConfigDict()
    config.lr_scheduler.args.num_warmup_steps = 5
    config.lr_scheduler.args.milestones = [60, 120, 160]
    config.lr_scheduler.args.gamma = 0.2
    config.lr_scheduler.args.last_epoch = -1
    config.weight_prediction = ml_collections.ConfigDict()
    config.weight_prediction.type = 'msnag'
    config.weight_prediction.args = ml_collections.ConfigDict()
    config.weight_prediction.args.pred_mem = 'clone'
    config.weight_prediction.args.nag_with_predictor = True
    config.weight_prediction.args.sched_aware = False
    config.gap_aware = ml_collections.ConfigDict()
    config.gap_aware.type = 'sgd1'
    config.gap_aware.policy = 'almost_last_partition'
    config.gap_aware.args = ml_collections.ConfigDict()
    config.gap_aware.args.big_gamma = 0.999
    config.gap_aware.args.epsilon = 1e-08
    config.epochs = 200
    config.steps = -1
    config.seed = 42
    config.num_chunks = 4
    config.verbose_comm = True
    config.flush_rate = -1
    config.train_batches_limit = -1
    config.test_batches_limit = -1
    config.weight_stashing = True
    config.work_scheduler = '1F1B'
    config.cpu = False
    config.seed_from_cmd = True
    config.bs_train_from_cmd = False
    config.auto_file_name = True
    config.stage_to_device_map = [0, 0, 0, 0]
    config.step_every = 1
    config.log_frequency = 100
    config.ddp_sim_num_gpus = 4
    config.ddp = True
    config.cudnn_benchmark = True
    config.keep_buffers_alive = True
    config.no_recomputation = True
    config.nesterov_set_for_last_partition = True
    config.max_buffers = 1
    config.gap_aware_just_loss = True
    config.base_config_path = 'configs/dummy_base.json'
    config.base_config_path_is_relative = True
    config.model_name_or_path = 'gpt2'
    config.overwrite_cache = False
    config.train_seq_len = 1024
    config.valid_seq_len = 1024
    config.test_seq_len = 1024
    config.dont_drop_last = True
    config.checkpoints_save_dir = 'results/saved_checkpoints'
    config.checkpoints_save_name_prefix = 'my_checkpint_prefix'
    return config


def get_t5_t5_3b_p8_virtual_stages_boolq_common_config():
    """
    Simplest config, Autogenerated from json file"""
    config = ml_collections.ConfigDict()
    config.logdir = 'logs/t5/virtual_stages/'
    config.data_dir = '/home_local/saareliad/data'
    config.out_dir = 'results/t5/super_glue/boolq'
    config.auto_file_name = True
    config.out_filename = 'test_vs'
    config.distributed_backend = 'mpi'
    config.model = 't5_3b_tied_lmheads_512_4_8p_bw12_squad1_virtual_stages'
    config.stage_to_device_map = [0, 1, 2, 3, 4, 5, 6, 7, 6, 0, 5, 3, 2, 4, 1, 7]
    config.nprocs = 16
    config.dataset = 't5_tfds'
    config.mixture_or_task_name = 'super_glue_boolq_v102'
    config.preproc_batch_size = 128
    config.trainer = ml_collections.ConfigDict()
    config.trainer.type = 't5'
    config.trainer.args = ml_collections.ConfigDict()
    config.trainer.args.always_calc_grad_norm = False
    config.trainer.args.loss_multiplier = 2.59
    config.statistics = 'squad_loss_per_batch'
    config.step_every = 5
    config.bs_train = 4
    config.bs_test = 4
    config.max_seq_length = 512
    config.answer_max_seq_length = 4
    config.num_data_workers = 5
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.type = 'adafactor'
    config.optimizer.args = ml_collections.ConfigDict()
    config.optimizer.args.lr = 0.001
    config.optimizer.args.weight_decay = 0
    config.optimizer.args.scale_parameter = True
    config.optimizer.args.relative_step = False
    config.lr_scheduler = ml_collections.ConfigDict()
    config.lr_scheduler.type = 'get_constant_schedule_with_warmup'
    config.lr_scheduler.preproc_args = ml_collections.ConfigDict()
    config.lr_scheduler.args = ml_collections.ConfigDict()
    config.lr_scheduler.args.num_warmup_steps = 200
    config.lr_scheduler.args.last_epoch = -1
    config.epochs = -1
    config.steps = 3200
    config.seed_from_cmd = False
    config.seed = 42
    config.bs_train_from_cmd = False
    config.bs_test_from_cmd = False
    config.num_chunks = 1
    config.verbose_comm = False
    config.flush_rate = -1
    config.work_scheduler = 'virtual_stages_1f1b'
    config.supremum_staleness = 100
    config.cudnn_benchmark = True
    config.max_buffers = 1
    config.keep_buffers_alive = False
    config.train_batches_limit = -1
    config.log_frequency = 200
    config.model_name_or_path = 't5-3b'
    config.do_lower_case = True
    config.overwrite_cache = False
    config.dont_drop_last = True
    config.model_type = 't5'
    config.precomputed_masks = True
    config.save_checkpoints = True
    config.checkpoints_save_name_prefix = 'stale_adafactor'
    config.checkpoints_save_dir = '/nfs_Disk2/virtual_stages/checkpoints/t5/3b/boolq/stale/'
    config.load_model_one_by_one = False
    return config

if __name__ == '__main__':
    # _convert_json_to_to_python_config(path="pipe/configs/t5/t5_mpipe/boolq/stale.json")
    _convert_json_to_to_python_config(path="pipe/configs/t5/t5_mpipe/boolq/common.json")


# from absl import app
# def main(_):
#     pass
# if __name__ == '__main__':
#     app.run(main)
