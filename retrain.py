import subprocess
from concurrent.futures import ThreadPoolExecutor


def run_training(adapter_config, args):
    # Convert the adapter config list to a space-separated string
    adapter_config_str = "_".join(map(str, adapter_config))

    if args.dataset == 'camvid':
        dataset_args = [
            "--dataset=camvid",
            "--data-path=./data/CamVid",
        ]
    else:
        raise NotImplementedError(f'Only camvid is supported in this pre-release version (got {args.dataset}).')

    if len(adapter_config) == 10:
        # Construct the command
        command = [
            "python", "-u", "train.py",
            f"--models-path={args.retrain_root}/{args.dataset_save_name}/retrain/{args.model_name}_{args.dataset_save_name}_{adapter_config_str}_eps{args.retrain_epochs}",
            "--batch-size=4",
            f"--epochs={args.retrain_epochs}",
            f"--backend=mutable_{args.backend}",
            "--no_multimask",
            f"--snapshot={args.snapshot}",
            "--adapter_config"
        ]
        if args.backend == 'efficient_vit_t' or args.backend == 'efficient_vit_t_hq':
            command = command + [str(i) for i in adapter_config]
        elif args.backend == 'efficient_vit_b':
            command = command + ['-1', '-1'] + [str(i) for i in adapter_config]
        else:
            raise NotImplementedError(f'No {args.backend} model name')

        command = command + dataset_args

    elif len(adapter_config) == 30:
        command = [
            "python", "-u", "train.py",
            f"--models-path={args.retrain_root}/{args.dataset_save_name}/retrain/{args.model_name}_{args.dataset_save_name}_{adapter_config_str}_eps{args.retrain_epochs}",
            "--batch-size=4",
            f"--epochs={args.retrain_epochs}",
            f"--backend=mutable_{args.backend}",
            "--no_multimask",
            f"--snapshot={args.snapshot}",
            "--adapter_config"
        ]

        if args.backend == 'efficient_vit_t' or args.backend == 'efficient_vit_t_hq':
            command = command + [str(i) for i in adapter_config[:10]]
            command = command + ['--mlp_config']
            command = command + [str(i) for i in adapter_config[10:]]
        elif args.backend == 'efficient_vit_b':
            command = command + ['-1', '-1'] + [str(i) for i in adapter_config[:10]]
            command = command + ['--mlp_config']
            command = command + ['0.25', '0.25', '0.25', '0.25'] + [str(i) for i in adapter_config[10:]]
        else:
            raise NotImplementedError(f'No {args.backend} model name')
        command = command + dataset_args

    print(command)
    subprocess.run(command)


if __name__ == '__main__':
    gpu_ids = [1, 2]

    adapter_configs = [
        [0, 1, 2, 1, 1, 1, 0, 2, 2, 0],
        [0, 1, 2, 1, 1, 1, 0, 2, 1, 0],
        [0, 1, 2, 1, 1, 0, 0, 2, 2, 0],
        [0, 1, 2, 1, 1, 1, 1, 2, 2, 0],
        [1, 1, 2, 1, 1, 1, 1, 2, 2, 0]
    ]
    num_configs = len(adapter_configs)
    max_workers = len(gpu_ids)

    import math

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(math.ceil(num_configs / max_workers)):
            start_idx = i * max_workers
            end_idx = min((i + 1) * max_workers, len(adapter_configs))
            config_batch = adapter_configs[start_idx:end_idx]
            retrain_models = [executor.submit(run_training, adapter_config, gpu_id) for adapter_config, gpu_id in zip(config_batch, gpu_ids)]
            for retrain_model in retrain_models:
                retrain_model.result()
