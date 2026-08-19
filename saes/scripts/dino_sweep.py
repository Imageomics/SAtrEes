"""Training sweep for 16K-latent SAEs on Cambridge butterflies (384p, v1.6)."""


def make_cfgs() -> list[dict]:
    n_train = 100_000_000
    batch_size = 1024 * 8


    # NEON HARV 256 x 256 image shards 
    shards_256 = "/fs/ess/PAS2136/SAtrEes/saev/shards/b0d74ed4"

    cfgs = []
    for layer in [-2]:
        for k in [16, 32, 64, 128]:
            for lr in [1e-3]:
                cfgs.append({
                    "tags": ["satrees_dinov3"],
                    "slurm_acct": "PAS2136",
                    "slurm_partition": "nextgen",
                    "n_hours": 8.0,
                    "lr": lr,
                    "n_lr_warmup": 500,
                    "n_sparsity_warmup": n_train // batch_size,
                    "runs_root": "/fs/ess/PAS2136/SAtrEes/saev/runs",
                    "n_train": n_train,
                    "sae": {
                        "d_model": 1024,
                        "d_sae": 1024 * 16,
                        "normalize_w_dec": True,
                        "remove_parallel_grads": True,
                        "activation": {"top_k": k},
                        "reinit_blend": 0.8,
                    },
                    "train_data": {
                        "layer": layer,
                        "shards": shards_256,
                        "batch_size": batch_size,
                        "min_buffer_fill": 0.2,
                        "use_tmpdir": True,
                    },
                    "val_data": {
                        "layer": layer,
                        "shards": shards_256,
                        "batch_size": batch_size,
                        "use_tmpdir": True,
                    },
                })
    return cfgs
