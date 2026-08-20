"""Training sweep for 16K-latent SAEs on Cambridge butterflies (384p, v1.6)."""


def make_cfgs() -> list[dict]:
    n_train = 100_000_000
    batch_size = 1024 * 8


    # NEON HARV 256 x 256 image shards 
    shards_256 = "/fs/ess/PAS2136/SAtrEes/saev/shards/b0d74ed4"
    base_run_dir = "/fs/ess/PAS2136/SAtrEes/saev/runs/"

    runs = [
        "agw9clvn", 
        "mu3c9fsx", 
        "p0f7lglc", 
        "s74a1cbp"
    ]
    
    cfgs = []
    for run in runs:
        cfgs.append({
            "run": base_run_dir + run,
            "data": {
                "shards": shards_256,
                "batch_size": batch_size,
            }
        })
    return cfgs
