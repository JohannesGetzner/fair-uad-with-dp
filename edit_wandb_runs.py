import wandb
api = wandb.Api()
current = "2023.08.19-15:49:47-FAE-rsna-age-SWEEP-lr-bs1024-mgn001-old-down-weighted-DP"
target = "2023-08-19 15:49:47-FAE-rsna-age-SWEEP-lr-bs1024-mgn001-old-down-weighted-DP"
for r in api.runs(
        "thesis-getzner/unsupervised-fairness",
        filters={
            "group": current
        }):
    r.group = target
    r.update()