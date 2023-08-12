import wandb
api = wandb.Api()
current = "2023-08-11 13:24:51-FAE-rsna-age-bs1024_mgn001_old_down_weighted-DP"
target = "2023-08-07 16:45:30-FAE-rsna-age-bs1024_mgn001_old_down_weighted-DP"
for r in api.runs(
        "thesis-getzner/unsupervised-fairness",
        filters={
            "group": current
        }):
    r.group = target
    r.update()