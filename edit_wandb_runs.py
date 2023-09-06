import wandb
api = wandb.Api()
current = "2023-09-01 14:30:01-FAE-rsna-age-bs512-mgn001-upsamplingeven-DP-DEPRECATED"
target = "2023-09-01 14:30:01-FAE-rsna-age-bs512-mgn001-upsamplingeven-DP-DEPRECATED"
for r in api.runs(
        "thesis-getzner/unsupervised-fairness",
        filters={
            "group": current
        }):
    r.group = target
    # r.tags = r.tags + ["DEPRECATED"]
    r.update()