import wandb
api = wandb.Api()
current = ""
target = ""
for r in api.runs(
        "thesis-getzner/unsupervised-fairness",
        filters={
            "group": current
        }):
    r.group = target
    # r.tags = r.tags + ["DEPRECATED"]
    r.update()