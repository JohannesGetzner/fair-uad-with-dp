import wandb
api = wandb.Api()
current = ""
target =  ""
for r in api.runs(
        "thesis-getzner/unsupervised-fairness",
        filters={
            "group": current
        }):
    # protected_attr = r._attrs["config"]["protected_attr"]
    r.group = target
    # r.tags = r.tags + ["FINAL_RUN"]
    r.update()