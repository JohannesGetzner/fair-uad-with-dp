import wandb
api = wandb.Api()
current = "2023-10-26 20:24:31-FAE-rsna-age-bs32-ms-noDP"
target =  "2023-10-25 21:54:19-FAE-rsna-age-bs32-ms-noDP"
for r in api.runs(
        "thesis-getzner/unsupervised-fairness",
        filters={
            "group": current
        }):
    # protected_attr = r._attrs["config"]["protected_attr"]
    r.group = target
    # r.tags = r.tags + ["FINAL_RUN"]
    r.update()