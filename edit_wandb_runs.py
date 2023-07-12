import wandb
api = wandb.Api()
for r in api.runs("thesis-getzner/unsupervised-fairness", filters={"group": "2023.07.11-20:38:32-FAE-rsna-age-bs32"}):
    r.group = "2023.07.11-20:38:32-FAE-rsna-age-bs32-noDP"
    r.update()