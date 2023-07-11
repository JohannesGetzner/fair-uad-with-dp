import wandb
api = wandb.Api()
for r in api.runs("thesis-getzner/unsupervised-fairness", filters={"group": "FAE-rsna-age-2023.07.05-12:00:53-bs1024_mgn001"}):
    r.group = "PARTIAL-FAE-rsna-age-2023.07.05-12:00:53-bs1024_mgn001"
    r.update()