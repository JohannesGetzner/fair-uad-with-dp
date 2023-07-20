import wandb
api = wandb.Api()
for r in api.runs("thesis-getzner/unsupervised-fairness", filters={"group": "2023.07.19-08:07:48-FAE-rsna-age-bs1024_mgn001-DP"}):
    r.group = "2023.07.13-09:20:21-FAE-rsna-age-bs1024_mgn001"
    r.update()