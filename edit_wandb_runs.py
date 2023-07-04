import wandb
api = wandb.Api()
for r in api.runs("thesis-getzner/unsupervised-fairness", filters={"group": "FAE-rsna-sex-2023.07.03-10:00:48"}):
    r.group = "FAE-rsna-sex-2023.06.29-18:27:55"
    r.update()