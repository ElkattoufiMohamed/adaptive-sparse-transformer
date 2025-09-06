import wandb

wandb.init(
    project="adaptive-transformer",  # choose any name
    name="connection-test"
)

for step in range(5):
    wandb.log({"loss": 1.0 / (step + 1)})

wandb.finish()
