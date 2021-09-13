from ray import tune

tune.run("DQN", config={
    "framework": "torch",
    "env": "PongNoFrameskip-v4"
})