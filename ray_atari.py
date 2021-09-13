from ray import tune

tune.run("DQN", config={
    "env": "BreakoutNoFrameskip-v4",
    "framework": "torch"
})
