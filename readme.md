# Reward Shaping

## 训练随机游走

```sh
python train_maze.py
```

## 训练小车

```sh
python train_mountain_car.py
```

## 查看曲线

```sh
python viz.py rewards.***.npy "图例名称1" rewards.***.npy "图例名称2"
```

其中`rewards.***.npy`文件是训练时生成的，记录了每一步的累计奖励