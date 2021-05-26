# Snake-game-with-Genetic-Algorithm-on-GPU

## Execution
Train on CPU:
```
g++ -o train_cpu cpu_code.cpp
./train_cpu
```

Train on GPU:
```
nvcc -o train_gpu gpu_code.cu -lcurand
./train_gpu
```


