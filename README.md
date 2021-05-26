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

The parameters of the best Neural Network model will be stored in output.txt

Watch the trained model in action:
```
g++ -o snake run_snake.cpp
./snake output.txt
```
![video](https://github.com/NipamB/Snake-game-with-Genetic-Algorithm-on-GPU/edit/master/snake_video.mp4)
