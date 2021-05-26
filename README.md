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

https://user-images.githubusercontent.com/26038610/119701411-ec4d4180-be71-11eb-8811-8c1859c1cdde.mp4


