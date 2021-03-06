#include <bits/stdc++.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include<time.h>

using namespace std;

#define ni 24                   // number of neurons in input layer
#define nh 20                   // number of neurons in hidden layer
#define no 4                    // number of neurons in output layer
#define width 30                // width of the game boundary
#define height 20               // height og the game boundary
#define max_snake_length 100    // maximum length of the snake 

#define population_size 4096
#define natural_selection_rate 0.4
#define mutation_rate 0.01
#define generations 10000
#define negative_reward -150
#define positive_reward 500
#define max_total_steps 1000

// randomly initialise neural network parameters to negative values
__global__ void initialise_nn(float *nns, unsigned int *random_int){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    nns[id] = (random_int[id] % 2) ? nns[id] : -nns[id];
}

// set the input for neural network, the input size is 24 i.e it looks for wall,
// it's body and fruit in all the 8 directions
__device__ void set_input(float input[], int x, int y, int fruitx, int fruity,
                            int tailx[], int taily[], int ntail){
    for(int i=0;i<ni;i++)
        input[i] = 0;

    // check up direction
    // check food
    if(fruitx == x && fruity < y)
        input[0] = 1;

    // check body
    for(int i=0;i<ntail;i++){
        if(tailx[i] == x && taily[i] < y){
            input[1] = 1;
            break;
        }
    }

    // check wall distance
    if(y != 0)
        input[2] = 1 / (float)y;

    // check down direction
    // check food
    if(fruitx == x && fruity > y)
        input[3] = 1;

    // check body
    for(int i=0;i<ntail;i++){
        if(tailx[i] == x && taily[i] > y){
            input[4] = 1;
            break;
        }
    }

    // check wall distance
    if(height-y != 0)
        input[5] = 1 / (float)(height-y);

    // check right direction
    // check food
    if(fruity == y && fruitx > x)
        input[6] = 1;

    // check body
    for(int i=0;i<ntail;i++){
        if(taily[i] == y && tailx[i] > x){
            input[7] = 1;
            break;
        }
    }

    // check wall distance
    if(width-x != 0)
        input[8] = 1 / (width-x);

    // check left direction
    // check food
    if(fruity == y && fruitx < x)
        input[9] = 1;

    // check body
    for(int i=0;i<ntail;i++){
        if(taily[i] == y && tailx[i] < x){
            input[10] = 1;
            break;
        }
    }

    // check wall distance
    if(x != 0)
        input[11] = 1 / (float)x;

    //check north-east direction
    int tempx = x, tempy = y;
    bool found_food = false, found_body = false;

    // check food and body
    while(tempx < width && tempy > 0){
        tempx++;
        tempy--;
        if(!found_food && tempx == fruitx && tempy == fruity){
            input[12] = 1;
            found_food = true;
        }
        if(!found_body){
            for(int i=0;i<ntail;i++){
                if(tempx == tailx[i] && tempy == taily[i]){
                    input[13] = 1;
                    found_body = true;
                    break;
                }
            }
        }
        if(found_body && found_food)
            break;
    }

    // check wall distance
    int min_value = min(width-x,y);
    float distance = sqrt(pow(min_value,2)*2);
    if(distance != 0)
        input[14] = 1 / distance; 

    //check north-west direction
    tempx = x, tempy = y;
    found_food = false, found_body = false;

    // check food and body
    while(tempx > 0 && tempy > 0){
        tempx--;
        tempy--;
        if(!found_food && tempx == fruitx && tempy == fruity){
            input[15] = 1;
            found_food = true;
        }
        if(!found_body){
            for(int i=0;i<ntail;i++){
                if(tempx == tailx[i] && tempy == taily[i]){
                    input[16] = 1;
                    found_body = true;
                    break;
                }
            }
        }
        if(found_body && found_food)
            break;
    }

    // check wall distance
    min_value = min(x,y);
    distance = sqrt(pow((min_value),2)*2);
    if(distance != 0)
        input[17] = 1 / distance; 

    //check south-west direction
    tempx = x, tempy = y;
    found_food = false, found_body = false;

    // check food and body
    while(tempx > 0 && tempy < height){
        tempx--;
        tempy++;
        if(!found_food && tempx == fruitx && tempy == fruity){
            input[18] = 1;
            found_food = true;
        }
        if(!found_body){
            for(int i=0;i<ntail;i++){
                if(tempx == tailx[i] && tempy == taily[i]){
                    input[19] = 1;
                    found_body = true;
                    break;
                }
            }
        }
        if(found_body && found_food)
            break;
    }

    // check wall distance
    min_value = min(x,height-y);
    distance = sqrt(pow((min_value),2)*2);
    if(distance != 0)
        input[20] = 1 / distance;

    //check south-east direction
    tempx = x, tempy = y;
    found_food = false, found_body = false;

    // check food and body
    while(tempx < width && tempy < height){
        tempx++;
        tempy++;
        if(!found_food && tempx == fruitx && tempy == fruity){
            input[21] = 1;
            found_food = true;
        }
        if(!found_body){
            for(int i=0;i<ntail;i++){
                if(tempx == tailx[i] && tempy == taily[i]){
                    input[22] = 1;
                    found_body = true;
                    break;
                }
            }
        }
        if(found_body && found_food)
            break;
    }

    // check wall distance
    min_value = min(width-x,height-y);
    distance = sqrt(pow((min_value),2)*2);
    if(distance != 0)
        input[23] = 1 / distance;
}

// function to calculate value of neuron in a layer during forward function
__device__ float forward(float input[], float weight[], float bias[], int len_i, int len_o, int index){
    float output = 0;
    for(int i=0;i<len_i;i++){
        output += weight[i*len_o+index] * input[i];
    }
    output += bias[index];

    // sigmoid function
    output = 1.0 / (1.0 + exp(-output));
    return output;
}

// play the game with each block corresponding to one neural network and each thread corresponding to a parameter of neural network
__global__ void play_game(float *nns, float *fitness, unsigned int *random_int_fruitx, unsigned int *random_int_fruity,
                        int parameter_size){

    int snake_id = blockIdx.x;
    int parameter_id = threadIdx.x;

    // neural network of a particular id
    extern __shared__ float nn[];
    nn[parameter_id] = nns[snake_id*parameter_size+parameter_id];

    __syncthreads();

    // weights and biases of the neural network
    float *w1 = &nn[0];
    float *b1 = &nn[ni*nh];
    float *w2 = &nn[ni*nh+nh];
    float *b2 = &nn[ni*nh+nh+nh*no];

    /* setup the game */
    // STOP: 0, LEFT: 1, RIGHT: 2, UP: 3, DOWN: 4

    // next direction to take
    __shared__ int dir;
    dir = 0;
    // next direction to take if the first value is not possible
    __shared__ int dir_next;
    dir_next = 0;
    // last direction taken
    __shared__ int last_dir;
    last_dir = 0;

    // position of head
    __shared__ int x;
    x = width/2;
    __shared__ int y;
    y = height/2;

    // position of fruit
    __shared__ int fruitx; 
    __shared__ int fruity;
    __shared__ int fruit_index;
    fruit_index = snake_id * max_snake_length; 

    fruitx = random_int_fruitx[fruit_index] % width;
    fruity = random_int_fruity[fruit_index] % height;

    fruit_index++;

    //snake length
    __shared__ int ntail;
    ntail = 3;

    // array to store snake body
    __shared__ int tailx[max_snake_length];
    __shared__ int taily[max_snake_length];

    // local variables
    int total_steps = 200;
    float total_reward = 0;
    float reward = 0;
    int steps = 0;

    // array to store input, hidden and output layer
    __shared__ float input[ni];
    __shared__ float hidden_output[nh];
    __shared__ float output[no];

    // flag used to exit all the threads in a block
    __shared__ int break_flag;
    break_flag = 0;
    
    // play until the snake dies
    while(true){
        // set the input for the game
        set_input(input,x,y,fruitx,fruity,tailx,taily,ntail);
        
        // forward function for the first layer
        if(parameter_id < nh){
            hidden_output[parameter_id] = forward(input,w1,b1,ni,nh,parameter_id);
        }

        __syncthreads();

        // forward function for the second layer and thus get the output layer
        if(parameter_id < no){
            output[parameter_id] = forward(hidden_output,w2,b2,nh,no,parameter_id);
        }

        __syncthreads();

        // thread id = 0 executes the logic of the game
        if(parameter_id == 0){
            // find the two best directions to be taken
            float max_value = output[0];
            float max_index = 0;
            for(int i=1;i<no;i++){
                if(output[i] > max_value){
                    max_value = output[i];
                    max_index = i;
                }
            }
            dir = max_index + 1;

            float max_value1 = INT16_MIN;
            float max_index1 = -1;
            for(int i=0;i<no;i++){
                if(i != max_index && output[i] > max_value1){
                    max_value1 = output[i];
                    max_index1 = i;
                }
            }
            dir_next = max_index1 + 1;

            // update the snake body
            int prevx = tailx[0];
            int prevy = taily[0];
            int prev2x, prev2y;
            tailx[0] = x;
            taily[0] = y;

            for(int i=1;i<ntail;i++)
            {
                prev2x = tailx[i];
                prev2y = taily[i];
                tailx[i] = prevx;
                taily[i] = prevy;
                prevx = prev2x;
                prevy = prev2y;
            }

            // move snake in the next direction 
            switch(dir)
            {
                case 1:
                    if(last_dir != 2)
                        x--;
                    else{
                        if(dir_next == 2)
                            x++;
                        else if(dir_next == 3)
                            y--;
                        else if(dir_next == 4)
                            y++;
                    }
                    break;
                case 2:
                    if(last_dir != 1)
                        x++;
                    else{
                        if(dir_next == 1)
                            x--;
                        else if(dir_next == 3)
                            y--;
                        else if(dir_next == 4)
                            y++;
                    }
                    break;
                case 3:
                    if(last_dir != 4)
                        y--;
                    else{
                        if(dir_next == 1)
                            x--;
                        else if(dir_next == 2)
                            x++;
                        else if(dir_next == 4)
                            y++;
                    }
                    break;
                case 4:
                    if(last_dir != 3)
                        y++;
                    else{
                        if(dir_next == 1)
                            x--;
                        else if(dir_next == 2)
                            x++;
                        else if(dir_next == 3)
                            y--;
                    }
                    break;
            }

            last_dir = dir;

            // snake hits the wall
            if(x >= width || x < 0 || y >= height || y < 0)
            {
                reward = negative_reward;
                break_flag = 1;
            }

            // snake hits its body
            for(int i =0; i<ntail;i++)
            {
                if(tailx[i]==x && taily[i]==y)
                {
                    reward = negative_reward;
                    break_flag = 1;
                }
            }

            // snake eats the fruit
            if(x==fruitx && y==fruity)
            {
                fruitx = random_int_fruitx[fruit_index] % width;
                fruity = random_int_fruity[fruit_index] % height;
                fruit_index++;
                ntail++;
                reward = positive_reward;
            }
            
            total_reward += reward;
            
            steps += 1;

            if(reward == -1){
                break_flag = 1;
            }

            reward = 0;

            // update total steps the snake can take
            if(reward > 0)
                total_steps = (total_steps+100 > max_total_steps) ? max_total_steps : total_steps + 100;

            if(steps > total_steps){
                break_flag = 1;
            }
        }

        __syncthreads();

        // exit while loop for all the threads in the block if the snake dies
        if(break_flag)
            break;
    }
    
    __syncthreads();

    // update the fitness score for the game
    if(parameter_id == 0){
        fitness[snake_id] = total_reward + steps;
    }
}

// update the device array to store top neural networks which will be used for crossover
__global__ void select_top(float *nns, float *nns_new, int *indices){
    int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    int id2 = indices[blockIdx.x] * blockDim.x + threadIdx.x;

    nns_new[id1] = nns[id2];
}

// intialise the device array for indices
__global__ void intialise_indices(int *indices){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    indices[id] = id;
}

// crossover the top neural networks to generate new neural networks for the generation population
__global__ void crossover(float *nns, float *fitness, unsigned int *random_int1, unsigned int *random_int2, int top){
    int snake_id = blockIdx.x;
    int parameter_id = threadIdx.x;

    // select parents using Roulette Wheel Selection method
    int fitness_sum = 0;
    for(int i=0;i<population_size;i++)
        fitness_sum += fitness[i];

    // select parent 1
    int parent1 = 0;
    if(fitness_sum != 0){
        int rand_num = random_int1[snake_id] % fitness_sum;
        int sum = 0;
        for(int i=0;i<population_size;i++){
            sum += fitness[i];
            if(sum > rand_num){
                parent1 = i;
                break;
            }
        }
    }

    // select parent 2
    int parent2 = 0;
    if(fitness_sum != 0){
        int rand_num = random_int2[snake_id + blockDim.x] % fitness_sum;
        int sum = 0;
        for(int i=0;i<population_size;i++){
            sum += fitness[i];
            if(sum > rand_num){
                parent2 = i;
                break;
            }
        }
    }

    // child index
    int child = top + snake_id;

    // choose index of the parameter randomly
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int rand_num = random_int2[id];

    // perform crossover to generate new neural network 
    if(rand_num%2 == 0){
        nns[child * blockDim.x + parameter_id] = nns[parent1 * blockDim.x + parameter_id];
    }
    else{
        nns[child * blockDim.x + parameter_id] = nns[parent2 * blockDim.x + parameter_id];
    }
}

// mutate the neural network parameters based on mutation rate
__global__ void mutate(float *nns, float *random_float1, float *random_float2){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // mutate only if random value is less than mutation rate
    if(random_float1[id] < mutation_rate){
        nns[id] += random_float2[id] / 5;
        if(nns[id] > 1)
            nns[id] = 1;
        if(nns[id] < -1)
            nns[id] = -1;
    }
}

int main(){
    srand(time(NULL));

    ofstream fout;
    // file to store best neural network parameters
    fout.open("output.txt");
    
    ofstream ftime;
    // file to store every generation time
    ftime.open("generation_time.txt");

    // write model parameters into the file
    fout<<"n_input\t\t"<<ni<<endl;
    fout<<"n_hidden\t"<<nh<<endl;
    fout<<"n_output\t"<<no<<endl;
    fout<<"height\t\t"<<height<<endl;
    fout<<"width\t\t"<<width<<endl;

    // number of parameters of neural network
    int parameter_size = ni*nh + nh + nh*no + no;
    cout<<"Parameter size: "<<parameter_size<<endl;

    // neural networks for device
    float *dnns, *dnns_new;

    // allocate memory for neural networks in device
    cudaMalloc((void **)&dnns,population_size*parameter_size*sizeof(float));
    cudaMalloc((void **)&dnns_new,population_size*parameter_size*sizeof(float));

    curandGenerator_t prng;
	
	// create pseudo random number generator
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_MT19937);
	curandSetPseudoRandomGeneratorSeed(prng, 41ULL);
	
	// initialise neural networks with uniform distribution
    curandGenerateUniform(prng, dnns, population_size*parameter_size);

    // create random number generator for integer values
    unsigned int *random_int;
    cudaMalloc((void**) &random_int,population_size*parameter_size*sizeof(int));

	curandGenerate(prng,random_int,population_size*parameter_size); 
    
    // initialse the neural networks to have negative values also
    initialise_nn<<<population_size,parameter_size>>>(dnns,random_int);

    // device variable to store fitness score and their indices
    float *dfitness;
    int *dindices;

	// fitness score on host
	float *fitness = (float *) malloc(population_size*sizeof(float));

	// fitness score and indices on device
	cudaMalloc((void**) &dfitness,population_size*sizeof(float));
	cudaMalloc((void**) &dindices,population_size*sizeof(int));

    // thrust device pointer to fitness score and indices array
    thrust::device_ptr<float> fitness_ptr(dfitness);
    thrust::device_ptr<int> indices_ptr(dindices);

    // random number generator used for generating indices of fruit
    unsigned int *random_int_fruitx;
    cudaMalloc((void**) &random_int_fruitx,population_size*max_snake_length*sizeof(int));
    unsigned int *random_int_fruity;
    cudaMalloc((void**) &random_int_fruity,population_size*max_snake_length*sizeof(int));

    // random number generator used during crossover
    unsigned int *random_int_crossover1;
    cudaMalloc((void**) &random_int_crossover1,2*population_size*sizeof(int));
    unsigned int *random_int_crossover2;
    cudaMalloc((void**) &random_int_crossover2,population_size*parameter_size*sizeof(int));

    // random number generator used during mutation
    float *random_float_mutate1;
    cudaMalloc((void**) &random_float_mutate1,population_size*parameter_size*sizeof(float));
    float *random_float_mutate2;
    cudaMalloc((void**) &random_float_mutate2,population_size*parameter_size*sizeof(float));
    
    // local variables
    float max_reward = 0;
    float avg_reward = 0;
    int max_index = 0;
    float global_max_reward = 0;
    int global_max_generation = 0;
    float max_avg_reward = 0;

    // array to store parameters of the best neural network
    float *best_snake = (float *)malloc(parameter_size*sizeof(float));

    // loop for number of generations
    for(int k=0;k<generations;k++){
    	clock_t tStart = clock();
    
        // intialise indices array corresponding to fitness array
        int num_threads = (population_size > 1024) ? 1024 : population_size;
        int num_blocks = population_size/1024 + 1;
        intialise_indices<<<num_blocks,num_threads>>>(dindices);

        // create random number generator for integer values of fruit
        curandGenerate(prng,random_int_fruitx,population_size*max_snake_length);
        curandGenerate(prng,random_int_fruity,population_size*max_snake_length);
        
        // play the games on GPU
        play_game<<<population_size,parameter_size,parameter_size*sizeof(float)>>>(dnns,dfitness,random_int_fruitx,random_int_fruity,parameter_size);
        
        // copy device fitness score to host
        cudaMemcpy(fitness,dfitness,population_size*sizeof(float),cudaMemcpyDeviceToHost);

        // find the index with maximum fitness score and also calculate average fitness score 
        avg_reward = 0;
        max_reward = fitness[0];
        max_index = 0;
        for(int i=1;i<population_size;i++){
            if(fitness[i] > max_reward){
                max_reward = fitness[i];
                max_index = i;
            }
            avg_reward += fitness[i];
        }
        avg_reward /= population_size;
        
        double generation_time = (double)(clock() - tStart)/CLOCKS_PER_SEC;
        ftime<<generation_time<<endl;
        
        printf("generation: %d\tAverage fitness: %f\tMax reward: %f\tTime: %f\n",k+1,avg_reward,max_reward,generation_time);

        // find the maximum fitness score among all the generations
        if(max_reward >= global_max_reward){
            global_max_reward = max_reward;
            global_max_generation = k+1;
        }

        // copy parameters of neural network with maximum average fitness score among all the generations
        if(avg_reward >= max_avg_reward){
            max_avg_reward = avg_reward;
            cudaMemcpy(best_snake,dnns+max_index*parameter_size,parameter_size*sizeof(float),cudaMemcpyDeviceToHost);
        }

        // number of neural networks passed on to next generation from current generation
        int top = population_size * natural_selection_rate;

        // sort the device fitness score array in descennding order along with the indices array
        thrust::sort_by_key(fitness_ptr,fitness_ptr+population_size,indices_ptr,thrust::greater<float>());

        // update device neural network array with top neural network parameters
        select_top<<<top,parameter_size>>>(dnns,dnns_new,dindices);

        float *temp = dnns_new;
        dnns_new = dnns;
        dnns = temp;

        // create random number generator for integer values used during crossover
        curandGenerate(prng,random_int_crossover1,2*population_size);
        curandGenerate(prng,random_int_crossover2,population_size*parameter_size);
        
        // crossover the top neural networks to generate the remaining neural networks in the population
        crossover<<<population_size-top,parameter_size>>>(dnns,dfitness,random_int_crossover1,random_int_crossover2,top);

        // create random number generator for float values used during mutation
        curandGenerateUniform(prng,random_float_mutate1,population_size*parameter_size);
        curandGenerateNormal(prng,random_float_mutate2,population_size*parameter_size,0.0,1.0);

        // mutate all neural network parameters in accordance to mutation rate
        mutate<<<population_size,parameter_size>>>(dnns,random_float_mutate1,random_float_mutate2);

    }

    // write parameters of the best neural network into file
    fout<<"Best neural network parameters: \n";
    for(int i=0;i<parameter_size;i++)
        fout<<best_snake[i]<<" ";
    fout<<endl;

    printf("Generation: %d\tGlobal max reward: %f\n",global_max_generation,global_max_reward);

    fout.close();
    ftime.close();
    
    cudaFree(dnns);
    cudaFree(dnns_new);
    cudaFree(random_int);
    cudaFree(dfitness);
    cudaFree(dindices);
    cudaFree(random_int_fruitx);
    cudaFree(random_int_fruity);
    cudaFree(random_int_crossover1);
    cudaFree(random_int_crossover2);
    cudaFree(random_float_mutate1);
    cudaFree(random_float_mutate2);
    free(fitness);
    free(best_snake);

    return 0;
}
