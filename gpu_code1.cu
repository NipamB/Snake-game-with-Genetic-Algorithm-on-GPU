#include <bits/stdc++.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

using namespace std;

#define ni 24
#define nh 30
#define no 4
#define width 30
#define height 40

#define population_size 4096
#define natural_selection_rate 0.2
#define mutation_rate 0.01
#define generations 500

__global__ void myprint(float *nns, int size){
    int x = 5*(ni*nh+nh+nh*no+no);
    for(int i=x;i<x+24;i++)
        printf("%f ",nns[i]);
    printf("\n");
}

__global__ void initialise_nn(float *nns, unsigned int *random_int){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    nns[id] = (random_int[id] % 2) ? nns[id] : -nns[id];
}

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

__device__ float forward(float input[], float weight[], float bias[], int len_i, int len_o, int index){
    float output = 0;
    for(int i=0;i<len_i;i++){
        output += weight[i*len_o+index] * input[i];
    }
    output += bias[index];

    // sigmoid function
    output = 1.0 / (1.0 + expf(-output));
    return output;
}

__global__ void play_game(float *nns, int *fitness, unsigned int *random_int_fruitx, unsigned int *random_int_fruity,
                        int parameter_size, int gen){

    // int id = blockIdx.x * blockDim.x + threadIdx.x;
    int snake_id = blockIdx.x;
    int parameter_id = threadIdx.x;

    extern __shared__ float nn[];
    // __shared__ float *nn;
    // nn = (float *)malloc(parameter_size*sizeof(float));    
    nn[parameter_id] = nns[snake_id*parameter_size+parameter_id];

    __syncthreads();

    float *w1 = &nn[0];
    float *b1 = &nn[ni*nh];
    float *w2 = &nn[ni*nh+nh];
    float *b2 = &nn[ni*nh+nh+nh*no];

    /* setup teh game */
    // STOP: 0, LEFT: 1, RIGHT: 2, UP: 3, DOWN: 4
    int dir = 0;

    // position of head
    int x = width/2;
    int y = height/2;

    // position of fruit
    int fruitx; 
    int fruity;
    int fruit_index = 0; 

    fruitx = random_int_fruitx[fruit_index] % width;
    fruity = random_int_fruity[fruit_index] % height;

    fruit_index++;

    //snake length
    int ntail = 2;

    int tailx[100], taily[100];

    int total_steps = 200;
	double total_reward = 0;
	double reward = 0;
	int steps = 0;
    __shared__ float input[ni];
    __shared__ float hidden_output[nh];
    __shared__ float output[no];
    
    while(true){
        set_input(input,x,y,fruitx,fruity,tailx,taily,ntail);
        
        if(parameter_id < nh){
            hidden_output[parameter_id] = forward(input,w1,b1,ni,nh,parameter_id);
        }

        __syncthreads();

        if(parameter_id < no){
            output[parameter_id] = forward(hidden_output,w2,b2,nh,no,parameter_id);
        }

        __syncthreads();

        if(parameter_id == 0){
            float max_value = output[0];
            float max_index = 0;
            for(int i=1;i<no;i++){
                if(output[i] > max_value){
                    max_value = output[i];
                    max_index = i;
                }
            }
            dir = max_index + 1;
        }

        __syncthreads();

        // if(snake_id == 0 && parameter_id == 0){
        //     for(int i=0;i<no;i++)
        //         printf("%f ",output[i]);
        //     printf("\n");
        //     // printf("%d\n",dir);
        // }

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

        switch(dir)
        {
            case 1:
                x--;
                break;
            case 2:
                x++;
                break;
            case 3:
                y--;
                break;
            case 4:
                y++;
                break;
        }

        if(x >= width || x < 0 || y >= height || y < 0)
        {
            reward = -1;
        }

        for(int i =0; i<ntail;i++)
        {
            if(tailx[i]==x && taily[i]==y)
            {
                reward = -1;
            }
        }

        if(x==fruitx && y==fruity)
        {
            fruitx = random_int_fruitx[fruit_index] % width;
            fruity = random_int_fruity[fruit_index] % height;
            fruit_index++;
            ntail++;
            reward = 1;
        }

        if(reward == -1)
            break;

        total_reward += reward;
        reward = 0;

        steps += 1;

        if(reward > 0)
            total_steps = (total_steps+100 > 500) ? 500 : total_steps + 100;

        if(steps > total_steps)
            break;
    }
    
    __syncthreads();

    if(parameter_id == 0){
        fitness[snake_id] = total_reward;
        if(gen == 2 && snake_id == 5){
            fitness[snake_id] = 2;
        }
    }

    __syncthreads();

    // if(id == 0){
    //     for(int i=0;i<population_size;i++)
    //         printf("%d ",fitness[i]);
    //     printf("\n");
    // }
}

__global__ void select_top(float *nns, float *nns_new, int *indices){
    int id1 = blockIdx.x * blockDim.x + threadIdx.x;
    int id2 = indices[blockIdx.x] * blockDim.x + threadIdx.x;

    nns_new[id1] = nns[id2];
}

__global__ void myprint1(int *fitness, int *indices){
    for(int i=0;i<population_size;i++)
        printf("%d\t%d\n",fitness[i],indices[i]);
}

__global__ void intialise_indices(int *indices){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    indices[id] = id;
}

__global__ void crossover(float *nns, unsigned int *random_int1, unsigned int *random_int2){
    int snake_id = blockIdx.x;
    int parameter_id = threadIdx.x;

    int top = population_size * natural_selection_rate;

    if(parameter_id <= random_int2[snake_id] % blockDim.x){
        nns[(top + snake_id) * blockDim.x + parameter_id] = nns[(random_int1[snake_id] % top) * blockDim.x + parameter_id];
    }
    else{
        nns[(top + snake_id) * blockDim.x + parameter_id] = nns[(random_int1[snake_id + blockDim.x] % top) * blockDim.x + parameter_id];
    }
}

__global__ void mutate(float *nns, float *random_float1, float *random_float2){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

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
	curandSetPseudoRandomGeneratorSeed(prng, 42ULL);
	
	// initialise neural networks with uniform distribution
    curandGenerateUniform(prng, dnns, population_size*parameter_size);

    // create random number generator for integer values
    unsigned int *random_int;
    cudaMalloc((void**) &random_int,population_size*parameter_size*sizeof(int));

	curandGenerate(prng,random_int,population_size*parameter_size); 
    
    // initialse the neural networks to have negative values also
    initialise_nn<<<population_size,parameter_size>>>(dnns,random_int);

    // myprint<<<1,1>>>(dnns,population_size*parameter_size);

    // cudaDeviceSynchronize();

    int *dfitness, *dindices;

	// fitness score on host
	int *fitness = (int *) malloc(population_size*sizeof(int));

	// fitness score on device
	cudaMalloc((void**) &dfitness,population_size*sizeof(int));
	cudaMalloc((void**) &dindices,population_size*sizeof(int));

    thrust::device_ptr<int> fitness_ptr(dfitness);
    thrust::device_ptr<int> indices_ptr(dindices);

    unsigned int *random_int_fruitx;
    cudaMalloc((void**) &random_int_fruitx,parameter_size*sizeof(int));
    unsigned int *random_int_fruity;
    cudaMalloc((void**) &random_int_fruity,parameter_size*sizeof(int));

    unsigned int *random_int_crossover1;
    cudaMalloc((void**) &random_int_crossover1,2*population_size*sizeof(int));
    unsigned int *random_int_crossover2;
    cudaMalloc((void**) &random_int_crossover2,population_size*sizeof(int));

    float *random_float_mutate1;
    cudaMalloc((void**) &random_float_mutate1,population_size*parameter_size*sizeof(float));
    float *random_float_mutate2;
    cudaMalloc((void**) &random_float_mutate2,population_size*parameter_size*sizeof(float));
    
    int max_reward = 0;
    float avg_reward = 0;
    int global_max_reward = 0;
    int global_max_generation = 0;

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);
    for(int k=0;k<generations;k++){
        // cout<<"Generation: "<<k+1<<endl;

        // intialise indices array corresponding to fitness array
        int num_threads = (population_size > 1024) ? 1024 : population_size;
        int num_blocks = population_size/1024 + 1;
        intialise_indices<<<num_blocks,num_threads>>>(dindices);

        curandSetStream(prng,stream1);

        // create random number generator for integer values of fruit
        curandGenerate(prng,random_int_fruitx,parameter_size);
        curandGenerate(prng,random_int_fruity,parameter_size);
        
        play_game<<<population_size,parameter_size,parameter_size*sizeof(float)>>>(dnns,dfitness,random_int_fruitx,random_int_fruity,parameter_size,k);
        
        cudaMemcpy(fitness,dfitness,population_size*sizeof(int),cudaMemcpyDeviceToHost);
        
        avg_reward = 0;
        max_reward = fitness[0];
        for(int i=1;i<population_size;i++){
            if(fitness[i] > max_reward){
                max_reward = fitness[i];
            }
            avg_reward += fitness[i];
        }
        avg_reward /= population_size;
        
        printf("generation: %d\tAverage fitness: %f\tMax reward: %d\n",k+1,avg_reward,max_reward);

        if(max_reward > global_max_reward){
            global_max_reward = max_reward;
            global_max_generation = k+1;
        }
        
        int top = population_size * natural_selection_rate;

        thrust::sort_by_key(fitness_ptr,fitness_ptr+population_size,indices_ptr,thrust::greater<int>());

        // myprint1<<<1,1>>>(dfitness,dindices);

        select_top<<<top,parameter_size>>>(dnns,dnns_new,dindices);

        float *temp = dnns_new;
        dnns_new = dnns;
        dnns = temp;

        curandGenerate(prng,random_int_crossover1,2*population_size);
        curandGenerate(prng,random_int_crossover2,population_size);

        curandGenerateUniform(prng,random_float_mutate1,population_size*parameter_size);
        curandGenerateNormal(prng,random_float_mutate2,population_size*parameter_size,0.0,1.0);

        cudaStreamSynchronize(stream1);
        
        crossover<<<population_size-top,parameter_size>>>(dnns,random_int_crossover1,random_int_crossover2);

        mutate<<<population_size,parameter_size>>>(dnns,random_float_mutate1,random_float_mutate2);

        // myprint<<<1,1>>>(dnns,parameter_size);

        // cudaDeviceSynchronize();
    }

    printf("Generation: %d\tGlobal max reward: %d\n",global_max_generation,global_max_reward);

    cudaStreamDestroy(stream1);
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

    return 0;
}