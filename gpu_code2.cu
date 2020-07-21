#include <bits/stdc++.h>
#include <unistd.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

using namespace std;

#define ni 24
#define nh 20
#define no 4
#define width 30
#define height 20
#define max_snake_length 100

#define population_size 4096
#define natural_selection_rate 0.2
#define mutation_rate 0.01
#define generations 300
#define negative_reward -150
#define positive_reward 500
#define max_total_steps 500
#define threshold_view_game 50

////////////////////////////////////////////////
//for color


#ifndef _COLORS_
#define _COLORS_

/* FOREGROUND */
#define RST  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"

#define FRED(x) KRED x RST
#define FGRN(x) KGRN x RST
#define FYEL(x) KYEL x RST
#define FBLU(x) KBLU x RST
#define FMAG(x) KMAG x RST
#define FCYN(x) KCYN x RST
#define FWHT(x) KWHT x RST

#define BOLD(x) "\x1B[1m" x RST
#define UNDL(x) "\x1B[4m" x RST

#endif  /* _COLORS_ */

 

////////////////////////////////////////////////

__global__ void myprint(float *nns, int size){
    int x = 3*(ni*nh+nh+nh*no+no);
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
    output = 1.0 / (1.0 + exp(-output));
    return output;
}

__global__ void play_game(float *nns, float *fitness, unsigned int *random_int_fruitx, unsigned int *random_int_fruity,
                        int parameter_size){

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
    __shared__ int dir;
    dir = 0;
    __shared__ int dir_next;
    dir_next = 0;
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

    // if(id == 0)
    //     printf("%d:%d ",fruitx,fruity);

    fruit_index++;

    //snake length
    __shared__ int ntail;
    ntail = 3;

    __shared__ int tailx[max_snake_length];
    __shared__ int taily[max_snake_length];

    int total_steps;
    total_steps = 200;
    float total_reward;
    total_reward = 0;
    float reward;
    reward = 0;
    int steps;
    steps = 0;
    __shared__ int last_fruit_distance;
    last_fruit_distance = height;
    __shared__ float input[ni];
    __shared__ float hidden_output[nh];
    __shared__ float output[no];

    __shared__ int break_flag;
    break_flag = 0;
    
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

            float max_value1 = INT16_MIN;
            float max_index1 = -1;
            for(int i=0;i<no;i++){
                if(i != max_index && output[i] > max_value1){
                    max_value1 = output[i];
                    max_index1 = i;
                }
            }
            dir_next = max_index1 + 1;
        // }

        // __syncthreads();

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

            // if(parameter_id == 0){
            //     float fruit_distance = 0;
            //     float distance_x = abs(x - fruitx);
            //     float distance_y = abs(y - fruity);
            //     fruit_distance = sqrt(pow(distance_x,2) + pow(distance_y,2));
                
            //     if(fruit_distance < last_fruit_distance)
            //         reward = 0.1;
            //     else
            //         reward = -0.2;

            //     last_fruit_distance = fruit_distance;
            // }

            // __syncthreads();

            if(x >= width || x < 0 || y >= height || y < 0)
            {
                reward = negative_reward;
                break_flag = 1;
            }

            for(int i =0; i<ntail;i++)
            {
                if(tailx[i]==x && taily[i]==y)
                {
                    reward = negative_reward;
                    break_flag = 1;
                }
            }

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
                // break;
            }

            reward = 0;

            if(reward > 0)
                total_steps = (total_steps+100 > max_total_steps) ? max_total_steps : total_steps + 100;

            if(steps > total_steps){
                break_flag = 1;
                // break;
            }
        }
        __syncthreads();
        if(break_flag)
            break;
    }
    
    __syncthreads();

    if(parameter_id == 0){
        // fitness[snake_id] = pow(3,(total_reward+2)) + steps;
        fitness[snake_id] = total_reward + steps;
        // if(gen == 2 && snake_id == 5){
        //     fitness[snake_id] = 2;
        // }
    }

    // __syncthreads();

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

__global__ void myprint1(float *fitness, int *indices){
    for(int i=0;i<population_size;i++)
        printf("%f\t%d\n",fitness[i],indices[i]);
}

__global__ void intialise_indices(int *indices){
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    indices[id] = id;
}

__global__ void crossover(float *nns, float *fitness, unsigned int *random_int1, unsigned int *random_int2, int top){
    int snake_id = blockIdx.x;
    int parameter_id = threadIdx.x;

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

    // int top = population_size * natural_selection_rate;

    // child index
    int child = top + snake_id;

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // int rand_num = random_int2[snake_id] % blockDim.x;
    int rand_num = random_int2[id];
    if(rand_num%2 == 0){
        nns[child * blockDim.x + parameter_id] = nns[parent1 * blockDim.x + parameter_id];
    }
    else{
        nns[child * blockDim.x + parameter_id] = nns[parent2 * blockDim.x + parameter_id];
    }

    // if(parameter_id <= random_int2[snake_id] % blockDim.x){
    //     nns[(top + snake_id) * blockDim.x + parameter_id] = nns[(random_int1[snake_id] % top) * blockDim.x + parameter_id];
    // }
    // else{
    //     nns[(top + snake_id) * blockDim.x + parameter_id] = nns[(random_int1[snake_id + blockDim.x] % top) * blockDim.x + parameter_id];
    // }
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

// game variables
bool gameover;
int dir, dir_next, last_dir;
int score;
int x, y;
int fruitx, fruity;
int tailx[max_snake_length], taily[max_snake_length];
int ntail;
int steps;
int max_steps = 300;

void setup(){
    gameover=false;
    dir = 0;
    dir_next = 0;
    x = width/2;
    y = height/2;
    fruitx=rand() % width;
    fruity=rand() % height;
    ntail = 3;
    score = 0;
    steps = 0;
}

void draw(){

    system("clear");

    for(int i =0; i<width+2;i++)
        cout << FGRN("+"); 

    cout<<endl;

    for(int i=0;i<height;i++)
    {
        for(int j=0; j<width;j++)
        {
            if(j == 0)
                cout << FBLU("+"); 

            if(i==y && j==x)
            {
                cout << FGRN("0");
            }

            else if(i == fruity && j == fruitx)
            {
                cout << FGRN("*"); 
            }
            
            else
            {
                bool print = false;
                for(int k =0;k < ntail;k++)
                {
                    if(tailx[k] == j && taily[k] ==i)
                    {
                        cout<<FWHT("o"); 
                        print = true;
                    }
                }
                if(!print)
                {
                    cout<<" ";
                }

            }

            if(j==width-1)
                cout << FRED("+");
        }

        cout<<endl;
    }
    for(int i = 0;i<width+2;i++)
    cout << FBLU("+");

    cout<<endl;

    cout<< UNDL(FRED("Score:")) <<score<<"\t"<<endl; 

    cout<< FMAG("hi");
    cout<<x<<" "<<y<<" : "<<fruitx<<" "<<fruity<<" : "<<tailx[ntail-1]<<" "<<taily[ntail-1]<<endl;
 
}

void set_input1(float *input){
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

void forward(float *input, float *output, float*w1,
            float *w2, float *b1, float *b2){
    float *layer1 = (float *)malloc(nh*sizeof(float));
    for(int i=0;i<nh;i++){
        layer1[i] = 0;
        for(int j=0;j<ni;j++){
            layer1[i] += input[j]*w1[j*nh+i];
        }
        layer1[i] += b1[i];

        // sigmoid activation
        layer1[i] = 1 / (1 + exp(-layer1[i]));
    }

    // for(int i=0;i<nh;i++)
    //     cout<<layer1[i]<<" ";
    // cout<<endl;

    for(int i=0;i<no;i++){
        output[i] = 0;
        for(int j=0;j<nh;j++){
            output[i] += layer1[j]*w2[j*no+i];
        }
        output[i] += b2[i];

        // sigmoid activation
        output[i] = 1 / (1 + exp(-output[i]));
    }

    // for(int i=0;i<no;i++)  
    //     cout<<output[i]<<" ";
    // cout<<endl;

    free(layer1);
}

void get_direction(float *parameters){
    float *w1 = &parameters[0];
    float *b1 = &parameters[ni*nh];
    float *w2 = &parameters[ni*nh+nh];
    float *b2 = &parameters[ni*nh+nh+nh*no];

    float *input = (float *)malloc(ni*sizeof(float));
    set_input1(input);

    float *output = (float *)malloc(no*sizeof(float));

    forward(input,output,w1,w2,b1,b2);

    // for(int i=0;i<no;i++)
    //     cout<<output[i]<<" ";
    // cout<<endl;

    int index = -1;
    float max = INT16_MIN;
    for(int j=0;j<no;j++){
        if(output[j] > max){
            max = output[j];
            index = j;
        }
    }

    dir = index + 1;

    int index1 = -1;
    float max1 = INT16_MIN;
    for(int j=0;j<no;j++){
        if(output[j] > max1 && j != index){
            max1 = output[j];
            index1 = j;
        }
    }

    dir_next = index1 + 1;

    free(input);
    free(output);
}

void logic(){

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

    if(x >= width || x < 0 || y >= height || y < 0)
    {
        gameover=true;
        cout<<"GAME OVER"<<endl;
    }

    for(int i =0; i<ntail;i++)
    {
        if(tailx[i]==x && taily[i]==y)
        {
            gameover = true;
            cout<<"GAME OVER"<<endl;
        }
    }

    if(x==fruitx && y==fruity)
    {
        score = score + 1;
        fruitx=rand() % width;
        fruity=rand() % height;
        ntail++;
    }
}

void view_game(float *parameters){
    setup();
    while(!gameover && steps < max_steps)
    {
        draw();
        get_direction(parameters);
        logic();
        usleep(150000);
        steps++;
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
	curandSetPseudoRandomGeneratorSeed(prng, 41ULL);
	
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

    float *dfitness;
    int *dindices;

	// fitness score on host
	float *fitness = (float *) malloc(population_size*sizeof(float));

	// fitness score on device
	cudaMalloc((void**) &dfitness,population_size*sizeof(float));
	cudaMalloc((void**) &dindices,population_size*sizeof(int));

    thrust::device_ptr<float> fitness_ptr(dfitness);
    thrust::device_ptr<int> indices_ptr(dindices);

    unsigned int *random_int_fruitx;
    cudaMalloc((void**) &random_int_fruitx,population_size*max_snake_length*sizeof(int));
    unsigned int *random_int_fruity;
    cudaMalloc((void**) &random_int_fruity,population_size*max_snake_length*sizeof(int));

    unsigned int *random_int_crossover1;
    cudaMalloc((void**) &random_int_crossover1,2*population_size*sizeof(int));
    unsigned int *random_int_crossover2;
    cudaMalloc((void**) &random_int_crossover2,population_size*parameter_size*sizeof(int));

    float *random_float_mutate1;
    cudaMalloc((void**) &random_float_mutate1,population_size*parameter_size*sizeof(float));
    float *random_float_mutate2;
    cudaMalloc((void**) &random_float_mutate2,population_size*parameter_size*sizeof(float));
    
    float max_reward = 0;
    float avg_reward = 0;
    int max_index = 0;
    float global_max_reward = 0;
    int global_max_generation = 0;

    float *best_snake = (float *)malloc(parameter_size*sizeof(float));

    // cudaStream_t stream1;
    // cudaStreamCreate(&stream1);

    // int k=0;
    // while(max_reward < 10){
    for(int k=0;k<generations;k++){
        // cout<<"Generation: "<<k+1<<endl;

        // intialise indices array corresponding to fitness array
        int num_threads = (population_size > 1024) ? 1024 : population_size;
        int num_blocks = population_size/1024 + 1;
        intialise_indices<<<num_blocks,num_threads>>>(dindices);

        // curandSetStream(prng,stream1);

        // create random number generator for integer values of fruit
        curandGenerate(prng,random_int_fruitx,population_size*max_snake_length);
        curandGenerate(prng,random_int_fruity,population_size*max_snake_length);
        
        play_game<<<population_size,parameter_size,parameter_size*sizeof(float)>>>(dnns,dfitness,random_int_fruitx,random_int_fruity,parameter_size);
        
        // cudaDeviceSynchronize();

        cudaMemcpy(fitness,dfitness,population_size*sizeof(float),cudaMemcpyDeviceToHost);

        // cudaDeviceSynchronize();
        
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
        
        printf("generation: %d\tAverage fitness: %f\tMax reward: %f\n",k+1,avg_reward,max_reward);

        if(max_reward > global_max_reward){
            global_max_reward = max_reward;
            global_max_generation = k+1;
        }

        cudaMemcpy(best_snake,dnns+max_index*parameter_size,parameter_size*sizeof(float),cudaMemcpyDeviceToHost);

        // cudaDeviceSynchronize();

        if(k > threshold_view_game)
            view_game(best_snake);

        // printf("Score: %d\n",score);
        
        int top = population_size * natural_selection_rate;

        // int top = 0;
        // for(int i=0;i<population_size;i++)
        //     if(fitness[i] > 0)
        //         top++;

        thrust::sort_by_key(fitness_ptr,fitness_ptr+population_size,indices_ptr,thrust::greater<float>());

        // cudaDeviceSynchronize();

        // myprint1<<<1,1>>>(dfitness,dindices);

        select_top<<<top,parameter_size>>>(dnns,dnns_new,dindices);

        float *temp = dnns_new;
        dnns_new = dnns;
        dnns = temp;

        curandGenerate(prng,random_int_crossover1,2*population_size);
        curandGenerate(prng,random_int_crossover2,population_size*parameter_size);

        curandGenerateUniform(prng,random_float_mutate1,population_size*parameter_size);
        curandGenerateNormal(prng,random_float_mutate2,population_size*parameter_size,0.0,1.0);

        // cudaDeviceSynchronize();

        // cudaStreamSynchronize(stream1);
        // myprint<<<1,1>>>(dnns,parameter_size);

        // cudaDeviceSynchronize();
        
        crossover<<<population_size-top,parameter_size>>>(dnns,dfitness,random_int_crossover1,random_int_crossover2,top);

        // cudaDeviceSynchronize();

        mutate<<<population_size,parameter_size>>>(dnns,random_float_mutate1,random_float_mutate2);

        // myprint<<<1,1>>>(dnns,parameter_size);

        // cudaDeviceSynchronize();
        
        // // increment number of generations
        // k++;
    }

    printf("Generation: %d\tGlobal max reward: %f\n",global_max_generation,global_max_reward);

    // cudaStreamDestroy(stream1);
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