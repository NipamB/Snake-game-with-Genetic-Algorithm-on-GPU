#include<cuda.h>
#include<iostream>
#include<math.h>
#include<random>
#include<vector>
#include<algorithm>
#include<curand.h>
#include<curand_kernel.h>
using namespace std;

#define ni 24
#define nh 50
#define no 4
#define width 30
#define height 40

// const unsigned ni = 24;
// const unsigned no = 4;
// const unsigned nh = 50;
const unsigned ps = 500;		// population size
const double nsr = 0.4;		// natural selection rate
const double mr = 0.05;		// mutation rate
const unsigned gn = 200;		// generation number

struct neuralNetwork{
	double w1[ni*nh];
	double w2[nh*no];
	double b1[nh];
	double b2[no];
	int reward;
};

void initialise_network(double w1[], double w2[],
						double b1[], double b2[]){

	for(int i=0;i<ni;i++){
		for(int j=0;j<nh;j++){
			w1[i*nh+j] = (double) rand() / RAND_MAX;
			if(rand() % 2)
				w1[i*nh+j] *= -1;
		}
	}

	for(int i=0;i<nh;i++){
		for(int j=0;j<no;j++){
			w2[i*no+j] = (double) rand() / RAND_MAX;
			if(rand() % 2)
				w2[i*no+j] *= -1;
		}
	}

	for(int i=0;i<nh;i++){
		b1[i] = (double) rand() / RAND_MAX;
		if(rand() % 2)
			b1[i] *= -1;
	}

	for(int i=0;i<no;i++){
		b2[i] = (double) rand() / RAND_MAX;
		if(rand() % 2)
			b2[i] *= -1;
	}
}

__device__ int forward(double *inp, double *w1, double *w2,
						double *b1, double *b2){

	double *layer1;
	layer1 = (double *)malloc(nh*sizeof(double));
	for(int i=0;i<nh;i++){
		layer1[i] = 0;
		for(int j=0;j<ni;j++){
			layer1[i] += inp[j]*w1[j*nh+i];
		}
		layer1[i] += b1[i];

		// activation
		layer1[i] = 1 / (1 + exp(-layer1[i]));
	}

	// output layer
	double *output;
	output = (double *)malloc(no*sizeof(double));
	for(int i=0;i<no;i++){
		output[i] = 0;
		for(int j=0;j<nh;j++){
			output[i] += layer1[j]*w2[j*no+i];
		}
		output[i] += b2[i];

		// activation
		output[i] = 1 / (1 + exp(-output[i]));
	}

	// for(int i=0;i<no;i++)
	// 	printf("%f ",output[i]);
	// printf("\n");

	free(layer1);
	free(output);

	int index = -1;
	double max = 0;
	for(int j=0;j<no;j++){
		if(output[j] > max){
			max = output[j];
			index = j;
		}
	}
	// printf("%f ",max);

	return index;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed ){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init ( seed, idx, 0, &state[idx] );
}

__device__ int logic(int dir, int &x, int &y,int tailx[], int taily[], int &ntail,
                      int &fruitx, int &fruity, bool &gameover, int &score, curandState localState){
    
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
        gameover=true;
        return -1;
    }

    for(int i =0; i<ntail;i++)
    {
        if(tailx[i]==x && taily[i]==y)
        {
            gameover = true;
            return -1;
        }
    }

    if(x==fruitx && y==fruity)
    {
        score = score + 500;
        fruitx = curand_uniform(&localState) * 1000;
        fruitx = fruitx % width;
        fruity = curand_uniform(&localState) * 1000;
        fruity = fruity % height;
        ntail++;
        return 1;
    }
    return 0;
}

__device__ void set_input(double input[], int x, int y,int fruitx,int fruity,
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
    // input[0] = is up is blocked, input[1] = if right is blocked, input[2] = if left is blocked, input[3] = if down is blocked
    // input[4] = apple_x, input[5] = apple_y, input[6] = snake_x, input[7] = snake_y
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

__device__ float calculate_fitness(int reward, int steps){
    if(reward < 5) {
        return (steps * steps) * pow(2,reward); 
    } 
	else{
        float fitness = (steps * steps);
        fitness *= pow(2,5);
        fitness *= (reward-9);
		return fitness;
    }
}

__global__ void playGame(double *dw1, double *dw2, double *db1, double *db2,
						 double *dr, curandState *globalState){

	unsigned id = blockIdx.x * blockDim.x + threadIdx.x;

	curandState localState = globalState[id];

	double *w1 = (double *)malloc(ni*nh*sizeof(double));
	int start = id*ni*nh;
	int end = start + ni*nh;
	for(int i=start;i<end;i++)
		w1[i-start] = dw1[i];

	double *w2 = (double *)malloc(nh*no*sizeof(double));
	start = id*nh*no;
	end = start + nh*no;
	for(int i=start;i<end;i++)
		w2[i-start] = dw2[i];

	double *b1 = (double *)malloc(nh*sizeof(double));
	start = id*nh;
	end = start + nh;
	for(int i=start;i<end;i++)
		b1[i-start] = db1[i];

	double *b2 = (double *)malloc(no*sizeof(double));
	start = id*no;
	end = start + no;
	for(int i=start;i<end;i++)
		b2[i-start] = db2[i];	

	// __syncthreads();
		     
    // setup the game
    bool gameover = false;
    // STOP: 0, LEFT: 1, RIGHT: 2, UP: 3, DOWN: 4
    int dir = 0;
    // position of head
    int x = width/2;
    int y = height/2;
    // position of fruit
    int fruitx = curand_uniform(&localState) * 1000;
    fruitx = fruitx % width;
    int fruity = curand_uniform(&localState) * 1000;
    fruity = fruity % height;
    //snake length
    int ntail = 2;
    int score = 0;
    int tailx[100], taily[100];

    int total_steps = 100;
	double total_reward = 0;
	double reward = 0;
	int steps = 0;
	double input[ni];
	while(!gameover){
		set_input(input,x,y,fruitx,fruity,tailx,taily,ntail);

		// double output[no];
		int index = forward(input,w1,w2,b1,b2);	
		// int index = 0;
        // printf("%d : %d, ",index,id);
        
        dir = index + 1;    // dir = 0 is STOP

        reward = logic(dir,x,y,tailx,taily,ntail,fruitx,fruity,gameover,score,localState);
        
		total_reward += reward;
        steps += 1;
        
        if(reward > 0)
            total_steps = (total_steps+50 > 300) ? 300 : total_steps + 50;

        if(steps > total_steps)
            break;

		// free(input);
		// free(output);

	}

	dr[id] = calculate_fitness(total_reward,steps);

	// __syncthreads();

	free(w1);
	free(w2);
	free(b1);
	free(b2);

	if(id == 0){
        // printf("%f\n",total_reward);
        // printf("ntail : %d\n",ntail);
        // printf("steps : %d\n",steps);
        // printf("x and y: %d and %d\n",x,y);
        // printf("score: %d\n",score);
        // printf("reward: %d\n",total_reward);
		// printf("fitness: %f\n",dr[id]);
		// for(int i=0;i<no;i++)
		// 	printf("%f ",b2[i]);
		// printf("\n");
	}
}

__global__ void crossover(double *dtw1, double *dtw2, double *dtb1, double *dtb2,
						double *dw1, double *dw2, double *db1, double *db2,
						double *dtr, int top, unsigned ps, curandState *globalState){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0;
	for(int i=0;i<top;i++)
		sum += dtr[i];
	int reward_sum = sum;
	if(sum < 0)
		reward_sum = 0;

	curandState localState = globalState[id];

	int rand_num = curand_uniform( &localState ) * 10000;
	rand_num = rand_num % reward_sum;
	// printf("%d\n",rand_num);

	// select parent 1
	int p1 = 0;
	sum = 0;
	for(int i=0;i<top;i++){
		sum += dtr[i];
		if(sum > rand_num){
			p1 = i;
			break;
		}
	}

	rand_num = curand_uniform( &localState ) * 10000;
	rand_num = rand_num % reward_sum;
	// printf("%d\n",rand_num);

	// select parent 2
	int p2 = 0;
	sum = 0;
	for(int i=0;i<top;i++){
		sum += dtr[i];
		if(sum > rand_num){
			p2 = i;
			break;
		}
	}

	// printf("%d %d\n",p1,p2);

	// crossover
	int randR = curand_uniform( &localState ) * 10000;
	randR = randR % ni;
	int randC = curand_uniform( &localState ) * 10000;
	randC = randC % nh;

	int startC = id*ni*nh;
	int startP1 = p1*ni*nh;
	int startP2 = p2*ni*nh;

	for(int i=0;i<ni;i++){
		for(int j=0;j<nh;j++){
			if((i < randR) || (i == randR && j <= randC))
				dw1[startC+i*nh+j] = dtw1[startP1+i*nh+j];
			else
				dw1[startC+i*nh+j] = dtw1[startP2+i*nh+j];
		}
	}

	randR = curand_uniform( &localState ) * 10000;
	randR = randR % nh;
	randC = curand_uniform( &localState ) * 10000;
	randC = randC % no;

	startC = id*nh*no;
	startP1 = p1*nh*no;
	startP2 = p2*nh*no;

	for(int i=0;i<nh;i++){
		for(int j=0;j<no;j++){
			if((i < randR) || (i == randR && j <= randC))
				dw2[startC+i*no+j] = dtw2[startP1+i*no+j];
			else
				dw2[startC+i*no+j] = dtw2[startP2+i*no+j];
		}
	}

	randR = curand_uniform( &localState ) * 10000;
	randR = randR % nh;

	startC = id*nh;
	startP1 = p1*nh;
	startP2 = p2*nh;

	for(int i=0;i<nh;i++){
		if(i < randR)
			db1[startC+i] = dtb1[startP1+i];
		else
			db1[startC+i] = dtb1[startP2+i];
	}

	randR = curand_uniform( &localState ) * 10000;
	randR = randR % no;

	startC = id*no;
	startP1 = p1*no;
	startP2 = p2*no;

	for(int i=0;i<no;i++){
		if(i < randR)
			db2[startC+i] = dtb2[startP1+i];
		else
			db2[startC+i] = dtb2[startP2+i];
	}

	// __syncthreads();

	if(id == 0){

		for(int i=0;i<top*ni*nh;i++)
			dw1[(ps-top)*ni*nh + i] = dtw1[i];

		for(int i=0;i<top*nh*no;i++)
			dw2[(ps-top)*nh*no + i] = dtw2[i];

		for(int i=0;i<top*nh;i++)
			db1[(ps-top)*nh + i] = dtb1[i];

		for(int i=0;i<top*no;i++)
			db2[(ps-top)*no + i] = dtb2[i];

		// for(int i=0;i<no;i++)
		// 	printf("%f ",db2[(ps-top)*no+i]);
		// printf("\n");
	}
}

__global__ void mutate(double *dw1, double *dw2, double *db1, double *db2, 
					   curandState *globalState, double mr){
	
	//////////////// add gaussian random number generator //////////////////

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	curandState localState = globalState[id];

	for(int i=0;i<ps*ni*nh;i++){
		double rand_num = curand_uniform( &localState );
		if(mr < rand_num){
			double rand_add = curand_uniform( &localState ) / 5;
			// int rand_check = rand_num * 1000;
			// if(rand_check % 2 != 0)
			// 	rand_add *= -1; 
			dw1[id*ni*nh+i] += rand_add;
		}
		if(dw1[id*ni*nh] > 1)
			dw1[id*ni*nh] = 1;
		else if(dw1[id*ni*nh] < -1)
			dw1[id*ni*nh] = -1;
	}

	for(int i=0;i<ps*nh*no;i++){
		double rand_num = curand_uniform( &localState );
		if(rand_num < mr){
			double rand_add = curand_uniform( &localState ) / 5;
			// int rand_check = rand_num * 1000;
			// if(rand_check % 2 != 0)
			// 	rand_add *= -1;
			dw2[id*nh*no+i] += rand_add;
		}
		if(dw2[id*nh*no] > 1)
			dw2[id*nh*no] = 1;
		else if(dw2[id*nh*no] < -1)
			dw2[id*nh*no] = -1;
	}

	for(int i=0;i<ps*nh;i++){
		double rand_num = curand_uniform( &localState );
		if(rand_num < mr){
			double rand_add = curand_uniform( &localState ) / 5;
			// int rand_check = rand_num * 1000;
			// if(rand_check % 2 != 0)
			// 	rand_add *= -1; 
			db1[id*nh+i] += rand_add;
		}
		if(db1[id*nh] > 1)
			db1[id*nh] = 1;
		else if(db1[id*nh] < -1)
			db1[id*nh] = -1;
	}

	for(int i=0;i<ps*no;i++){
		double rand_num = curand_uniform( &localState );
		if(rand_num < mr){
			double rand_add = curand_uniform( &localState ) / 5;
			// int rand_check = rand_num * 1000;
			// if(rand_check % 2 != 0)
			// 	rand_add *= -1; 
			db2[id*no+i] += rand_add;
		}
		if(db2[id*no] > 1)
			db2[id*no] = 1;
		else if(db2[id*no] < -1)
			db2[id*no] = -1;
	}
}

bool comp(pair<double,int> p1, pair<double,int> p2){
	return p1.first > p2.first;
}

int main(int argc, char const *agrv[]){
	srand(time(0));

	vector<struct neuralNetwork> nns;
	vector<struct neuralNetwork> nns_new;

	// host neural network parameters in the form of 1D array
	double *hw1 = (double *)malloc(ps*ni*nh*sizeof(double));
	double *hw2 = (double *)malloc(ps*nh*no*sizeof(double));
	double *hb1 = (double *)malloc(ps*nh*sizeof(double));
	double *hb2 = (double *)malloc(ps*no*sizeof(double));
	// double hw1[ps * ni * nh];
	// double hw2[ps * nh * no];
	// double hb1[ps * nh];
	// double hb2[ps * no];

	int hw1_index = 0;
	int hw2_index = 0;
	int hb1_index = 0;
	int hb2_index = 0;

	// initialising the neural networks
	for(int i=0;i<ps;i++){
		struct neuralNetwork nn;
		initialise_network(nn.w1,nn.w2,nn.b1,nn.b2);
		nn.reward = 0;
		nns.push_back(nn);

		// copying the neural network parameters into 1D array
		for(int j=0;j<ni*nh;j++)
			hw1[hw1_index++] = nn.w1[j];
		for(int j=0;j<nh*no;j++)
			hw2[hw2_index++] = nn.w2[j];
		for(int j=0;j<nh;j++)
			hb1[hb1_index++] = nn.b1[j];
		for(int j=0;j<no;j++)
			hb2[hb2_index++] = nn.b2[j];
	}

	// selecting the best game
	neuralNetwork best_nn = nns[0];

	// number of top games to be crossovered
	int top = ps * nsr;

	// host neural network parameters of the top games 
	double *htw1 = (double *)malloc(top*ni*nh*sizeof(double));
	double *htw2 = (double *)malloc(top*nh*no*sizeof(double));
	double *htb1 = (double *)malloc(top*nh*sizeof(double));
	double *htb2 = (double *)malloc(top*no*sizeof(double));
	// double htw1[top * ni * nh];
	// double htw2[top * nh * no];
	// double htb1[top * nh];
	// double htb2[top * no];

	// device neural network parameters of the top games
	double *dtw1, *dtw2, *dtb1, *dtb2;
	cudaMalloc(&dtw1,top*ni*nh*sizeof(double));
	cudaMalloc(&dtw2,top*nh*no*sizeof(double));
	cudaMalloc(&dtb1,top*nh*sizeof(double));
	cudaMalloc(&dtb2,top*no*sizeof(double));

	// device neural network parameters of all games
	double *dw1, *dw2, *db1, *db2;
	cudaMalloc(&dw1,ps*ni*nh*sizeof(double));
	cudaMalloc(&dw2,ps*nh*no*sizeof(double));
	cudaMalloc(&db1,ps*nh*sizeof(double));
	cudaMalloc(&db2,ps*no*sizeof(double));

	// copy neural network data to GPU
	cudaMemcpy(dw1,hw1,ps*ni*nh*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dw2,hw2,ps*nh*no*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(db1,hb1,ps*nh*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(db2,hb2,ps*no*sizeof(double),cudaMemcpyHostToDevice);

	// device state array to generate random number
	curandState *devStates;
	cudaMalloc(&devStates,ps*sizeof(curandState));

    // init kernel for random number generation
	setup_kernel<<<1,ps>>>(devStates,time(NULL));

	// device reward array for all the games
	double *dr;
	cudaMalloc(&dr,ps*sizeof(double));

	// init host reward array
	double *hr = (double *)malloc(ps*sizeof(double));
	for(int i=0;i<ps;i++)
		hr[i] = 0;

	// copy reward aaray to GPU
	cudaMemcpy(dr,hr,ps*sizeof(double),cudaMemcpyHostToDevice);

	// reward array for top games
	double *htr = (double *)malloc(top*sizeof(double));

	// iterate for number of generations
	for(int k=0;k<gn;k++){
		// cout<<k<<endl;	

		// play the game for the population
		playGame<<<1,ps>>>(dw1,dw2,db1,db2,dr,devStates);

		// cudaDeviceSynchronize();

		// get the reawrd array to CPU
		cudaMemcpy(hr,dr,ps*sizeof(double),cudaMemcpyDeviceToHost);

		// cudaDeviceSynchronize();

        float avg = 0;
        for(int i=0;i<ps;i++){
            // cout<<hr[i]<<" ";
            avg += hr[i];
        }
        // cout<<endl;
        avg /= ps;
        cout<<"Genertion number: "<<k+1<<"\tAverage fitness: "<<avg<<endl;

		// compute the top games
		vector<pair<double,int>> rp;
		for(int i=0;i<ps;i++){
			rp.push_back(make_pair(hr[i],i));
		}

		/////////// can use thrust also /////////////
		sort(rp.begin(),rp.end(),comp);

		// for(int i=0;i<ps;i++)
		// 	cout<<rp[i].first<<"\t"<<rp[i].second<<endl;

		for(int i=0;i<top;i++)
			htr[i] = rp[i].second;

		double *dtr;
		cudaMalloc(&dtr,top*sizeof(double));
		cudaMemcpy(dtr,htr,top*sizeof(double),cudaMemcpyHostToDevice);

		// cudaDeviceSynchronize();

		int htw1_index = 0;
		int htw2_index = 0;
		int htb1_index = 0;
		int htb2_index = 0;

		// init host neuaral network parameters for top games
		for(int i=0;i<top;i++){
			int it = rp[i].second;
			for(int j=0;j<ni*nh;j++)
				htw1[htw1_index++] = hw1[it*ni*nh+j];
			for(int j=0;j<nh*no;j++)
				htw2[htw2_index++] = hw2[it*nh*no+j];
			for(int j=0;j<nh;j++)
				htb1[htb1_index++] = hb1[it*nh+j];
			for(int j=0;j<no;j++)
				htb2[htb2_index++] = hb2[it*no+j];
		}

		// copy neural network parameters of top games to GPU 
		cudaMemcpy(dtw1,htw1,top*ni*nh*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(dtw2,htw2,top*nh*no*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(dtb1,htb1,top*nh*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(dtb2,htb2,top*no*sizeof(double),cudaMemcpyHostToDevice);

		// cudaDeviceSynchronize();

		// crossover to generate new population 
		crossover<<<1,ps-top>>>(dtw1,dtw2,dtb1,dtb2,dw1,dw2,db1,db2,dtr,top,ps,devStates);

		// cudaDeviceSynchronize();

		// mutate the population
		// mutate<<<1,ps>>>(dw1,dw2,db1,db2,devStates,mr);
	}

	cudaFree(dw1);
	cudaFree(dw2);
	cudaFree(db1);
	cudaFree(db2);
	cudaFree(dtw1);
	cudaFree(dtw2);
	cudaFree(dtb1);
	cudaFree(dtb2);
	cudaFree(devStates);
	cudaFree(dr);

	return 0;
}