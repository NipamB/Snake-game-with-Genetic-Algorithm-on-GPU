#include<iostream>
#include<math.h>
#include<algorithm>
#include<vector>
#include<random>
#include<unistd.h>
#include<fstream>

using namespace std;

// global varibales for model
const int n_input = 24;
const int n_output = 4;
const int n_hidden = 20;
const int population_size = 4096;
const float natural_selection_ratio = 0.2;
const float mutation_rate = 0.01;
const int generation_num = 150;
const int max_snake_length = 100;
const int positive_reward = 500;
const int negative_reward = -150;
const int threshold_view_game = 300;

// random number generator for normal distribution
static std::random_device __randomDevice;
static std::mt19937 __randomGen(__randomDevice());
static std::normal_distribution<float> __normalDistribution(0.5, 1);

// structure for neural network
struct neuralNetwork{
	float w1[n_input][n_hidden];
	float w2[n_hidden][n_output];
	float b1[n_hidden];
	float b2[n_output];
	int reward;
};


// global variables for game
bool gameover;
const int width=30;
const int height=20;
int x,y,fruitx,fruity,score;
int tailx[max_snake_length],taily[max_snake_length];
int ntail;

enum edirection{STOP=0,LEFT,RIGHT,UP,DOWN};

edirection dir, last_dir=STOP, dir_next;


////////// game code /////////

// initialise variables to start the game
void setup()
{
	// flag to exit the game
    gameover=false;

	// direction to be taken next
    dir = STOP;
	dir_next = STOP;

	// head of the snake
    x = width/2;
    y = height/2;

	// position of fruit 
    fruitx=rand() % width;
    fruity=rand() % height;

	// size of the snake
	ntail = 3;

	// game score
    score = 0;
}

// logic of the game
int logic(int steps)
{
	// update values of the snake body
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

	// move the snake based on the direction
    switch(dir)
    {
        case LEFT:
			if(last_dir != RIGHT)
				x--;
			else{
				if(dir_next == RIGHT)
					x++;
				else if(dir_next == UP)
					y--;
				else if(dir_next == DOWN)
					y++;
			}
			break;
		case RIGHT:
			if(last_dir != LEFT)
				x++;
			else{
				if(dir_next == LEFT)
					x--;
				else if(dir_next == UP)
					y--;
				else if(dir_next == DOWN)
					y++;
			}
			break;
		case UP:
			if(last_dir != DOWN)
				y--;
			else{
				if(dir_next == LEFT)
					x--;
				else if(dir_next == RIGHT)
					x++;
				else if(dir_next == DOWN)
					y++;
			}
			break;
		case DOWN:
			if(last_dir != UP)
				y++;
			else{
				if(dir_next == LEFT)
					x--;
				else if(dir_next == RIGHT)
					x++;
				else if(dir_next == UP)
					y--;
			}
			break;
    }

	// snake hits the wall
    if(x >= width || x < 0 || y >= height || y < 0)
    {
        gameover=true;
		return negative_reward;
    }

	// snake hits its body
    for(int i =0; i<ntail;i++)
    {
        if(tailx[i]==x && taily[i]==y)
        {
            gameover = true;
			return negative_reward;
        }
    }

	// snake eats the fruit
    if(x==fruitx && y==fruity)
    {
        score = score + 500;
		fruitx=rand() % width;
		fruity=rand() % height;
		ntail++;
		return positive_reward;
    }

	return 0;
}


////////// model code ////////

// initialise the neural network with random values
void initialise_network(float w1[][n_hidden], float w2[][n_output],
						float b1[], float b2[]){

	for(int i=0;i<n_input;i++){
		for(int j=0;j<n_hidden;j++){
			w1[i][j] = (double) rand() / RAND_MAX;
			if(rand() % 2)
				w1[i][j] *= -1;
		}
	}

	for(int i=0;i<n_hidden;i++){
		for(int j=0;j<n_output;j++){
			w2[i][j] = (double) rand() / RAND_MAX;
			if(rand() % 2)
				w2[i][j] *= -1;
		}
	}

	for(int i=0;i<n_hidden;i++){
		b1[i] = (double) rand() / RAND_MAX;
		if(rand() % 2)
			b1[i] *= -1;
	}

	for(int i=0;i<n_output;i++){
		b2[i] = (double) rand() / RAND_MAX;
		if(rand() % 2)
			b2[i] *= -1;
	}
}


float *forward(float *input, float *output, float w1[][n_hidden], float w2[][n_output],
						float b1[], float b2[]){

	// forward pass for first layer
	float *layer1;
	layer1 = (float *)malloc(n_hidden*sizeof(float));
	for(int i=0;i<n_hidden;i++){
		layer1[i] = 0;
		for(int j=0;j<n_input;j++){
			layer1[i] += input[j]*w1[j][i];
		}
		layer1[i] += b1[i];

		// sigmoid activatiion
		layer1[i] = 1 / (1 + exp(-layer1[i]));
	}

	// forward pass for second layer and thus get the output layer
	for(int i=0;i<n_output;i++){
		output[i] = 0;
		for(int j=0;j<n_hidden;j++){
			output[i] += layer1[j]*w2[j][i];
		}
		output[i] += b2[i];

		// sigmoid activatiion
		// output[i] = 1 / (1 + exp(-output[i]));
	}

	// softmax activation on the output layer
	float exp_sum = 0;
	for(int i=0;i<n_output;i++)
		exp_sum += exp(output[i]);

	for(int i=0;i<n_output;i++)
		output[i] = exp(output[i]) / exp_sum;

	free(layer1);
	return output;
}

// select parent based on Roulette Wheel Selection method
int selectParent(vector<struct neuralNetwork> nns, int reward_sum){
	if(reward_sum == 0)
		return 0;
	int rand_num = rand() % reward_sum;
	int sum = 0;
	for(int i=0;i<nns.size();i++){
		sum += nns[i].reward;
		if(sum > rand_num)
			return i;
	}
	return 0;
}

// crossover two parent neural networks to generate new child neural network
struct neuralNetwork crossover(struct neuralNetwork parent1,
								struct neuralNetwork parent2){
	struct neuralNetwork child;
	child.reward = 0;

	int randR = rand() % n_input;
	int randC = rand() % n_hidden;

	for(int i=0;i<n_input;i++){
		for(int j=0;j<n_input;j++){
			if((i < randR) || (i == randR && j <= randC))
				child.w1[i][j] = parent1.w2[i][j];
			else
				child.w1[i][j] = parent2.w2[i][j];
		}
	}

	randR = rand() % n_hidden;
	randC = rand() % n_output;

	for(int i=0;i<n_hidden;i++){
		for(int j=0;j<n_output;j++){
			if((i < randR) || (i == randR && j <= randC))
				child.w2[i][j] = parent1.w2[i][j];
			else
				child.w2[i][j] = parent2.w2[i][j];
		}
	}

	randR = rand() % n_hidden;
	for(int i=0;i<n_hidden;i++){
		if(i < randR)
			child.b1[i] = parent1.b1[i];
		else
			child.b1[i] = parent2.b1[i];
	}

	randR = rand() % n_output;
	for(int i=0;i<n_output;i++){
		if(i < randR)
			child.b2[i] = parent1.b2[i];
		else
			child.b2[i] = parent2.b2[i];
	}	

	return child;
}

// mutate neural network based on mutation rate
void mutate(struct neuralNetwork &nn){
	for(int i=0;i<n_input;i++){
		for(int j=0;j<n_hidden;j++){
			float check = (double) rand() / RAND_MAX;
			if(check < mutation_rate)
				nn.w1[i][j] += __normalDistribution(__randomGen) / 5;
			if(nn.w1[i][j] > 1)
				nn.w1[i][j] = 1;
			else if(nn.w1[i][j] < -1)
				nn.w1[i][j] = -1;
		}
	}

	for(int i=0;i<n_hidden;i++){
		for(int j=0;j<n_output;j++){
			float check = (double) rand() / RAND_MAX;
			if(check < mutation_rate)
				nn.w2[i][j] += __normalDistribution(__randomGen) / 5;
			if(nn.w2[i][j] > 1)
				nn.w2[i][j] = 1;
			else if(nn.w2[i][j] < -1)
				nn.w2[i][j] = -1;
		}
	}

	for(int i=0;i<n_hidden;i++){
		float check = (double) rand() / RAND_MAX;
		if(check < mutation_rate)
			nn.b1[i] += __normalDistribution(__randomGen) / 5;
		if(nn.b1[i] > 1)
			nn.b1[i] = 1;
		else if(nn.b1[i] < -1)
			nn.b1[i] = -1;
	}

	for(int i=0;i<n_output;i++){
		float check = (double) rand() / RAND_MAX;
		if(check < mutation_rate)
			nn.b2[i] += __normalDistribution(__randomGen) / 5;
		if(nn.b2[i] > 1)
			nn.b2[i] = 1;
		else if(nn.b2[i] < -1)
			nn.b2[i] = -1;
	}
}

// comparator function used for sorting neural networks
bool comp(neuralNetwork nn1, neuralNetwork nn2){
	return nn1.reward > nn2.reward;
}

// set input for foward pass, input size is 24 i.e it check in all 8 directions
// for food, its body and wall
void set_input(float *input){
	for(int i=0;i<n_input;i++)
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

int main(){
	srand(time(0));

	// write parameters of best neural network into file
	ofstream fout;
	fout.open("output.txt");

	// write model parameter values into file
	fout<<"n_input\t\t"<<n_input<<endl;
    fout<<"n_hidden\t"<<n_hidden<<endl;
    fout<<"n_output\t"<<n_output<<endl;
    fout<<"height\t\t"<<height<<endl;
    fout<<"width\t\t"<<width<<endl;

	// initial population
	vector<struct neuralNetwork> nns;
	vector<struct neuralNetwork> nns_new;

	// initialise all the neural networks with random values
	for(int i=0;i<population_size;i++){
		struct neuralNetwork nn;
		initialise_network(nn.w1,nn.w2,nn.b1,nn.b2);
		nn.reward = 0;
		nns.push_back(nn);
	}

	// stores the parameters of best neural network
	neuralNetwork best_nn = nns[0];

	// input to neural network
	float *input;
	input = (float *)malloc(n_input*sizeof(float));

	// output of neural network
	float *output;
	output = (float *)malloc(n_output*sizeof(float));

	// local variables
	int max_reward = 0;
	int max_steps = 0;
	int max_gen_num = 0;
	int first_score = 0;
	float fitness_avg = 0;
	float max_fitness_avg = 0;
	float global_max_reward = 0;
	int global_max_generation = 0;

	// loop for number of generations
	for(int k=0;k<generation_num;k++){
		// index of the best model
		int best_index = 0;

		// number of steps the snake can move
		int total_steps = 200;
		
		max_reward = 0;
		max_steps = 0;

		// play game for the population
		for(int i=0;i<population_size;i++){
			// setup the game before starting to play
			setup();

			int total_reward = 0;
			int reward = 0;
			int steps = 0;

			// loop while game is not over i.e sanke is not dead
			while(!gameover){
				// set the input to neural network
				set_input(input);

				// do forward pass to get the output 
				forward(input,output,nns[i].w1,nns[i].w2,nns[i].b1,nns[i].b2);

				// get the best direction based on output
				int index = -1;
				float max = INT16_MIN;
				for(int j=0;j<n_output;j++){
					if(output[j] > max){
						max = output[j];
						index = j;
					}
				}

				// set the direction
				if(index == 0)
					dir = LEFT;
				else if(index == 1)
					dir = RIGHT;
				else if(index == 2)
					dir = UP;
				else if(index == 3)
					dir = DOWN;

				// get the second best direction based on output
				int index1 = -1;
				float max1 = INT16_MIN;
				for(int j=0;j<n_output;j++){
					if(output[j] > max1 && j != index){
						max1 = output[j];
						index1 = j;
					}
				}

				// set the second best direction
				if(index1 == 0)
					dir_next = LEFT;
				else if(index1 == 1)
					dir_next = RIGHT;
				else if(index1 == 2)
					dir_next = UP;
				else if(index1 == 3)
					dir_next = DOWN;

				// get the reward based on the move
				reward = logic(steps);

				// update varibales
				last_dir = dir;

				total_reward += reward;
				steps += 1;

				if(reward < 0)
					break;

				// update total number of steps the snake can move
				if(reward > 0)
					total_steps = (total_steps+100 > 500) ? 500 : total_steps + 100;

				if(steps > total_steps)
					break;
			}
			
			// set the fitness score
			nns[i].reward = total_reward + steps;

			// calculate the best fitness score in the population
			if(nns[i].reward >= max_reward){
				max_reward = nns[i].reward;
				max_steps = steps;
				best_index = i;
			}
		}

		// calculate the best fitness score among all the generations
		if(max_reward > global_max_reward){
			global_max_reward = max_reward;
			global_max_generation = k+1;
		}
		
		// calculate average fitness score
		for(int i=0;i<population_size;i++)
			fitness_avg += nns[i].reward;
		fitness_avg /= population_size;

		cout<<"Generation number: "<<k+1<<"\tAverage Fitness: "<<fitness_avg<<"\tMax reward: "<<max_reward<<endl;

		// sort the neural networks based on fitness score
		sort(nns.begin(),nns.end(),comp);

		// set the best neural network based on average fitness score
		if(fitness_avg >= max_fitness_avg){
			max_fitness_avg = fitness_avg;
			best_nn = nns[0];
		}

		// number of neural networks selected for next generation from current generation
		int top_nns = population_size * natural_selection_ratio;
		
		// calculate sum of fitness score which will be used to select parent during crossover
		int reward_sum = 0;
		for(int i=0;i<top_nns;i++){
			nns_new.push_back(nns[i]);
			reward_sum += nns[i].reward;
		}

		nns.clear();
		for(int i=0;i<top_nns;i++)
			nns.push_back(nns_new[i]);

		// do crossover and mutation for the rest of the population using Roulette Wheel Selection method
		for(int i=0;i<population_size-top_nns;i++){
			// select parent 1
			int index1 = selectParent(nns_new,reward_sum);
			neuralNetwork parent1 = nns_new[index1];

			// select parent 2
			int index2 = selectParent(nns_new,reward_sum);
			neuralNetwork parent2 = nns_new[index2];

			// crossover parent 1 and parent 2 to generate the child neural network
			struct neuralNetwork child = crossover(parent1,parent2);

			// mutate the child based on mutation rate 
			mutate(child);
			
			nns.push_back(child);
		}

		nns_new.clear();
	}

	// write parameters of best neural network into file
	fout<<"Best neural network parameters:\n";
	for(int i=0;i<n_input;i++)
		for(int j=0;j<n_hidden;j++)
			fout<<best_nn.w1[i][j]<<" ";

	for(int i=0;i<n_hidden;i++)
		fout<<best_nn.b1[i]<<" ";

	for(int i=0;i<n_hidden;i++)
		for(int j=0;j<n_output;j++)
			fout<<best_nn.w2[i][j]<<" ";

	for(int i=0;i<n_output;i++)
		fout<<best_nn.b2[i]<<" ";
	
	fout<<endl;
	fout.close();
	
	cout<<"Max reward: "<<global_max_reward<<endl;
	cout<<"Generation num: "<<global_max_generation<<endl;

	return 0;
}