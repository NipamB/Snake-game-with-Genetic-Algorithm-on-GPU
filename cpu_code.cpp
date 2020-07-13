#include<iostream>
#include<math.h>
#include<algorithm>
#include<vector>
#include<random>
#include<unistd.h>

using namespace std;

// global varibales for model
const int n_input = 7;
const int n_output = 3;
const int n_hidden = 50;
const int population_size = 25;
const float natural_selection_ratio = 0.2;
const float mutation_rate = 0.05;
const int generation_num = 50;

static std::random_device __randomDevice;
static std::mt19937 __randomGen(__randomDevice());
static std::normal_distribution<float> __normalDistribution(0.5, 1);

struct neuralNetwork{
	float w1[n_input][n_hidden];
	float w2[n_hidden][n_output];
	float b1[n_hidden];
	float b2[n_output];
	int reward;
};


// global variables for game
bool gameover;
const int width=40;
const int height=40;
int x,y,fruitx,fruity,score;
int tailx[100],taily[100];
int ntail;


enum edirection{STOP=0,LEFT,RIGHT,UP,DOWN};

edirection dir;

// keep trach of last step
edirection last_dir = STOP;

unsigned int microseconds = 10000;

////////// game code /////////

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

// initialise variables to start the game
void setup()
{
    gameover=false;
    dir = STOP;
    x = width/2;
    y = height/2;

	//testing fruit position
	fruitx = x;
	fruity = y;
	if(rand()%2)
		fruitx += rand()%3;
	else
		fruitx -= rand()%3;
	
	if(rand()%2)
		fruity += rand()%3;
	else
		fruity -= rand()%3;

    // fruitx=rand() % width;
    // fruity=rand() % height;
	ntail = 0;
    score = 0;
}

// display the game
void draw(int gen_num, int pop_num)
{

    system("clear");

    for(int i =0; i<width+2;i++)
        cout << FGRN("#"); 

    cout<<endl;

    for(int i=0;i<height;i++)
    {
        for(int j=0; j<width;j++)
        {
            if(j == 0)
                cout << FBLU("#"); 

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
                cout << FRED("|");
        }

        cout<<endl;
    }
    for(int i = 0;i<width+2;i++)
    cout << FBLU("-");

    cout<<endl;

    cout<< UNDL(FRED("Score:")) <<score<<"\t"<<endl; 

    // cout<< FMAG("hi");
	cout<<"Generartion number: "<<gen_num+1<<"\tPopulation number: "<<pop_num<<endl;
    cout<<x<<" "<<y<<" : "<<fruitx<<" "<<fruity<<" : "<<tailx[ntail-1]<<" "<<taily[ntail-1]<<endl;
 
}

// logic of the game
int logic()
{

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
        case LEFT:
			if(last_dir == LEFT){
				y++;
				last_dir = DOWN;
			}
			else if(last_dir == RIGHT){
				y--;
				last_dir = UP;
			}
			else if(last_dir == UP){
				x--;
				last_dir = LEFT;
			}
			else if(last_dir == DOWN){
				x++;
				last_dir = RIGHT;
			}
			else if(last_dir == STOP){
				x--;
				last_dir = LEFT;
			}
            // x--;
            break;
        case RIGHT:
			if(last_dir == LEFT){
				y--;
				last_dir = RIGHT;
			}
			else if(last_dir == RIGHT){
				y++;
				last_dir = DOWN;
			}
			else if(last_dir == UP){
				x++;
				last_dir = RIGHT;
			}
			else if(last_dir == DOWN){
				x--;
				last_dir = LEFT;
			}
			else if(last_dir == STOP){
				x++;
				last_dir = RIGHT;
			}
            // x++;
            break;
        case UP:
			if(last_dir == LEFT){
				x--;
				last_dir = LEFT;
			}
			else if(last_dir == RIGHT){
				x++;
				last_dir = RIGHT;
			}
			else if(last_dir == UP){
				y--;
				last_dir = UP;
			}
			else if(last_dir == DOWN){
				y++;
				last_dir = DOWN;
			}
			else if(last_dir == STOP){
				y--;
				last_dir = UP;
			}
            // y--;
            break;
        // case DOWN:
        //     y++;
        //     break;
    }

	// if((dir == UP && last_dir == DOWN) || (dir == DOWN && last_dir == UP)){
	// 	gameover = true;
	// 	return -150;
	// }
	// else if((dir == LEFT && last_dir == RIGHT) || (dir == RIGHT && last_dir == LEFT)){
	// 	gameover = true;
	// 	return -150;
	// }

    if(x> width || x<0 || y > height || y<0)
    {
        gameover=true;
        // cout<<"GAME OVER"<<endl;
		return -100;
    }

    for(int i =0; i<ntail;i++)
    {
        if(tailx[i]==x && taily[i]==y)
        {
            gameover = true;
            // cout<<"GAME OVER"<<endl;
			return -100;
        }
    }

    if(x==fruitx && y==fruity)
    {
        score = score +500;

		// testing fruit position
		fruitx = x;
		fruity = y;
		int rand_num = rand();
		if(rand_num%2)
			fruitx = (fruitx + rand_num%3 + 1 > width) ? width : fruitx + rand_num%3 + 1;
			// fruitx = rand()%3 + 1;
		else
			fruitx = (fruitx - rand_num%3 - 1 < 0) ? 0 : fruitx - rand_num%3 - 1;
			// fruitx -= rand()%3 - 1;
		
		if(rand()%2)
			fruity = (fruity + rand_num%3 + 1 > height) ? height : fruity + rand_num%3 + 1;
			// fruity += rand()%3 + 1;
		else
			fruity = (fruity - rand_num%3 - 1 < 0) ? 0 : fruity - rand_num%3 - 1;
			// fruity -= rand()%3 - 1;

        // fruitx=rand() % width;
        // fruity=rand() % height;
        ntail++;
		return 500;
    }
	return -5;
}


////////// model code ////////

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

	// layer1
	float *layer1;
	layer1 = (float *)malloc(n_hidden*sizeof(float));
	for(int i=0;i<n_hidden;i++){
		layer1[i] = 0;
		for(int j=0;j<n_input;j++){
			layer1[i] += input[j]*w1[j][i];
		}
		layer1[i] += b1[i];

		// activatiion
		layer1[i] = 1 / (1 + exp(-layer1[i]));
	}

	// output layer
	// float *output;
	// output = (float *)malloc(n_output*sizeof(float));
	for(int i=0;i<n_output;i++){
		output[i] = 0;
		for(int j=0;j<n_hidden;j++){
			output[i] += layer1[j]*w2[j][i];
		}
		output[i] += b2[i];

		// activatiion
		output[i] = 1 / (1 + exp(-output[i]));
	}

	float exp_sum = 0;
	for(int i=0;i<n_output;i++)
		exp_sum += exp(output[i]);

	for(int i=0;i<n_output;i++)
		output[i] = exp(output[i]) / exp_sum;

	free(layer1);
	return output;
}

int selectParent(vector<struct neuralNetwork> nns, int reward_sum){
	int rand_num = rand() % reward_sum;
	int sum = 0;
	for(int i=0;i<nns.size();i++){
		sum += nns[i].reward;
		if(sum > rand_num)
			return i;
	}
	return 0;
}

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

bool comp(neuralNetwork nn1, neuralNetwork nn2){
	return nn1.reward > nn2.reward;
}

// input[0] = is front is blocked, input[1] = if right is blocked, input[2] = if left is blocked
// input[3] = apple_x, input[4] = apple_y, input[5] = snake_x, input[6] = snake_y
void set_input(float *input){
	int right_wall = (x+1 > width) ? 1 : 0;
	int left_wall = (x-1 < 0) ? 1 : 0;
	int up_wall = (y-1 < 0) ? 1 : 0;
	int down_wall = (y+1 > height) ? 1 : 0;

	if(last_dir == UP){
		input[0] = up_wall;
		input[1] = right_wall;
		input[2] = left_wall;
	}
	else if(last_dir == DOWN){
		input[0] = down_wall;
		input[1] = left_wall;
		input[2] = right_wall;
	}
	else if(last_dir == RIGHT){
		input[0] = right_wall;
		input[1] = down_wall;
		input[2] = up_wall;
	}
	else if(last_dir == LEFT){
		input[0] = left_wall;
		input[1] = up_wall;
		input[2] = down_wall;
	}
	else if(last_dir == STOP){
		input[0] = 0;
		input[1] = 0;
		input[2] = 0;
	}

	float min_value = INT16_MAX;
	float max_value = INT16_MIN;

	input[3] = fruitx;
	if(input[3] < min_value)
		min_value = input[3];
	if(input[3] > max_value)
		max_value = input[3];

	input[4] = fruity;
	if(input[4] < min_value)
		min_value = input[4];
	if(input[4] > max_value)
		max_value = input[4];
	
	input[5] = x;
	if(input[5] < min_value)
		min_value = input[5];
	if(input[5] > max_value)
		max_value = input[5];

	input[6] = y;
	if(input[6] < min_value)
		min_value = input[6];
	if(input[6] > max_value)
		max_value = input[6];

	input[3] -= min_value;
	input[3] /= max_value;

	input[4] -= min_value;
	input[4] /= max_value;

	input[5] -= min_value;
	input[5] /= max_value;

	input[6] -= min_value;
	input[6] /= max_value;
	// input[0] = x;
	// input[1] = y;
	// input[2] = tailx[ntail-1];
	// input[3] = taily[ntail-1];
	// input[4] = fruitx;
	// input[5] = fruity;
}

int main(){
	srand(time(0));

	// initial population
	vector<struct neuralNetwork> nns;
	vector<struct neuralNetwork> nns_new;

	for(int i=0;i<population_size;i++){
		struct neuralNetwork nn;
		initialise_network(nn.w1,nn.w2,nn.b1,nn.b2);
		nn.reward = 0;
		nns.push_back(nn);
	}

	neuralNetwork best_nn = nns[0];

	// input to neural network
	float *input;
	input = (float *)malloc(n_input*sizeof(float));

	// output of neural network
	float *output;
	output = (float *)malloc(n_output*sizeof(float));

	int max_reward = 0;
	int max_steps = 0;
	int max_gen_num = 0;

	for(int k=0;k<generation_num;k++){
		// cout<<"Generation number: "<<k<<endl;

		// index of the best model
		int best_index = 0;

		// number of steps the snake can move
		int total_steps = 200;
		
		// play game for the population
		for(int i=0;i<population_size;i++){
			// cout<<"Population no: "<<i+1<<endl;

			// setup the game before starting to play
			setup();

			int total_reward = 0;
			int reward = 0;
			int steps = 0;
			while(!gameover){
				// if(i == 0){
				draw(k,i);
				usleep(150000);
				// }
				// for(int i=0;i<n_input;i++)
				// 	input[i] = (double) rand() / RAND_MAX;

				// setting input
				set_input(input);

				// float *output;
				forward(input,output,nns[i].w1,nns[i].w2,nns[i].b1,nns[i].b2);

				int index = -1;
				float max = INT16_MIN;
				for(int j=0;j<n_output;j++){
					if(output[j] > max){
						max = output[j];
						index = j;
					}
				}

				if(index == 0)
					dir = LEFT;
				else if(index == 1)
					dir = RIGHT;
				else if(index == 2)
					dir = UP;
				// else if(index == 3)
				// 	dir = DOWN;

				reward = logic();

				// last_dir = dir;

				// int num = rand();
				// if(num % 20 == 0)
				// 	reward = -1;
				// else if(num % 20 == 1 || num % 20 == 2)
				// 	reward = 1;
				// else
				// 	reward = 0;

				total_reward += reward;
				steps += 1;
				// cout<<total_reward<<" "<<steps<<endl;

				// free(input);
				// free(output);

				if(reward > 0)
					total_steps += 100;

				if(steps > total_steps)
					break;
			}
			if(i == 0)
				cout<<"Generation: "<<k+1<<"\tReward: "<<total_reward<<"\tSteps: "<<steps<<endl;
			nns[i].reward = total_reward;

			if(total_reward >= max_reward){
				max_reward = total_reward;
				max_steps = steps;
				max_gen_num = k+1;
				best_index = i;
			}
		}

		// cout<<nns.size()<<endl;
		// find top nns
		sort(nns.begin(),nns.end(),comp);

		best_nn = nns[0];

		int top_nns = population_size * natural_selection_ratio;
		int reward_sum = 0;
		for(int i=0;i<top_nns;i++){
			nns_new.push_back(nns[i]);
			reward_sum += nns[i].reward;
		}

		nns.clear();
		for(int i=0;i<top_nns;i++)
			nns.push_back(nns_new[i]);

		for(int i=0;i<population_size-top_nns;i++){
			int index1 = selectParent(nns_new,reward_sum);
			neuralNetwork parent1 = nns_new[index1];
			int index2 = selectParent(nns_new,reward_sum);
			neuralNetwork parent2 = nns_new[index2];

			struct neuralNetwork child = crossover(parent1,parent2);

			mutate(child);
			
			nns.push_back(child);
		}
		// cout<<best_nn.reward<<endl;
		nns_new.clear();
	}

	cout<<"Max reward: "<<max_reward<<endl;
	cout<<"No of steps: "<<max_steps<<endl;
	cout<<"Generation num: "<<max_gen_num<<endl;

	return 0;
}
