#include<bits/stdc++.h>
#include<unistd.h>
using namespace std;

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

// global  variables
int play_number = 5;

int n_input = 24;
int n_hidden = 20;
int n_output = 4;

bool gameover;
int height = 30;
int width = 30;
int dir, dir_next, last_dir;
int score;
int x, y;
int fruitx, fruity;
int tailx[100], taily[100];
int ntail;
int steps;
int max_steps = 1000;

// setup to start the game
void setup(){
    // flag to exit game
    gameover=false;
    
    // next direction to be taken
    dir = 0;
    dir_next = 0;

    // head of the snake
    x = width/2;
    y = height/2;

    // location of the fruit
    fruitx=rand() % width;
    fruity=rand() % height;

    // size of the snake
    ntail = 3;

    // score of the game
    score = 0;

    // number of steps taken by the snake
    steps = 0;
}

// display the game on screen
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

// set input for forward pass
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

// forward pass to calculate the output
void forward(float *input, float *output, float*w1,
            float *w2, float *b1, float *b2){
    // forward pass for first layer
    float *layer1 = (float *)malloc(n_hidden*sizeof(float));
    for(int i=0;i<n_hidden;i++){
        layer1[i] = 0;
        for(int j=0;j<n_input;j++){
            layer1[i] += input[j]*w1[j*n_hidden+i];
        }
        layer1[i] += b1[i];

        // sigmoid activation
        layer1[i] = 1 / (1 + exp(-layer1[i]));
    }

    // forward pass for second layer and thus get the output layer
    for(int i=0;i<n_output;i++){
        output[i] = 0;
        for(int j=0;j<n_hidden;j++){
            output[i] += layer1[j]*w2[j*n_output+i];
        }
        output[i] += b2[i];

        // sigmoid activation
        output[i] = 1 / (1 + exp(-output[i]));
    }

    free(layer1);
}

// get the next direction based on the output
void get_direction(float parameters[]){
    // parameters of the neural network
    float *w1 = &parameters[0];
    float *b1 = &parameters[n_input*n_hidden];
    float *w2 = &parameters[n_input*n_hidden+n_hidden];
    float *b2 = &parameters[n_input*n_hidden+n_hidden+n_hidden*n_output];

    // input to neural network
    float *input = (float *)malloc(n_input*sizeof(float));
    set_input(input);

    // output of neural network
    float *output = (float *)malloc(n_output*sizeof(float));

    // do forward pass to get the output
    forward(input,output,w1,w2,b1,b2);

    // get the best direction based on output
    int index = -1;
    float max = INT16_MIN;
    for(int j=0;j<n_output;j++){
        if(output[j] > max){
            max = output[j];
            index = j;
        }
    }

    dir = index + 1;

    // get the second best direction based on output
    int index1 = -1;
    float max1 = INT16_MIN;
    for(int j=0;j<n_output;j++){
        if(output[j] > max1 && j != index){
            max1 = output[j];
            index1 = j;
        }
    }

    dir_next = index1 + 1;

    free(input);
    free(output);
}

// logic of the game
void logic(){
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

    // move the snake based on direction
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

    // snake hit the wall
    if(x >= width || x < 0 || y >= height || y < 0)
    {
        gameover=true;
        cout<<"GAME OVER"<<endl;
    }

    // snake hit its body
    for(int i =0; i<ntail;i++)
    {
        if(tailx[i]==x && taily[i]==y)
        {
            gameover = true;
            cout<<"GAME OVER"<<endl;
        }
    }

    // snake eat the fruit
    if(x==fruitx && y==fruity)
    {
        score = score + 1;
        fruitx=rand() % width;
        fruity=rand() % height;
        ntail++;
    }
}

int main(int argc, char **argv)
{  
    srand(time(NULL));

    // read parameter values of the best neural network from file
    ifstream fin;
    fin.open(argv[1]);

    string line;
    fin>>line;
    fin>>line;
    n_input = stoi(line);

    fin>>line;
    fin>>line;
    n_hidden = stoi(line);

    fin>>line;
    fin>>line;
    n_output = stoi(line);

    fin>>line;
    fin>>line;
    height = stoi(line);

    fin>>line;
    fin>>line;
    width = stoi(line);

    getline(fin,line);
    getline(fin,line);

    int parameter_size = n_input*n_hidden + n_hidden + n_hidden*n_output + n_output;
    float nn[parameter_size];

    for(int i=0;i<parameter_size;i++){
        fin>>line;
        size_t found = line.find('e');
        if(found != string::npos){
            nn[i] = 0.0;
        }
        else
            nn[i] = stof(line);
    }
    
    // play the game 'play_number' of times
    for(int i=0;i<play_number;i++){
        // setup to start the game
        setup();

        // loop till the game is not over
        while(!gameover && steps < max_steps)
        {
            draw();
            get_direction(nn);
            logic();
            usleep(150000);
            steps++;
        }
    }

    return 0;
}