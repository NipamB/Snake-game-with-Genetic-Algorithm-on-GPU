#include<iostream>
#include<cstdlib>
#include<stdio.h>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>     
#include <stdlib.h>
#include<ctime>
#include<vector>
using namespace std;
   

//////////////////////////////////////////////////
char getch(){
    char buf=0;
    struct termios old={0};
    fflush(stdout);
    if(tcgetattr(0, &old)<0)
        perror("tcsetattr()");
    old.c_lflag&=~ICANON;
    old.c_lflag&=~ECHO;
    old.c_cc[VMIN]=1;
    old.c_cc[VTIME]=0;
    if(tcsetattr(0, TCSANOW, &old)<0)
        perror("tcsetattr ICANON");
    if(read(0,&buf,1)<0)
        perror("read()");
    old.c_lflag|=ICANON;
    old.c_lflag|=ECHO;
    if(tcsetattr(0, TCSADRAIN, &old)<0)
        perror ("tcsetattr ~ICANON");

    return buf;
 }

///////////////////////////////////////////////


////////////////////////////////////////////////

//keyboard hit


int kbhit(void)
{
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if(ch != EOF)
    {
      ungetc(ch, stdin);
      return 1;
    }

    return 0;
}


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


bool gameover;
const int width=40;
const int height=40;
int x,y,fruitx,fruity,score;
int tailx[100],taily[100];
int ntail;

enum edirection{STOP=0,LEFT,RIGHT,UP,DOWN};

edirection dir;

unsigned int microseconds = 10000;

vector<int> global_score;

void setup()
{
    gameover=false;
    dir = STOP;
    x = width/2;
    y = height/2;
    fruitx=rand() % width;
    fruity=rand() % height;
    score = 0;
}



void draw()
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

    cout<< FMAG("hi");
    cout<<x<<" "<<y<<" : "<<fruitx<<" "<<fruity<<" : "<<tailx[ntail-1]<<" "<<taily[ntail-1]<<endl;
 
}

void input()
{
    int num = rand();
    if(num%4 == 0)
        dir = LEFT;
    else if(num%4 == 1)
        dir = RIGHT;
    else if(num%4 == 2)
        dir = UP;
    else if(num%4 == 3)
        dir = DOWN;
    // if (kbhit())
    // {
    //     switch (getch())
    //     {
    //         case 'a': 
    //             dir = LEFT;
    //             break;
    //         case 'd': 
    //             dir = RIGHT;
    //             break;
    //         case 'w':
    //             dir = UP;
    //             break;
    //         case 's':
    //             dir = DOWN;
    //             break;
    //     }
    // }
}

void logic()
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
            x--;
            break;
        case RIGHT:
            x++;
            break;
        case UP:
            y--;
            break;
        case DOWN:
            y++;
            break;
    }

    if(x> width || x<0 || y > height || y<0)
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
        score = score +1000;
        fruitx=rand() % width;
        fruity=rand() % height;
        ntail++;
    }
}




int main()
{  
    for(int i=0;i<50;i++){
        setup();

        while(!gameover)
        {
            draw();
            input();
            sleep(0.1);
            logic();
        }
        global_score.push_back(score);
    }

    for(int i=0;i<global_score.size();i++)
        cout<<global_score[i]<<endl;

    return 0;
}




 

