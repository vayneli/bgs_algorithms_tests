#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "AdaptiveSelectiveBackgroundLearning.h"

using namespace std;
using namespace cv;
using namespace bgslibrary::algorithms;

int main(int argc,char** argv){

    VideoCapture cap("../log_color_8.avi");
    if(!cap.isOpened()){
        cerr<<"cannot open video"<<endl;
        return -1;
    }

    //cout<<"1"<<endl;
    IBGS *bgs;
    bgs=new AdaptiveSelectiveBackgroundLearning;
    Mat img_input,img_output,img_bgmodel;
    while(1)
    {   
        cap>>img_input;
        if(img_input.empty())
            break;
        imshow("img_input",img_input);
        bgs->process(img_input,img_output,img_bgmodel);
        if(cvWaitKey(33)>=0)
            break;
    }

    delete bgs;

    cvDestroyAllWindows();

    return 0;
}