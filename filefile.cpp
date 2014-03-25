//#include <iostream>
//#include <fstream>
//#include "opencv2\opencv.hpp"
//#include <stdio.h>
//#include <string>
//#include <vector>
//#include <math.h>
//#include <map>
//#include <cmath>
//#include <stdlib.h>
//#include <cv.h>
//#include <highgui.h>
//#include <cxcore.h>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/legacy/legacy.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/nonfree/features2d.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <list>
//#include "pyramid.h"
//#include "helpers.h"
//#include "affine.h"
//#include "siftdesc.h"
//#include "direct.h"
//#include "io.h"
//
//using namespace cv;
//using namespace std;
//
//int main(){
//    char buf_dir[] = "C:/Cassandra/here/image_index.txt";
//    std::ifstream t;
//    int t_length = 0;
//
//    t.open("C:/Cassandra/here/image_index.txt");      // open input file
//    t.seekg(0, t.end);
//    //streampos sp=t.tellg();    // go to the end
//    t_length = t.tellg();           // report location (this is the length)
//    t.seekg(0, t.beg);    // go back to the beginning
//    char t_buffer[t_length];    // allocate memory for a buffer of appropriate dimension
//    memset(t_buffer, 0, t_length);//不初始化会出问题的
//    t.read(t_buffer, t_length);       // read the whole file into the buffer
//    t.close(); //t完成了使命
//
//    ofstream out;
//    out.open(buf_dir, ios::out|ios::trunc);
//    out << 40 << endl << t_buffer << endl;
//    out.close();
//    return 0;
//}
