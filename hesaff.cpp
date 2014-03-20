///*
// * Copyright (C) 2008-12 Michal Perdoch
// * All rights reserved.
// *
// * This file is part of the HessianAffine detector and is made available under
// * the terms of the BSD license (see the COPYING file).
// *
// */
//
//#include <iostream>
//#include <fstream>
//#include "opencv2\opencv.hpp"
//#include <stdio.h>
//#include <iostream>
//#include <string>
//#include <vector>
//#include <string>
//#include <iostream>
//#include <math.h>
//#include <vector>
//#include <map>
//
//#include <cv.h>
//#include <highgui.h>
//#include <cxcore.h>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/legacy/legacy.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/nonfree/features2d.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include "pyramid.h"
//#include "helpers.h"
//#include "affine.h"
//#include "siftdesc.h"
//#include "stdlib.h"
//#include "direct.h"
//#include "io.h"
//
//using namespace cv;
//using namespace std;
//
////Hessian parameters
//struct HessianAffineParams
//{
//
//   float threshold;
//   int   max_iter;
//   float desc_factor;
//   int   patch_size;
//   bool  verbose;  //??
//   HessianAffineParams()
//      {
//         threshold = 16.0f/3.0f;
//         max_iter = 16;
//         desc_factor = 3.0f*sqrt(3.0f);
//         patch_size = 41;
//         verbose = false;
//      }
//};
//
//int g_numberOfPoints = 0;
//int g_numberOfAffinePoints = 0;
//
//struct Keypoint
//{
//   float x, y, s;
//   float a11,a12,a21,a22;//这四个量是干啥的？
//   float response;
//   int type;
//   unsigned char desc[128];//128维的数值？
//};
//
//struct AffineHessianDetector : public HessianDetector, AffineShape, HessianKeypointCallback, AffineShapeCallback
//{
//   const Mat image;
//   SIFTDescriptor sift;
//   vector<Keypoint> keys;//关键点
//   //int key_count;//关键点的个数，要求从外面可以读取
//
//public:
//   AffineHessianDetector(const Mat &image, const PyramidParams &par, const AffineShapeParams &ap, const SIFTDescriptorParams &sp) :
//      HessianDetector(par),
//      AffineShape(ap),
//      image(image),
//      sift(sp)
//      {
//         this->setHessianKeypointCallback(this);
//         this->setAffineShapeCallback(this);
//      }
//      // SIFT?
//   void onHessianKeypointDetected(const Mat &blur, float x, float y, float s, float pixelDistance, int type, float response)
//      {
//         g_numberOfPoints++;
//         findAffineShape(blur, x, y, s, pixelDistance, type, response);
//      }
//
//   void onAffineShapeFound(
//      const Mat &blur, float x, float y, float s, float pixelDistance,
//      float a11, float a12,
//      float a21, float a22,
//      int type, float response, int iters)
//      {
//         // convert shape into a up is up frame
//         rectifyAffineTransformationUpIsUp(a11, a12, a21, a22);
//
//         // now sample the patch
//         if (!normalizeAffine(image, x, y, s, a11, a12, a21, a22))
//         {
//            // compute SIFT
//            sift.computeSiftDescriptor(this->patch);
//            // store the keypoint
//            keys.push_back(Keypoint());
//            Keypoint &k = keys.back();
//            k.x = x; k.y = y; k.s = s; k.a11 = a11; k.a12 = a12; k.a21 = a21; k.a22 = a22; k.response = response; k.type = type;
//            for (int i=0; i<128; i++)
//               k.desc[i] = (unsigned char)sift.vec[i];
//            // debugging stuff
//            if (0)
//            {
//               cout << "x: " << x << ", y: " << y
//                    << ", s: " << s << ", pd: " << pixelDistance
//                    << ", a11: " << a11 << ", a12: " << a12 << ", a21: " << a21 << ", a22: " << a22
//                    << ", t: " << type << ", r: " << response << endl;
//               for (size_t i=0; i<sift.vec.size(); i++)
//                  cout << " " << sift.vec[i];
//               cout << endl;
//            }
//            g_numberOfAffinePoints++;
//         }
//      }
//
////楼下这个函数就是特征点的文件输出
//   void exportKeypoints(ostream &out)
//      {
//         out << 128 << endl;// 128维SIFT
//         out << keys.size() << endl;//检测到的特征点的个数
//
//         //接下来，对于每个特征点
//         for (size_t i=0; i<keys.size(); i++)
//         {
//             //注意Keypoint结构
//            Keypoint &k = keys[i];
//
//            float sc = AffineShape::par.mrSize * k.s;
//            Mat A = (Mat_<float>(2,2) << k.a11, k.a12, k.a21, k.a22);//这是啥矩阵？
//            SVD svd(A, SVD::FULL_UV);//SVD都上了啊
//
//
//            float *d = (float *)svd.w.data;
//            d[0] = 1.0f/(d[0]*d[0]*sc*sc);
//            d[1] = 1.0f/(d[1]*d[1]*sc*sc);
//
//            A = svd.u * Mat::diag(svd.w) * svd.u.t();
//            //文件中第二行开始前5个数值： k.x k.y A(0,0) A(0,1) A(1,1)
//            out << k.x << " " << k.y << " " << A.at<float>(0,0) << " " << A.at<float>(0,1) << " " << A.at<float>(1,1);
//            //后面才是128维的数值
//            for (size_t i=0; i<128; i++)
//               out << " " << int(k.desc[i]);
//            out << endl;
//         }
//      }
//
//
//
//};
//
//
//class CBrowseDir
//{
//    //vector<Keypoint> keys_in_image;
//
//protected:
//    //存放初始目录的绝对路径，以'\'结尾
//    char m_szInitDir[_MAX_PATH];
//
//public:
//    //缺省构造器
//    CBrowseDir();//Keypoint &k = keys[i];
//
//    //设置初始目录为dir，如果返回false，表示目录不可用
//    bool SetInitDir(const char *dir);
//
//    //开始遍历初始目录及其子目录下由filespec指定类型的文件
//    //filespec可以使用通配符 * ?，不能包含路径。
//    //如果返回false，表示遍历过程被用户中止
//    bool BeginBrowse(const char *filespec);
//
//protected:
//    //遍历目录dir下由filespec指定的文件
//    //对于子目录,采用迭代的方法
//    //如果返回false,表示中止遍历文件
//    bool BrowseDir(const char *dir,const char *filespec);
//
//    //函数BrowseDir每找到一个文件,就调用ProcessFile
//    //并把文件名作为参数传递过去
//    //如果返回false,表示中止遍历文件
//    //用户可以覆写该函数,加入自己的处理代码
//    virtual bool ProcessFile(const char *filename);
//
//    //函数BrowseDir每进入一个目录,就调用ProcessDir
//    //并把正在处理的目录名及上一级目录名作为参数传递过去
//    //如果正在处理的是初始目录,则parentdir=NULL
//    //用户可以覆写该函数,加入自己的处理代码
//    //比如用户可以在这里统计子目录的个数
//    virtual void ProcessDir(const char *currentdir,const char *parentdir);
//};
//
//CBrowseDir::CBrowseDir()
//{
//    //用当前目录初始化m_szInitDir
//    getcwd(m_szInitDir,_MAX_PATH);
//
//    //如果目录的最后一个字母不是'\',则在最后加上一个'\'
//    int len=strlen(m_szInitDir);
//    if (m_szInitDir[len-1] != '\\')
//        strcat(m_szInitDir,"\\");
//}
//
//bool CBrowseDir::SetInitDir(const char *dir)
//{
//    //先把dir转换为绝对路径
//    if (_fullpath(m_szInitDir,dir,_MAX_PATH) == NULL)
//        return false;
//
//    //判断目录是否存在
//    if (_chdir(m_szInitDir) != 0)
//        return false;
//
//    //如果目录的最后一个字母不是'\',则在最后加上一个'\'
//    int len=strlen(m_szInitDir);
//    if (m_szInitDir[len-1] != '\\')
//        strcat(m_szInitDir,"\\");
//
//    return true;
//}
//
//bool CBrowseDir::BeginBrowse(const char *filespec)
//{
//    ProcessDir(m_szInitDir,NULL);
//    return BrowseDir(m_szInitDir,filespec);
//}
//
//
////能再来一个变量不
//bool CBrowseDir::BrowseDir(const char *dir,const char *filespec)
//{
//    _chdir(dir);
//
//    //首先查找dir中符合要求的文件, in io.h
//    long hFile;
//    _finddata_t fileinfo;// what?
//    if ((hFile=_findfirst(filespec,&fileinfo)) != -1)
//    {
//        //我们需要一个计数器。用j好了。
//        //int j = 0;
//        do
//        {
//            //检查是不是目录
//            //如果不是,则进行处理
//            if (!(fileinfo.attrib & _A_SUBDIR))
//            {
//                char filename[_MAX_PATH];// length: _MAX_PATH
//                strcpy(filename,dir);
//                strcat(filename,fileinfo.name);// what is fileinfo.name?
//
//
//                //here we GOOOOOOOOOOOOOOOOOO!
//
//                Mat tmp = imread(filename);// read it into tem
//                Mat image(tmp.rows, tmp.cols, CV_32FC1, Scalar(0)); // create a image?
//
//                float *out = image.ptr<float>(0);// pointer, pointing at the output image
//                unsigned char *in  = tmp.ptr<unsigned char>(0); // char??
//
//                for (size_t i=tmp.rows*tmp.cols; i > 0; i--)
//                {
//                    *out = (float(in[0]) + in[1] + in[2])/3.0f;  //  averaging 3 channels.
//                    out++; // move the pointer, only one channel.
//                    in+=3; // move the pointer, 3 channels.
//                }
//
//                HessianAffineParams par; // kind of pre-defined
//                double t1 = 0;
//                {
//                    // copy params
//                    PyramidParams p; // struct
//                    p.threshold = par.threshold;
//
//                    AffineShapeParams ap; // struct
//                    ap.maxIterations = par.max_iter;
//                    ap.patchSize = par.patch_size;
//                    ap.mrSize = par.desc_factor;
//
//                    SIFTDescriptorParams sp;  //SIFT?
//                    sp.patchSize = par.patch_size;
//
//                    AffineHessianDetector detector(image, p, ap, sp);//action?
//                    t1 = getTime(); //?
//                    g_numberOfPoints = 0;
//                    detector.detectPyramidKeypoints(image);
//
//                     //在这里改变一下计数值
//                    //detector.key_count = g_numberOfPoints;
//
//                    //因为detector 本身是 AffineHessianDetector的一个例子，而AffineHessianDetector里面有对g_numberOfPoints进行变化
//
//
//                    cout << "Detected " << g_numberOfPoints << " keypoints and " << g_numberOfAffinePoints << " affine shapes in " << getTime()-t1 << " sec." << endl;
//
//                    // write the file
//                    char suffix[] = ".hesaff.txt";//后缀名
//                    int len = strlen(filename)+strlen(suffix)+1;
//                    char buf_go[len];
//                    snprintf(buf_go, len, "%s%s", filename, suffix); buf_go[len-1]=0;//
//                    ofstream out(buf_go);// Here comes the output. "buf" is the outgoing stream with length of "len"
//                    detector.exportKeypoints(out); // using the function "exportKeypoints". Target: "out";  Source: "detector"
//
//                    //写到这儿，还是在这张图片里
//                    //for (int i = 0; i <= detector.key_count; ++i){
//                    //Keypoint &k = detector.keys[j];
//                    //}
//                }
//
//                //
//                cout << filename << endl;// here, print the filename
//                if (!ProcessFile(filename))
//                    return false;
//            }
//        } while (_findnext(hFile,&fileinfo) == 0);
//        _findclose(hFile);
//    }
//
//    //查找dir中的子目录
//    //因为在处理dir中的文件时，派生类的ProcessFile有可能改变了
//    //当前目录，因此还要重新设置当前目录为dir。
//    //执行过_findfirst后，可能系统记录下了相关信息，因此改变目录
//    //对_findnext（类型）没有影响。
//
//    _chdir(dir);//what's this? dir?
//    if ((hFile=_findfirst("*.*",&fileinfo)) != -1)
//    {
//        do
//        {
//            //检查是不是目录
//            //如果是,再检查是不是 . 或 ..
//            //如果不是,进行迭代
//            //iteration
//
//            if ((fileinfo.attrib & _A_SUBDIR))
//            {
//                if (strcmp(fileinfo.name,".") != 0 && strcmp
//                    (fileinfo.name,"..") != 0)
//                {
//                    char subdir[_MAX_PATH];
//                    strcpy(subdir,dir);
//                    strcat(subdir,fileinfo.name);
//                    strcat(subdir,"\\");
//                    ProcessDir(subdir,dir);// did nothing
//                    //end of iteration?
//                    if (!BrowseDir(subdir,filespec))
//                        return false;
//                }
//            }
//        } while (_findnext(hFile,&fileinfo) == 0);
//        _findclose(hFile);
//    }
//    return true;
//}
//
//bool CBrowseDir::ProcessFile(const char *filename)
//{
//    return true;
//}
//
//// do nothing
//void CBrowseDir::ProcessDir(const char *currentdir,const char *parentdir)
//{
//}
//
////从CBrowseDir派生出的子类，用来统计目录中的文件及子目录个数
//class CStatDir:public CBrowseDir
//{
//protected:
//    int m_nFileCount;   //保存文件个数
//    int m_nSubdirCount; //保存子目录个数
//
//public:
//    //缺省构造器
//    CStatDir()
//    {
//        //初始化数据成员m_nFileCount和m_nSubdirCount
//        m_nFileCount=m_nSubdirCount=0;
//    }
//
//    //返回文件个数
//    int GetFileCount()
//    {
//        return m_nFileCount;
//    }
//
//    //返回子目录个数
//    int GetSubdirCount()
//    {
//        //因为进入初始目录时，也会调用函数ProcessDir，
//        //所以减1后才是真正的子目录个数。
//        return m_nSubdirCount-1;
//    }
//
//protected:
//    //覆写虚函数ProcessFile，每调用一次，文件个数加1
//    virtual bool ProcessFile(const char *filename)
//    {
//        m_nFileCount++;
//        return CBrowseDir::ProcessFile(filename);
//    }
//
//    //覆写虚函数ProcessDir，每调用一次，子目录个数加1
//    virtual void ProcessDir
//        (const char *currentdir,const char *parentdir)
//    {
//        m_nSubdirCount++;
//        CBrowseDir::ProcessDir(currentdir,parentdir);
//    }
//};
//
////
//int main()
//{
//
//    //获取目录名
//    char buf[256];
//    printf("input the document dir:");
//    gets(buf);
//
//    //构造类对象
//    //important
//    CStatDir statdir;
//
//    //设置要遍历的目录
//    if (!statdir.SetInitDir(buf))
//    {
//        puts("Dir does not exist.");
//        return -1;
//    }
//
//    //开始遍历
//    statdir.BeginBrowse("*.jpg*");
//    printf("Number of images: %d\nNumber of sub_dir:%d\n",statdir.GetFileCount(),statdir.GetSubdirCount());
//
//    //here
//    //后面的工作，1，确定有多少个keypoint（num_of_keys）。这需要读取各个保存下来的特征文件。
//    //2. 弄一个足够大的矩阵
//    //3. Kmean
//}
//
