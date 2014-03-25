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
//typedef vector<int> vec;
//struct Image_statistic
//{
//    int image_code; //图片的编号
//    int des_count;  //descriptor的个数
//    int K;  //一共多少个词汇
//    Mat the_vocabulary;  //一个mat，记录每个descriptor所属的词汇
//};
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
//    void exportKeypoints(ostream &out, ostream &out2)
//    {
//         out << 128 << endl;// 128维SIFT
//         out << keys.size() << endl;//检测到的特征点的个数
//         out2 << keys.size() << endl;//我加的部分，给index文件也来一个。
//
//         //接下来，对于每个特征点
//         for (size_t i=0; i<keys.size(); ++i)
//         {
//             //注意Keypoint结构
//            Keypoint &k = keys[i];
//            float sc = AffineShape::par.mrSize * k.s;
//            Mat A = (Mat_<float>(2,2) << k.a11, k.a12, k.a21, k.a22);//这是啥矩阵？
//            SVD svd(A, SVD::FULL_UV);//SVD都上了啊
//            float *d = (float *)svd.w.data;
//            d[0] = 1.0f/(d[0]*d[0]*sc*sc);
//            d[1] = 1.0f/(d[1]*d[1]*sc*sc);
//
//            A = svd.u * Mat::diag(svd.w) * svd.u.t();
//            //文件中第二行开始前5个数值： k.x k.y A(0,0) A(0,1) A(1,1)
//            out << k.x << " " << k.y << " " << A.at<float>(0,0) << " " << A.at<float>(0,1) << " " << A.at<float>(1,1);
//            //后面才是128维的数值
//            for (size_t j=0; j<128; ++j)
//               out << " " << int(k.desc[j]);
//            out << endl;
//         }
//    }
//    void outputKeypoints(Mat &Key_mat)
//    {
//        Mat tmp = Mat::zeros(keys.size(),128,CV_32S);
//        for (size_t i=0; i<keys.size(); ++i)
//         {
//            Keypoint &k2 = keys[i];
//            for (size_t j=0; j<128; ++j){
//               tmp.at<int>(i,j) = int(k2.desc[j]);
//               }
//         }
//        tmp.copyTo(Key_mat);
//    }
//};
////这里我们来写我们的归一化函数
//    void mat_normalization(Mat &src, Mat &dst)
//    {
//        cout << "K = "<<src.cols << endl; // K
//        cout << "image_count = " << src.rows << endl;// image_count
//        for(int i=0; i < src.rows; ++i){
//            double value_sum = 0;
//            for(int j=0; j < src.cols; ++j){//K
//                value_sum = value_sum + src.at<double>(i,j)*src.at<double>(i,j);
//            }
//            value_sum = sqrt(value_sum);
//            //这个地方都出现负值了。这肯定是有问题的。建议改成double如何？
//            cout << "value_sum[" << i << "]: " <<value_sum <<endl;
//            for(int j=0; j < src.cols;++j){
//                    if(src.at<double>(i,j)==0){
//                        dst.at<double>(i,j) = 0;
//                    }else{
//                        dst.at<double>(i,j) = src.at<double>(i,j)/value_sum;}
//            }
//        }
//    }
//    //距离计算，对位进行比对
//    float calculate_dis(Mat &mat_A, Mat &mat_B)
//    {
//        if((mat_A.rows!=1)||(mat_B.rows!=1)){
//            cout << "Errow: one of the Mat's row number does not equal to 1." << endl;
//            return -1;}
//        if(mat_A.cols!=mat_B.cols){
//            cout << "Error: two Mats should have equal col number." << endl;
//            return -2;}
//        float Mat_distance = 0;
//        for(int i=0; i< mat_A.cols; ++i)
//        {
////            cout << mat_A.rows << " " << mat_A.cols << endl;
////            cout << mat_A << endl;
////            cout << mat_A.at<float>(0,0) << endl;
////            cout << mat_B << endl;
////            cout << mat_B.at<float>(0,i) << endl;
//            float Cal_A = mat_A.at<float>(0,i)/100;
//            float Cal_B = mat_B.at<float>(0,i)/100;
//            float Cal_C = abs(Cal_A - Cal_B);
//            Mat_distance = Mat_distance + pow(Cal_C,2);
//        }
//        float Mat_distance_233 = sqrt(sqrt(Mat_distance));
//        //没有办法，太大了，只能进行两次开方
//        return Mat_distance_233;
//    }
//
//////新图片的descriptor与原cluster center之间进行比对的函数。
//    void find_cluster_center(Mat &descriptor_mat, Mat &cluster_mat, Mat &cluster_output, Mat &cluster_all)
//    {
//        //先统计一共多少个descriptor
//        if(descriptor_mat.cols != 128){
//            return;
//        }
//        if(cluster_mat.cols != 128){
//            return;
//        }
//        //
//        Mat cluster_temp = Mat::zeros(1,cluster_mat.rows,CV_32S);
//        Mat cluster_all_temp = Mat::zeros(descriptor_mat.rows, cluster_mat.rows, CV_32F);
//        cout << descriptor_mat.rows << " " << cluster_mat.rows <<endl;
//        Mat dump_A;
//        Mat dump_B;
////        Mat dump_C;
//        for(int i=0; i < descriptor_mat.rows; ++i)
//        {
//            float dump = 0;
//            float dump_cal = 0;
//            int cluster_marker = 0;
//            for(int j =0; j < cluster_mat.rows; ++j)
//            {
//                descriptor_mat.row(i).copyTo(dump_A);
////                cout << dump_A << endl;
//                cluster_mat.row(j).copyTo(dump_B);
//                // 问题是，这两个并不都是float啊。cluster那个是，所以要对descriptor进行强制的类型转换。
//                dump_A.convertTo(dump_A, CV_32F);
////                cout << dump_A.rows << " " << dump_A.cols << endl;
////                cout << dump_B.rows << " " << dump_B.cols << endl;
//                //cout << dump_B << endl;
//                dump_cal = calculate_dis(dump_A, dump_B);
//                //cout << dump_cal << endl;
//                cluster_all_temp.at<float>(i,j) = dump_cal;
//                //我们要记录的是最小值。千万小心。
//                if(j==0){
//                    dump = dump_cal;
//                }
//                if(dump > dump_cal){
//                    dump = dump_cal;
//                    cluster_marker = j;
//                }
//                dump_cal = 0;
//                dump_A.release();
//                dump_B.release();
////                dump_C.release();
//            }
//            cluster_temp.at<int>(0,cluster_marker) = cluster_temp.at<int>(0,cluster_marker) + 1;
//        }
//        cluster_temp.copyTo(cluster_output);
//        cluster_all_temp.copyTo(cluster_all);
//    }
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
//    // change
//    char suffix_dir[] = "/image_index.txt";//后缀名
//    int len_dir = strlen(dir)+strlen(suffix_dir)+1;
//    char buf_dir[len_dir];
//    snprintf(buf_dir, len_dir, "%s%s", dir, suffix_dir); buf_dir[len_dir-1]=0;//
//
//    cout << dir << endl;
//
//    if (!access(buf_dir,0)){
//    ofstream out(buf_dir, ios::trunc);//干掉原先的文件
//    out.close();
//    cout<<"file " << buf_dir << " exist."<<endl;}else{
//    ofstream out(buf_dir);//踹门！写文件啦！
//    out.close();
//    cout<<"file " << buf_dir << " does not exist."<<endl;}
//    //成功了。能接着我之前的文件从后面写。这样我们可以写循环了。
//    //fstream foi(buf_go, ios::in | ios::out | ios::app);
//    //foi << filename << endl;
//
//    // change
//
//    //首先查找dir中符合要求的文件, in io.h
//    long hFile;
//    _finddata_t fileinfo;// what?
//    int count_lalala = 0;
//    // filespec? jpg后缀
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
//                //here we GOOOOOOOOOOOOOOOOOO!
//
//                Mat tmp = imread(filename,0);// read it into tem
//                Mat image; // create a image?
//                tmp.convertTo(image, CV_32F);
//                //
//                char filename02[_MAX_PATH];
//                memset(filename02, 0, _MAX_PATH);
//                char filename03[_MAX_PATH];
//                memset(filename03, 0, _MAX_PATH);
//                count_lalala += 1;
//                strcpy(filename02,"C:/Cassandra/hereafter/grey_image");
//                itoa(count_lalala,filename03,10);
//                strcat(filename02,filename03);
//                strcat(filename02,".jpg");
//                imwrite(filename02,image);
//                //
//                //我觉得这里读图片可能是有问题的
//
//                // let us resize the image.
////                float *out = image.ptr<float>(0);// pointer, pointing at the output image
////                unsigned char *in  = tmp.ptr<unsigned char>(0); // char??
////
////                for (size_t i=tmp.rows*tmp.cols; i > 0; i--)
////                {
////                    *out = (float(in[0] + in[1] + in[2]))/3.0f;  //  averaging 3 channels.
////                    out++; // move the pointer, only one channel.
////                    in+=3; // move the pointer, 3 channels.
////                }
////                // resize。原图片太大了。
//                Mat dst;
//                cv::resize(image, dst, Size(), 0.25, 0.25, INTER_CUBIC);
//                HessianAffineParams par; // kind of pre-defined
//                double t1 = 0;//这个是计时用的
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
//                    AffineHessianDetector detector(dst, p, ap, sp);//action?这一步是进行计算的。
//                    t1 = getTime(); //?
//                    g_numberOfPoints = 0;
//                    detector.detectPyramidKeypoints(dst);
//
//                    //在这里改变一下计数值
//                    //detector.key_count = g_numberOfPoints;
//
//                    //因为detector 本身是 AffineHessianDetector的一个例子，而AffineHessianDetector里面有对g_numberOfPoints进行变化
//                    cout << "Detected " << g_numberOfPoints << " keypoints and " << g_numberOfAffinePoints << " affine shapes in " << getTime()-t1 << " sec." << endl;
//                    // write the file
//                    char suffix[] = ".hesaff.txt";//后缀名
//                    // change
//                    char filename_short[_MAX_PATH];
//                    memset(filename_short, 0, _MAX_PATH);//还是初始化一下比较安全啊
//                    //memset(filename_short,);
//                    memcpy(filename_short, filename, sizeof(char)*(strlen(filename)-4));//把“.jpg”4个字节去掉
//
//                    //printf("short filename %s\n", filename_short);
//                    //printf("filename %s\n", filename);
//                    // change
//                    int len = strlen(filename_short)+strlen(suffix)+1;
//                    char buf_go[len];
//                    //memset(buf_go,len,0);
//                    snprintf(buf_go, len, "%s%s", filename_short, suffix); buf_go[len-1]=0;//
//                    if (!access(buf_go,0)){
//                    ofstream out1(buf_go, ios::trunc);//干掉原先的文件
//                    out1.close();
//                    //cout<<"file " << suffix << " exist."<<endl;
//                    }else{
//                    ofstream out1(buf_go);//踹门！写文件啦！
//                    out1.close();
//                    //cout<<"file " << suffix << " does not exist."<<endl;
//                    }
//                    //
//                    //这里有个问题。没有考虑到，如果原来这个文件在的话怎么办呢？所以做一点儿小改动：ios::trunc
//                    ofstream out3(buf_go);// Here comes the output. "buf" is the outgoing stream with length of "len". "out" is a output stream.
//                    ofstream out2(buf_dir, ios::out | ios::app);
//                    out2 << filename << " ";
//                    detector.exportKeypoints(out3, out2); // using the function "exportKeypoints". Target: "out";  Source: "detector"
//                    out2.close();
//                    out3.close();
//                    //写到这儿，还是在这张图片里
//                    //for (int i = 0; i <= detector.key_count; ++i){
//                    //Keypoint &k = detector.keys[j];
//                    //}
//                }
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
//    //vector<AffineHessianDetector>;
//    //获取目录名
////    char buf[256];
////    printf("input the document dir:");
////    gets(buf);
//
//    char *buf = "C:/Cassandra/here";
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
//
//    char suffix_dir[] = "/image_index.txt";//
//    int len_dir = strlen(buf)+strlen(suffix_dir)+1;
//    char buf_dir[len_dir];
//    snprintf(buf_dir, len_dir, "%s%s", buf, suffix_dir); buf_dir[len_dir-1]=0;//
//
//    //下面的部分我们把图片总数给加到刚才的文件里
//    std::ifstream t;
//    int t_length;
//    t.open(buf_dir);      // open input file
//    t.seekg(0, std::ios::end);    // go to the end
//    t_length = t.tellg();           // report location (this is the length)
//    t.seekg(0, std::ios::beg);    // go back to the beginning
//    char t_buffer[t_length];    // allocate memory for a buffer of appropriate dimension
//    memset(t_buffer, 0, t_length);//不初始化会出问题的
//    t.read(t_buffer, t_length);       // read the whole file into the buffer
//    t.close(); //t完成了使命
//
//    ofstream out;
//    out.open(buf_dir, ios::out|ios::trunc);
//    out << statdir.GetFileCount() << endl << t_buffer << endl;
//    out.close();
//    //然后我们可以开始弄kmeans了。 另外刚才那步其实略多余。
//
//    ifstream in_index;
//    ifstream in_dir;
//    in_index.open(buf_dir, ios::in);
//    char dir_buffer[_MAX_PATH];
//    memset(dir_buffer, 0, _MAX_PATH);
//    int image_count = 0;//图片总数
//    int des_count = 0;
//    int des_count_all = 0;
//    in_index >> image_count;
//
//    vector<Mat> Descriptor_company;
//
//    //从这个循环开始，我们要一个一个的对付图片。读取他们的descriptor，完成矩阵拼接。
//    //我们是不是得弄个足够大的Mat容器？
//
//    int K = 32; // 8类
//    Mat bestLabels, centers, clustered;
//    Mat des_for_each = Mat::zeros(image_count, 1,CV_32S);//这个用来保存每张图片里的descriptor的数量，后面有用
//    int l;
//    for(l=0; l< image_count; ++l){
//        //下面这两句话是在index文件里的
//        in_index >> dir_buffer;//这里错了。这可是那个图片的地址。
//        in_index >> des_count;
//        des_count_all = des_count_all + des_count;
//        des_for_each.at<float>(l,0) = des_count;//这里做个小记录
//
//        //从这里开始我们进入hesaff文件了，但是地址还不对啊
//        char dir_front[_MAX_PATH];
//        memset(dir_front, 0, _MAX_PATH);//还是初始化一下比较安全啊
//        memcpy(dir_front, dir_buffer, sizeof(char)*(strlen(dir_buffer)-4));//把“.jpg”4个字节去掉
//        char suffix[] = ".hesaff.txt";
//        int len = strlen(dir_front)+strlen(suffix)+1;
//        char buf_hesaff[len];
//        //memset(buf_go,len,0);
//        snprintf(buf_hesaff, len, "%s%s", dir_front, suffix); buf_hesaff[len-1]=0;//
//        //
//        in_dir.open(buf_hesaff, ios::in);
//
//        // mat merge
//        Mat p = Mat::zeros(des_count, 128, CV_32F);// for kmeans
//        //
//        int whatever = 0;
//        //少读了两个
//        in_dir >> whatever;
//        in_dir >> whatever;
//        float other_para[des_count][5];
//        int SIFT_part[des_count][128];
//
//
//        for(int i = 0; i < des_count; ++i){
//        //提取前5个分量
//            for(int j = 0; j < 5; ++j){
//                in_dir >> other_para[i][j];
//                //cout << other_para[i][j] << " " ;
//            }
//            //cout<<endl;
//
//        //SIFT的128个分量
//            for(int k = 0; k < 128; ++k){
//                in_dir >> SIFT_part[i][k];
//                p.at<float>(i,k) = SIFT_part[i][k];
//                //cout << SIFT_part[i][k] << " " ;
//            }
//            //cout<< "\n" <<endl;
//        }
//        if(l == 0){
//            Descriptor_company.push_back(p);
//        }else{
//            Mat miao;//中间？miao是拼接的结果
//            vconcat(Descriptor_company[0], p, miao);
//            //这里少了一步。我们应该干掉原来的Descriptor_company[0]才对啊。
//            Descriptor_company.pop_back();
//            Descriptor_company.push_back(miao);
//        }
//
//    }
//    //Kmeans
//    cv::kmeans(Descriptor_company[0], K, bestLabels,
//            TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10000, 1.0),
//            3, KMEANS_RANDOM_CENTERS, centers);
//    for(int i=0; i<des_count_all; ++i) {
////      clustered.at<float>(i/src.cols, i%src.cols) = (float)(colors[bestLabels.at<int>(0,i)]);//上色
////      cout << bestLabels.at<int>(0,i) << " " <<
////              colors[bestLabels.at<int>(0,i)] << " " <<
////              clustered.at<float>(i/src.cols, i%src.cols) << " " <<
////              endl;
//        //cout << bestLabels.at<int>(0,i) << " ";
//    }
//    //cout << endl;
//
//    //小插曲。这里我们来统计一下每一幅图片里各个词汇的个数。那么我们先做一个mat
//    vector<Mat> count_the_des;
//
//    Mat num_by_des = Mat::zeros(image_count,K,CV_32S); // 这个用来判断每张图片里有没有某一个descriptor
//    Mat num_by_des_sum = Mat::zeros(1,K,CV_32S);//每个类别里总共的descriptor数？
//    Mat des_max_frequence = Mat::zeros(1,K,CV_32S); //这个Mat用来记录每张图片里出现频率最高的词汇的词汇频率
//
//    int ii = 0;//ii是descriptor的序号标志
//
//    for(int i=0; i < image_count;++i){
//        //Frozen是个中间过渡用的Mat
//        Mat VW_temp= Mat::zeros(1,K,CV_32S);
//        int VW_count_temp = int(des_for_each.at<float>(i,0));//这一步已经对了。能正确地读出一张图片里的descriptor数。
//        //cout << "VW_count_temp = " << VW_count_temp <<endl;
//        for(int j=0; j < VW_count_temp; ++j){
//            //p.at<float>(bestLabels.at<float>(ii,1),1) = int(p.at<float>(bestLabels.at<float>(ii,1),1)) + 1;
//            //count_the_des.push_back(p);
//            //cout << bestLabels.at<int>(j,0) << endl;
//            //cout << bestLabels.at<int>(0,ii) << " ";
//            VW_temp.at<int>(0,(bestLabels.at<int>(0,ii))) = VW_temp.at<int>(0,(bestLabels.at<int>(0,ii))) +1;
//            ++ii;
//        }
//        for(int j=0; j < K; ++j){
//            //cout <<endl << VW_temp.at<int>(0,j) << " ";
//        }
//        count_the_des.push_back(VW_temp);
//        //cout << endl;
//    }
//    char suffix_tf_idf_dir[] = "/tf_idf.txt";//
//    int len_tf_idf_dir = strlen(buf)+strlen(suffix_tf_idf_dir)+1;
//    char buf_dir2[len_tf_idf_dir];
//    snprintf(buf_dir2, len_tf_idf_dir, "%s%s", buf, suffix_tf_idf_dir); buf_dir2[len_tf_idf_dir-1]=0;//
//    std::ofstream tf_idf_out;
//    tf_idf_out.open(buf_dir2,ios::trunc);
//
//    tf_idf_out << "Step1: " << endl;
//    tf_idf_out << image_count << " " << des_count_all << " " << K << endl;
//    for(int i=0; i < image_count; ++i){
//        des_max_frequence.at<int>(0,i) = 0;
//        for(int j=0; j < K; ++j){
//            tf_idf_out << count_the_des[i].at<int>(0,j) << " ";
//            if(count_the_des[i].at<int>(0,j) > des_max_frequence.at<int>(0,i)){
//                    des_max_frequence.at<int>(0,i) = count_the_des[i].at<int>(0,j);
//            }
//        }
//        tf_idf_out << endl;
//    }
//
//    tf_idf_out << "Step2: 每个类别是否出现过" << endl;
//    for(int j=0; j < image_count ; ++j){
//        for(int i=0; i < K; ++i){
//            //注意这里的顺序，很容易错的。
//            if(count_the_des[j].at<int>(0,i)!=0){
//                num_by_des.at<int>(j,i) = 1;
//                //num_by_des_sum记录了每个类别都有多少张图片
//                num_by_des_sum.at<int>(0,i) = num_by_des_sum.at<int>(0,i) + 1;
//            }
//            tf_idf_out << num_by_des.at<int>(j,i) << " ";
//        }
//        tf_idf_out << endl;
//    }
//
//    tf_idf_out << "Step3: 每张图片中的最大频度" << endl;
//    for(int i=0; i < image_count; ++i){
//        tf_idf_out << des_max_frequence.at<int>(0,i) << " ";
//    }
//    tf_idf_out << endl;
//    // tf-idf score!
//    // First, tf
//    //建议这里全面64F化比较好
//    Mat des_tf = Mat::zeros(image_count,K,CV_64F);
//    Mat des_idf = Mat::zeros(1,K,CV_64F);
//    Mat des_tf_idf = Mat::zeros(image_count,K,CV_64F);
//
//    tf_idf_out << "Step4: tf" << endl;
//    for(int j=0; j < image_count; ++j){
//        for(int i=0; i < K; ++i){
//            des_tf.at<double>(j,i) = 0.5 + 0.5 * (double(count_the_des[j].at<int>(0,i))/double(des_max_frequence.at<int>(0,j)));
//            if(j==0){
//            des_idf.at<double>(0,i) = log10((2.0 + double(image_count))/(1.0 + double(num_by_des_sum.at<int>(0,i))));}
////            if(count_the_des[j].at<int>(0,i)==des_max_frequence.at<int>(0,j)){
//////                cout << float(count_the_des[j].at<int>(0,i) / des_max_frequence.at<int>(0,j)) << endl;
//////                cout << 0.5 * (count_the_des[j].at<int>(0,i) / des_max_frequence.at<int>(0,j)) << endl;
//////                cout << "count_the_des["<< j <<"].at<int>(0,"<< i <<"): " << count_the_des[j].at<int>(0,i) << endl;
//////                cout << "des_max_frequence.at<int>(0,"<<j<<"): " << des_max_frequence.at<int>(0,j) << endl;
//////                tf_idf_out << "count_the_des["<< j <<"].at<int>(0,"<< i <<"): " << count_the_des[j].at<int>(0,i) << endl;
//////                tf_idf_out << "des_max_frequence.at<int>(0,"<<j<<"): " << des_max_frequence.at<int>(0,j) << endl;
//////                tf_idf_out << "des_tf.at<double>("<<j<<","<<i<<"): " << des_tf.at<double>(j,i) <<endl;
////            }
//            des_tf_idf.at<double>(j,i) = des_tf.at<double>(j,i) * des_idf.at<double>(0,i);// i和j又搞错了哦亲
//            tf_idf_out << des_tf.at<double>(j,i) << " ";
//        }
//        tf_idf_out << endl;
//    }
//    tf_idf_out << "Step5: idf" << endl;
////    for(int i=0; i < K; ++i){
////        tf_idf_out << des_idf.at<double>(0,i) << endl;
////    }
//    tf_idf_out << num_by_des_sum << endl;
//    tf_idf_out << des_idf << endl;
//    tf_idf_out << "Step6: tf_idf" << endl;
//    for(int j=0; j < image_count; ++j){
//        for(int i=0; i < K; ++i){
//            tf_idf_out << des_tf_idf.at<double>(j,i) << " ";
//        }
//        tf_idf_out << endl;
//    }
//    //归一化
//    Mat tf_idf_normalized = des_tf_idf.clone();
//    mat_normalization(des_tf_idf,tf_idf_normalized);
////    tf_idf_out << "tf_idf_normalized:" << endl;
////    tf_idf_out << tf_idf_normalized<< endl;
//
//    vector<Mat> all_des;
//    for(int i=0; i < image_count; ++i){
//        if(i == 0){
//            all_des.push_back(count_the_des[i]);
//        }else{
//            Mat miao;//中间？miao是拼接的结果
//            vconcat(all_des[0], count_the_des[i], miao);
//            //这里少了一步。我们应该干掉原来的Descriptor_company[0]才对啊。
//            all_des.pop_back();
//            all_des.push_back(miao);
//        }
//    }
//    Mat tf_idf_finished;
//    Mat old_VW;
//    all_des[0].convertTo(old_VW,CV_64F);
//    all_des[0].convertTo(tf_idf_finished,CV_64F);
//    all_des.pop_back();
//    //这里出错了。两个矩阵的数据类型不同。
//    //tf_idf_finished : CV_64F
//    //tf_idf_normalized :
//    tf_idf_finished = tf_idf_finished.mul(tf_idf_normalized);
//    tf_idf_out << "tf_idf_normalized:" << endl;
//    tf_idf_out << tf_idf_normalized << endl;
//    //tf_idf_out << "centers.rows: " << centers.rows << endl;
//    //tf_idf_out << "centers.cols: " << centers.cols << endl;
//
///////////下面是对新图片进行descriptor提取的部分
//    Mat new_descriptor;
////    Mat tmptmp;
//    //接下来我们读一张新的图片？
//    Mat img_object_src = imread("C:/Cassandra/all_souls_000027.jpg",0);// read it into
//    Mat img_object;
//    img_object_src.convertTo(img_object, CV_32F);
//    if((img_object.rows > 500)||(img_object.cols) > 500){
//    cv::resize(img_object, img_object, Size(), 0.25, 0.25, INTER_CUBIC);
//    }
//    if(img_object.empty())
//	{
//		return -1;
//	}
//	imwrite("C:/Cassandra/all_souls_000027_grey.jpg",img_object);
////	dst_image.convertTo(tmptmp,CV_32FC1,1.0/255);
////    Mat tmptmp(dst_image.rows, dst_image.cols, CV_32FC1, Scalar(0)); // 仅仅是用来进行数据处理的，不好看
//// let us resize the image.
////    float *out233 = tmptmp.ptr<float>(0);// pointer, pointing at the output image
////    unsigned char *in233  = dst_image.ptr<unsigned char>(0); // char??
//
////    for (size_t i=dst_image.rows*dst_image.cols; i > 0; i--)
////        {
////            *out233 = (float(in233[0] + in233[1] + in233[2]))/3.0f;  //  averaging 3 channels.
////            out233++; // move the pointer, only one channel.
////            in233+=3; // move the pointer, 3 channels.
////        }
////    namedWindow("haha", CV_WINDOW_AUTOSIZE);
////    imshow("haha",tmptmp);
////    waitKey();
//    PyramidParams p02; // struct
//    HessianAffineParams par02;
//    p02.threshold = par02.threshold;
//    AffineShapeParams ap02; // struct
//    ap02.maxIterations = par02.max_iter;
//    ap02.patchSize = par02.patch_size;
//    ap02.mrSize = par02.desc_factor;
//    SIFTDescriptorParams sp02;  //SIFT?
//    sp02.patchSize = par02.patch_size;
//    AffineHessianDetector detector01(img_object, p02, ap02, sp02);
//    g_numberOfPoints = 0;
//    detector01.detectPyramidKeypoints(img_object);
//    detector01.outputKeypoints(new_descriptor);
//    //cout << "new_descriptor: "<< new_descriptor << endl;
//    cout << "new_descriptor.rows: " << new_descriptor.rows << " new_descriptor.cols: " << new_descriptor.cols << endl;
//    cout << "centers.rows: " << centers.rows << " centers.cols: " << centers.cols << endl;
//    //cout << typeof(centers) << endl;
////    tf_idf_out << "AAA: "<< endl;
////    tf_idf_out << AAA << endl;
////    tf_idf_out << "centers " << endl;
////    tf_idf_out << centers << endl;
//    //从这里开始，每个descriptor去跟所有的cluster center去计算距离
//    //还是写个函数吧
//    Mat new_des_cluster;
//    Mat object_cluster;
//    find_cluster_center(new_descriptor, centers, new_des_cluster,object_cluster);
////    cout << new_des_cluster << endl;
//    cout << "new_des_cluster.rows: " << new_des_cluster.rows << " new_des_cluster.cols: " << new_des_cluster.cols << endl;
////    tf_idf_out << "new_des_cluster.rows: " << new_des_cluster.rows << " new_des_cluster.cols: " << new_des_cluster.cols << endl;
////    cout << new_des_cluster << endl;
//    tf_idf_out << new_des_cluster << endl;
//    cout << "object_cluster.rows: " << object_cluster.rows << " object_cluster.cols: " << object_cluster.cols << endl;
//    tf_idf_out << "object_cluster.rows: " << object_cluster.rows << " object_cluster.cols: " << object_cluster.cols << endl;
//    //cout << object_cluster << endl;
////    tf_idf_out << object_cluster << endl;
//    tf_idf_out.close();
//
////    现在开始，VW的距离比较过程
//    Mat New_VW;
//    Mat VW_distance = Mat::zeros(1,image_count,CV_64F);
//    new_des_cluster.convertTo(New_VW,CV_64F);//double
//    double min_store = 0;
//    int min_location = 0;
//    cout << "old_VW.rows: " << old_VW.rows << "old_VW.cols: " << old_VW.cols << endl;
//    for(int i=0; i < image_count ; ++i){
//        Mat inter_VW = New_VW.mul(tf_idf_normalized.row(i))-old_VW.row(i).mul(tf_idf_normalized.row(i));
//        VW_distance.at<double>(0,i) = inter_VW.dot(inter_VW);
//        if(i==0){
//            min_store = VW_distance.at<double>(0,i);
//
//        }
//        if(VW_distance.at<double>(0,i) < min_store){
//
//            min_location = 0;
//            min_store = VW_distance.at<double>(0,i);
//        }
//    }
//    cout << VW_distance << endl;
//    cout << "min value: " << min_store << endl;
//    cout << "min location: " << min_location << endl;
//
//    // RANSAC匹配
//    //Mat H = findHomography( obj, scene, CV_RANSAC );
//    //vector<DMatch> RANSAC_matches;
//    //vector<Point2f> train_pts, query_pts;
//    //vector<unsigned char> match_mask;
//    //BFMatcher desc_matcher(NORM_HAMMING);
//    int key_image_count = 1; //先留这么一个。因为后面估计会有不止一张要比对的database image.
//    //接下来又要读写字符串了
//    vector<string> image_dirs;
//    vector<int> image_descriptor_counts;
//    //int len_dir = strlen(buf)+strlen(suffix_dir)+1;
//    //char buf_dir[len_dir];
//    snprintf(buf_dir, len_dir, "%s%s", buf, suffix_dir); buf_dir[len_dir-1]=0;
//    ifstream t_for_ransac;
//    ifstream t_single_hesaff;
//    t_for_ransac.open(buf_dir, ios::in);
//    t_for_ransac >> image_count;
//    string image_file_string;
//    char suffix_hesaff[] = "hesaff.txt";
//    int intra;
//    char filename_short_02[_MAX_PATH];
//
//    vector<Mat> descriptor_128_dump;
//    vector< vector<Point2f> > dataset_pts;
//
//    for(int i=0; i < image_count; ++i){
//        t_for_ransac >> image_file_string;
//        t_for_ransac >> intra;
//        image_descriptor_counts.push_back(intra);
//        memset(filename_short_02, 0, _MAX_PATH);
//        image_file_string.copy(filename_short_02,image_file_string.length()-3,0);//这里image_file_string.length()代表复制几个字符，0代表复制的位置
//        *(filename_short_02+image_file_string.length()-3)='\0';
//        int len_02 = strlen(filename_short_02)+strlen(suffix_hesaff)+1;
//        char buf_hesaff[len_02];
//        memset(buf_hesaff,len_02,0);
//        snprintf(buf_hesaff, len_02, "%s%s", filename_short_02, suffix_hesaff);
//        *(buf_hesaff+len_02)='\0';
//        //cout << buf_hesaff << endl;
//        string image_dirs_string = buf_hesaff;
//
//        if(i==min_location){ //当确定是这张图片的时候
//            cout << "Target: " << image_dirs_string << endl;
//            image_dirs.push_back(image_dirs_string);
//            vector<Mat> this_image_descriptor_dump;
//            vector<Point2f> this_image_pts;
//            t_single_hesaff.open(buf_hesaff, ios::in);
//            int little_marker = 0;
//            t_single_hesaff >> little_marker;
//            if(little_marker!=128){
//                return -3;
//            }
//            t_single_hesaff >> little_marker;
//            int descriptor_count_single = little_marker;
//            for(int j=0; j < descriptor_count_single; ++j){
//                Mat descriptor_single_row = Mat::zeros(1,128,CV_32S);
//                Point2f point_local;
//                t_single_hesaff >> little_marker;
//                point_local.x = float(little_marker);
//                t_single_hesaff >> little_marker;
//                point_local.y = float(little_marker);
//                for(int j_02=0; j_02 < 128; ++j_02){
//                    t_single_hesaff >> little_marker;
//                    descriptor_single_row.at<int>(0,j_02) = little_marker;
//                }
//
//                this_image_pts.push_back(point_local);
//                Mat intra_mat;
//                //
//                if(j==0){
//                    this_image_descriptor_dump.push_back(descriptor_single_row);
//                }else{
//                    //中间？miao是拼接的结果
//                    vconcat(this_image_descriptor_dump[0], descriptor_single_row, intra_mat);
//                    this_image_descriptor_dump.pop_back();
//                    this_image_descriptor_dump.push_back(intra_mat);
//                    intra_mat.release();
//                }
//            }
//            FlannBasedMatcher matcher;
//            vector< DMatch > matches;
//            Mat GGG;
//            this_image_descriptor_dump[0].copyTo(GGG);
//            new_descriptor.convertTo(new_descriptor,CV_32F);
//            GGG.convertTo(GGG,CV_32F);
//            matcher.match( new_descriptor, GGG, matches );
//            double max_dist = 0; double min_dist = 100;
//
//            //-- Quick calculation of max and min distances between keypoints
//            for( int i = 0; i < new_descriptor.rows; i++ ){
//                    double dist = matches[i].distance;
//                    if( dist < min_dist ) min_dist = dist;
//                    if( dist > max_dist ) max_dist = dist;
//            }
//            cout << "Max dist : " << max_dist;
//            cout << "Min dist : " << min_dist;
//              //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
//            std::vector< DMatch > good_matches;
//
//            for( int i = 0; i < new_descriptor.rows; i++ ){
//                if( matches[i].distance < 3*min_dist ){
//                    good_matches.push_back( matches[i]);
//                }
//            }
////////////////////////////////这后面是不确定的部分
////            Mat img_matches;
//////            drawMatches( img_object, keypoints_object, dst_image, keypoints_scene,
//////                       good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//////                       vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
////
////            drawMatches( img_object, keypoints_object, dst_image, keypoints_scene,
////                        good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
////                        vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
////
////          //-- Localize the object
////            std::vector<Point2f> obj;
////            std::vector<Point2f> scene;
////
////            for( int i = 0; i < good_matches.size(); i++ )
////            {
////            //-- Get the keypoints from the good matches
////                obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
////                scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
////            }
////
////            Mat H = findHomography( obj, scene, CV_RANSAC );
////
////          //-- Get the corners from the image_1 ( the object to be "detected" )
////            std::vector<Point2f> obj_corners(4);
////            obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
////            obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
////            std::vector<Point2f> scene_corners(4);
////
////            perspectiveTransform( obj_corners, scene_corners, H);
////
////          //-- Draw lines between the corners (the mapped object in the scene - image_2 )
////            line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
////            line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
////            line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
////            line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
////
////          //-- Show detected matches
////            imshow( "Good Matches & Object detection", img_matches );
//
//        }
//
//
//        image_file_string.clear();
//    }
//
//
//
//
//    return 0;
//}
//
