/*
 * Copyright (C) 2008-12 Michal Perdoch
 * All rights reserved.
 *
 * This file is part of the HessianAffine detector and is made available under
 * the terms of the BSD license (see the COPYING file).
 *
 */

#include <iostream>
#include <fstream>
#include "opencv2\opencv.hpp"
#include <stdio.h>
#include <string>
#include <vector>
#include <math.h>
#include <map>
#include <cmath>
#include <stdlib.h>
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <list>
#include "pyramid.h"
#include "helpers.h"
#include "affine.h"
#include "siftdesc.h"
#include "direct.h"
#include "io.h"

using namespace cv;
using namespace std;

typedef vector<int> vec;
struct Image_statistic
{
    int image_code; //图片的编号
    int des_count;  //descriptor的个数
    int K;  //一共多少个词汇
    Mat the_vocabulary;  //一个mat，记录每个descriptor所属的词汇
};

//Hessian parameters
struct HessianAffineParams
{

   float threshold;
   int   max_iter;
   float desc_factor;
   int   patch_size;
   bool  verbose;  //??
   HessianAffineParams()
      {
         threshold = 16.0f/3.0f;
         max_iter = 16;
         desc_factor = 3.0f*sqrt(3.0f);
         patch_size = 41;
         verbose = false;
      }
};

int g_numberOfPoints = 0;
int g_numberOfAffinePoints = 0;

struct Keypoint
{
   float x, y, s;
   float a11,a12,a21,a22;//这四个量是干啥的？
   float response;
   int type;
   unsigned char desc[128];//128维的数值？
};

struct AffineHessianDetector : public HessianDetector, AffineShape, HessianKeypointCallback, AffineShapeCallback
{
   const Mat image;
   SIFTDescriptor sift;
   vector<Keypoint> keys;//关键点
   //int key_count;//关键点的个数，要求从外面可以读取

public:
   AffineHessianDetector(const Mat &image, const PyramidParams &par, const AffineShapeParams &ap, const SIFTDescriptorParams &sp) :
      HessianDetector(par),
      AffineShape(ap),
      image(image),
      sift(sp)
      {
         this->setHessianKeypointCallback(this);
         this->setAffineShapeCallback(this);
      }
      // SIFT?
   void onHessianKeypointDetected(const Mat &blur, float x, float y, float s, float pixelDistance, int type, float response)
      {
         g_numberOfPoints++;
         findAffineShape(blur, x, y, s, pixelDistance, type, response);
      }

   void onAffineShapeFound(
      const Mat &blur, float x, float y, float s, float pixelDistance,
      float a11, float a12,
      float a21, float a22,
      int type, float response, int iters)
      {
         // convert shape into a up is up frame
         rectifyAffineTransformationUpIsUp(a11, a12, a21, a22);

         // now sample the patch
         if (!normalizeAffine(image, x, y, s, a11, a12, a21, a22))
         {
            // compute SIFT
            sift.computeSiftDescriptor(this->patch);
            // store the keypoint
            keys.push_back(Keypoint());
            Keypoint &k = keys.back();
            k.x = x; k.y = y; k.s = s; k.a11 = a11; k.a12 = a12; k.a21 = a21; k.a22 = a22; k.response = response; k.type = type;
            for (int i=0; i<128; i++)
               k.desc[i] = (unsigned char)sift.vec[i];
            // debugging stuff
            if (0)
            {
               cout << "x: " << x << ", y: " << y
                    << ", s: " << s << ", pd: " << pixelDistance
                    << ", a11: " << a11 << ", a12: " << a12 << ", a21: " << a21 << ", a22: " << a22
                    << ", t: " << type << ", r: " << response << endl;
               for (size_t i=0; i<sift.vec.size(); i++)
                  cout << " " << sift.vec[i];
               cout << endl;
            }
            g_numberOfAffinePoints++;
         }
      }

//楼下这个函数就是特征点的文件输出
    void exportKeypoints(ostream &out, ostream &out2)
    {
        out << 128 << endl;// 128维SIFT
        out << keys.size() << endl;//检测到的特征点的个数
        out2 << keys.size() << endl;//我加的部分，给index文件也来一个。
         //接下来，对于每个特征点
         for(size_t i=0; i<keys.size(); ++i)
         {
             //注意Keypoint结构
            Keypoint &k = keys[i];
            float sc = AffineShape::par.mrSize * k.s;
            Mat A = (Mat_<float>(2,2) << k.a11, k.a12, k.a21, k.a22);//这是啥矩阵？
            SVD svd(A, SVD::FULL_UV);//SVD都上了啊
            float *d = (float *)svd.w.data;
            d[0] = 1.0f/(d[0]*d[0]*sc*sc);
            d[1] = 1.0f/(d[1]*d[1]*sc*sc);

            A = svd.u * Mat::diag(svd.w) * svd.u.t();
            //文件中第二行开始前5个数值： k.x k.y A(0,0) A(0,1) A(1,1)
            out << k.x << " " << k.y << " " << A.at<float>(0,0) << " " << A.at<float>(0,1) << " " << A.at<float>(1,1);
            //后面才是128维的数值
            for (int j=0; j<128; ++j){
               out << " " << int(k.desc[j]);
            }
            out << endl;
         }
    }



    void exportKeypoints_Extra(ostream &out)
    {
         out << 128 << endl;// 128维SIFT
         out << keys.size() << endl;//检测到的特征点的个数
         //接下来，对于每个特征点
         for (size_t i=0; i<keys.size(); ++i)
         {
             //注意Keypoint结构
            Keypoint &k = keys[i];
            float sc = AffineShape::par.mrSize * k.s;
            Mat A = (Mat_<float>(2,2) << k.a11, k.a12, k.a21, k.a22);//这是啥矩阵？
            SVD svd(A, SVD::FULL_UV);//SVD都上了啊
            float *d = (float *)svd.w.data;
            d[0] = 1.0f/(d[0]*d[0]*sc*sc);
            d[1] = 1.0f/(d[1]*d[1]*sc*sc);

            A = svd.u * Mat::diag(svd.w) * svd.u.t();
            //文件中第二行开始前5个数值： k.x k.y A(0,0) A(0,1) A(1,1)
            out << k.x << " " << k.y << " " << A.at<float>(0,0) << " " << A.at<float>(0,1) << " " << A.at<float>(1,1);
            //后面才是128维的数值
            for (size_t j=0; j<128; ++j)
               out << " " << int(k.desc[j]);
            out << endl;
         }
    }

    void outputKeypoints(Mat &Key_mat, vector<KeyPoint> &img_KeyPoint)
    {
        Mat tmp = Mat::zeros(keys.size(),128,CV_32S);
        for (size_t i=0; i<keys.size(); ++i)
         {
            Keypoint &k2 = keys[i];
            KeyPoint img_KeyPoint_tmp;
            img_KeyPoint_tmp.pt.x = float(k2.x);
            img_KeyPoint_tmp.pt.y = float(k2.y);
            img_KeyPoint_tmp.size = 10;
            img_KeyPoint.push_back(img_KeyPoint_tmp);
            for (size_t j=0; j<128; ++j){
               tmp.at<int>(i,j) = int(k2.desc[j]);
               }
         }
        tmp.copyTo(Key_mat);
    }
};
//这里我们来写我们的归一化函数
    void mat_normalization(Mat &src, Mat &dst)
    {
//        cout << "K = "<<src.cols << endl; // K
//        cout << "image_count = " << src.rows << endl;// image_count
        for(int i=0; i < src.rows; ++i){
            double value_sum = 0;
            for(int j=0; j < src.cols; ++j){//K
                value_sum = value_sum + src.at<double>(i,j)*src.at<double>(i,j);
            }
            value_sum = sqrt(value_sum);
            //这个地方都出现负值了。这肯定是有问题的。建议改成double如何？
            //cout << "value_sum[" << i << "]: " <<value_sum <<endl;
            for(int j=0; j < src.cols;++j){
                    if(src.at<double>(i,j)==0){
                        dst.at<double>(i,j) = 0;
                    }else{
                        dst.at<double>(i,j) = src.at<double>(i,j)/value_sum;}
            }
        }
    }
    //将Mat中的数据输出到CSV文件里的函数
    void exportKeypoints_CSV(Mat &des_mat,ostream &out){
        for(int i=0; i<128; ++i){
            if(i==0){out << "SIFT" << i;}else{
            out << "," <<"SIFT" << i;}
        }
        out << endl;
        for(int i=0; i<des_mat.rows; ++i){
            for(int j=0; j<128; ++j){
                if(j==0){
                out << des_mat.at<float>(i,j);
            }else{
                out << "," << des_mat.at<float>(i,j);}
            }
            out << endl;
        }
    }

    //距离计算，对位进行比对
    float calculate_dis(Mat &mat_A, Mat &mat_B)
    {
        if((mat_A.rows!=1)||(mat_B.rows!=1)){
            cout << "Errow: one of the Mat's row number does not equal to 1." << endl;
            return -1;}
        if(mat_A.cols!=mat_B.cols){
            cout << "Error: two Mats should have equal col number." << endl;
            return -2;}
        float Mat_distance = 0;
        for(int i=0; i< mat_A.cols; ++i)
        {
//            cout << mat_A.rows << " " << mat_A.cols << endl;
//            cout << mat_A << endl;
//            cout << mat_A.at<float>(0,0) << endl;
//            cout << mat_B << endl;
//            cout << mat_B.at<float>(0,i) << endl;
            float Cal_A = mat_A.at<float>(0,i)/100;
            float Cal_B = mat_B.at<float>(0,i)/100;
            float Cal_C = abs(Cal_A - Cal_B);
            Mat_distance = Mat_distance + pow(Cal_C,2);
        }
        float Mat_distance_233 = sqrt(Mat_distance);
        //没有办法，太大了，只能进行两次开方
        return Mat_distance_233;
    }

////新图片的descriptor与原cluster center之间进行比对的函数。
void find_cluster_center(Mat &descriptor_mat, Mat &cluster_mat, Mat &cluster_output, Mat &cluster_all)
{
    //先统计一共多少个descriptor
    if(descriptor_mat.cols != 128){
        return;
    }
    if(cluster_mat.cols != 128){
        return;
    }
    //
    Mat cluster_temp = Mat::zeros(1,cluster_mat.rows,CV_32S);
    Mat cluster_all_temp = Mat::zeros(descriptor_mat.rows, cluster_mat.rows, CV_32F);
    cout << descriptor_mat.rows << " " << cluster_mat.rows <<endl;
    Mat dump_A;
    Mat dump_B;
//        Mat dump_C;
    for(int i=0; i < descriptor_mat.rows; ++i)
    {
        float dump = 0;
        float dump_cal = 0;
        int cluster_marker = 0;
        for(int j =0; j < cluster_mat.rows; ++j)
        {
            descriptor_mat.row(i).copyTo(dump_A);
//                cout << dump_A << endl;
            cluster_mat.row(j).copyTo(dump_B);
            // 问题是，这两个并不都是float啊。cluster那个是，所以要对descriptor进行强制的类型转换。
            dump_A.convertTo(dump_A, CV_32F);
            dump_B.convertTo(dump_B, CV_32F);
//                cout << dump_A.rows << " " << dump_A.cols << endl;
//                cout << dump_B.rows << " " << dump_B.cols << endl;
            //cout << dump_B << endl;
            dump_cal = calculate_dis(dump_A, dump_B);
            //cout << dump_cal << endl;
            cluster_all_temp.at<float>(i,j) = dump_cal;
            //我们要记录的是最小值。千万小心。
            if(j==0){
                dump = dump_cal;
            }
            if(dump > dump_cal){
                dump = dump_cal;
                cluster_marker = j;
            }
            dump_cal = 0;
            dump_A.release();
            dump_B.release();
//                dump_C.release();
        }
        cluster_temp.at<int>(0,cluster_marker) = cluster_temp.at<int>(0,cluster_marker) + 1;
    }
    cluster_temp.copyTo(cluster_output);
    cluster_all_temp.copyTo(cluster_all);
}

void write_KeyPoint(vector<KeyPoint> &input_kpt, ostream &out){
    out << input_kpt.size() << endl;
    for(size_t i=0; i< input_kpt.size(); ++i){
            out << input_kpt[i].pt.x << " " << input_kpt[i].pt.y << " ";
            out << input_kpt[i].size << " " << input_kpt[i].angle << " ";
            out << input_kpt[i].response << " " << input_kpt[i].octave << " ";
            out << input_kpt[i].class_id << endl;
        }
}

void write_SIFT_descriptor(Mat &input_SIFT_mat, ostream &out){
    out << input_SIFT_mat.rows << endl;
    for(int i=0; i < input_SIFT_mat.rows; ++i){
        for(int j=0; j < input_SIFT_mat.cols; ++j){
            out << input_SIFT_mat.at<float>(i,j) << " ";
        }
        out << endl;
    }
}

Mat VW_average_operation(vector<Mat> VW_storage, int cluster_number){
    Mat intra = Mat::zeros(1,cluster_number,CV_64F);
    for(size_t i=0; i< VW_storage.size(); ++i){
        if(VW_storage[i].rows!=1 || VW_storage[i].cols!= cluster_number || VW_storage[i].type()!=CV_32S){
            cout << "ERROR!!!"<< endl;
        }

        for(int j=0; j< cluster_number; ++j){
            intra.at<double>(0,j) += double(VW_storage[i].at<int>(0,j))/double(VW_storage.size());
        }
    }
    return intra;
}

bool locate_key_image(int i, int key_image_start, int key_image_num, Mat &Ranking_Mat){
    // Note, that the ranking mat should have acsending sequence.
    if( (Ranking_Mat.type()!=CV_32S) || (Ranking_Mat.rows!=1)){
        cout << "error! locate_key_image error!!!" << endl;
        return -4;
    }
    bool exsit = false;
    for(int j=key_image_start; j< (key_image_start+key_image_num); ++j){
        if(Ranking_Mat.at<int>(0,j)==i){
        exsit = true;}
    }
    return exsit;
}

class CBrowseDir
{
    //vector<Keypoint> keys_in_image;

protected:
    //存放初始目录的绝对路径，以'\'结尾
    char m_szInitDir[_MAX_PATH];

public:
    //缺省构造器
    CBrowseDir();//Keypoint &k = keys[i];

    //设置初始目录为dir，如果返回false，表示目录不可用
    bool SetInitDir(const char *dir);

    //开始遍历初始目录及其子目录下由filespec指定类型的文件
    //filespec可以使用通配符 * ?，不能包含路径。
    //如果返回false，表示遍历过程被用户中止
    bool BeginBrowse(const char *filespec);

protected:
    //遍历目录dir下由filespec指定的文件
    //对于子目录,采用迭代的方法
    //如果返回false,表示中止遍历文件
    bool BrowseDir(const char *dir,const char *filespec);

    //函数BrowseDir每找到一个文件,就调用ProcessFile
    //并把文件名作为参数传递过去
    //如果返回false,表示中止遍历文件
    //用户可以覆写该函数,加入自己的处理代码
    virtual bool ProcessFile(const char *filename);

    //函数BrowseDir每进入一个目录,就调用ProcessDir
    //并把正在处理的目录名及上一级目录名作为参数传递过去
    //如果正在处理的是初始目录,则parentdir=NULL
    //用户可以覆写该函数,加入自己的处理代码
    //比如用户可以在这里统计子目录的个数
    virtual void ProcessDir(const char *currentdir,const char *parentdir);
};

CBrowseDir::CBrowseDir()
{
    //用当前目录初始化m_szInitDir
    getcwd(m_szInitDir,_MAX_PATH);

    //如果目录的最后一个字母不是'\',则在最后加上一个'\'
    int len=strlen(m_szInitDir);
    if (m_szInitDir[len-1] != '\\')
        strcat(m_szInitDir,"\\");
}

bool CBrowseDir::SetInitDir(const char *dir)
{
    //先把dir转换为绝对路径
    if (_fullpath(m_szInitDir,dir,_MAX_PATH) == NULL)
        return false;

    //判断目录是否存在
    if (_chdir(m_szInitDir) != 0)
        return false;

    //如果目录的最后一个字母不是'\',则在最后加上一个'\'
    int len=strlen(m_szInitDir);
    if (m_szInitDir[len-1] != '\\')
        strcat(m_szInitDir,"\\");

    return true;
}

bool CBrowseDir::BeginBrowse(const char *filespec)
{
    ProcessDir(m_szInitDir,NULL);
    return BrowseDir(m_szInitDir,filespec);
}


//能再来一个变量不
bool CBrowseDir::BrowseDir(const char *dir,const char *filespec)
{
    _chdir(dir);
    char suffix_dir[] = "/image_index.txt";//后缀名
    int len_dir = strlen(dir)+strlen(suffix_dir)+1;
    char buf_dir[len_dir];
    snprintf(buf_dir, len_dir, "%s%s", dir, suffix_dir); buf_dir[len_dir-1]=0;//
    cout << dir << endl;
    if (!access(buf_dir,0)){
    ofstream out(buf_dir, ios::trunc);//干掉原先的文件
    out.close();
    cout<<"file " << buf_dir << " exist."<<endl;}else{
    ofstream out(buf_dir);//踹门！写文件啦！
    out.close();
    cout<<"file " << buf_dir << " does not exist."<<endl;}
    //首先查找dir中符合要求的文件, in io.h
    long hFile;
    _finddata_t fileinfo;// what?
    int count_lalala = 0;
    // filespec? jpg后缀
    if ((hFile=_findfirst(filespec,&fileinfo)) != -1)
    {
        do
        {
            //检查是不是目录
            //如果不是,则进行处理
            if (!(fileinfo.attrib & _A_SUBDIR))
            {
                char filename[_MAX_PATH];// length: _MAX_PATH
                memset(filename,0,_MAX_PATH);
                strcpy(filename,dir);
                strcat(filename,fileinfo.name);// what is fileinfo.name?
                //here we GOOOOOOOOOOOOOOOOOO!

                //Mat tmp;// read it into tem
                Mat image_scene_tmp = imread(filename,0); // create a image?
                //image_scene_tmp.convertTo(tmp, CV_32F);
                //
                char filename02[_MAX_PATH];
                memset(filename02, 0, _MAX_PATH);
                char filename03[_MAX_PATH];
                memset(filename03, 0, _MAX_PATH);
                count_lalala += 1;
                strcpy(filename02,"C:/Cassandra/hereafter/grey_image");
                itoa(count_lalala,filename03,10);
                strcat(filename02,filename03);
                strcat(filename02,".jpg");
                imwrite(filename02,image_scene_tmp);
                cv::resize(image_scene_tmp, image_scene_tmp, Size(), 0.25, 0.25, INTER_CUBIC);
                double t1 = 0;//这个是计时用的
                {
                    t1 = getTime(); //?
                    int minHessian = 400;
                    SurfFeatureDetector detector(minHessian);
                    vector<KeyPoint> keypoints_scene_tmp;
                    detector.detect( image_scene_tmp, keypoints_scene_tmp );
                    SiftDescriptorExtractor extractor;
                    Mat descriptors_scene_tmp;
                    extractor.compute( image_scene_tmp, keypoints_scene_tmp, descriptors_scene_tmp );
                    //在这里改变一下计数值
                    //detector.key_count = g_numberOfPoints;

                    //因为detector 本身是 AffineHessianDetector的一个例子，而AffineHessianDetector里面有对g_numberOfPoints进行变化
                    cout << "Detected " << keypoints_scene_tmp.size() << " keypoints in " << getTime()-t1 << " sec." << endl;
                    // write the file
                    char suffix_SIFT[] = ".SIFT.txt";//后缀名
                    char suffix_kpts[] = ".kpts.txt";
                    // change
                    char filename_short[_MAX_PATH];
                    memset(filename_short, 0, _MAX_PATH);//还是初始化一下比较安全啊
                    memcpy(filename_short, filename, sizeof(char)*(strlen(filename)-4));//把“.jpg”4个字节去掉

                    int len_SIFT = strlen(filename_short)+strlen(suffix_SIFT)+1;
                    char buf_SIFT[len_SIFT];
                    memset(buf_SIFT,0,len_SIFT);
                    snprintf(buf_SIFT, len_SIFT, "%s%s", filename_short, suffix_SIFT);buf_SIFT[len_SIFT-1] = 0;//

                    int len_kpts = strlen(filename_short)+strlen(suffix_kpts)+1;
                    char buf_kpts[len_kpts];
                    memset(buf_kpts,0,len_kpts);
                    snprintf(buf_kpts, len_kpts, "%s%s", filename_short, suffix_kpts);buf_kpts[len_kpts-1] = 0;//

                    ofstream out_index;
                    ofstream out_SIFT;
                    ofstream out_kpts;
                    out_index.open(buf_dir, ios::out|ios::app);// For: "/image_index.txt"
                    out_index << filename << " " << keypoints_scene_tmp.size() << endl;
                    if (access(buf_dir,0)){
                        cout << "Warning!!! Unable to open file." << endl;
                        out_index.close();}else{
                    out_index.close();}
                    out_SIFT.open(buf_SIFT,ios::out|ios::trunc);
                    write_SIFT_descriptor(descriptors_scene_tmp, out_SIFT);
                    out_SIFT.close();
                    out_kpts.open(buf_kpts,ios::out|ios::trunc);
                    write_KeyPoint(keypoints_scene_tmp, out_kpts);
                    out_kpts.close();
                    //写到这儿，还是在这张图片里
                    //for (int i = 0; i <= detector.key_count; ++i){
                    //Keypoint &k = detector.keys[j];
                    //}
                }
                cout << filename << endl;// here, print the filename
                if (!ProcessFile(filename))
                    return false;
            }
        } while (_findnext(hFile,&fileinfo) == 0);
        _findclose(hFile);
    }

    //查找dir中的子目录
    //因为在处理dir中的文件时，派生类的ProcessFile有可能改变了
    //当前目录，因此还要重新设置当前目录为dir。
    //执行过_findfirst后，可能系统记录下了相关信息，因此改变目录
    //对_findnext（类型）没有影响。

    _chdir(dir);//what's this? dir?
    if ((hFile=_findfirst("*.*",&fileinfo)) != -1)
    {
        do
        {
            //检查是不是目录
            //如果是,再检查是不是 . 或 ..
            //如果不是,进行迭代
            //iteration

            if ((fileinfo.attrib & _A_SUBDIR))
            {
                if (strcmp(fileinfo.name,".") != 0 && strcmp
                    (fileinfo.name,"..") != 0)
                {
                    char subdir[_MAX_PATH];
                    strcpy(subdir,dir);
                    strcat(subdir,fileinfo.name);
                    strcat(subdir,"\\");
                    ProcessDir(subdir,dir);// did nothing
                    //end of iteration?
                    if (!BrowseDir(subdir,filespec))
                        return false;
                }
            }
        } while (_findnext(hFile,&fileinfo) == 0);
        _findclose(hFile);
    }
    return true;
}

bool CBrowseDir::ProcessFile(const char *filename)
{
    return true;
}

// do nothing
void CBrowseDir::ProcessDir(const char *currentdir,const char *parentdir)
{
}

//从CBrowseDir派生出的子类，用来统计目录中的文件及子目录个数
class CStatDir:public CBrowseDir
{
protected:
    int m_nFileCount;   //保存文件个数
    int m_nSubdirCount; //保存子目录个数

public:
    //缺省构造器
    CStatDir()
    {
        //初始化数据成员m_nFileCount和m_nSubdirCount
        m_nFileCount=m_nSubdirCount=0;
    }

    //返回文件个数
    int GetFileCount()
    {
        return m_nFileCount;
    }

    //返回子目录个数
    int GetSubdirCount()
    {
        //因为进入初始目录时，也会调用函数ProcessDir，
        //所以减1后才是真正的子目录个数。
        return m_nSubdirCount-1;
    }

protected:
    //覆写虚函数ProcessFile，每调用一次，文件个数加1
    virtual bool ProcessFile(const char *filename)
    {
        m_nFileCount++;
        return CBrowseDir::ProcessFile(filename);
    }

    //覆写虚函数ProcessDir，每调用一次，子目录个数加1
    virtual void ProcessDir
        (const char *currentdir,const char *parentdir)
    {
        m_nSubdirCount++;
        CBrowseDir::ProcessDir(currentdir,parentdir);
    }
};

//
int main()
{
    //vector<AffineHessianDetector>;
    //获取目录名
//    char buf[256];
//    printf("input the document dir:");
//    gets(buf);

    char *buf = "C:/Cassandra/here";
    //构造类对象
    //important
    CStatDir statdir;

    //设置要遍历的目录
    if (!statdir.SetInitDir(buf))
    {
        puts("Dir does not exist.");
        return -1;
    }

    //开始遍历
    statdir.BeginBrowse("*.jpg*");

    printf("Number of images: %d\nNumber of sub_dir:%d\n",statdir.GetFileCount(),statdir.GetSubdirCount());

    char suffix_dir_statistics[] = "/image_database_statistics.txt";//
    int len_dir_statistics = strlen(buf)+strlen(suffix_dir_statistics)+1;
    char buf_dir_statistics[len_dir_statistics];
    snprintf(buf_dir_statistics, len_dir_statistics, "%s%s", buf, suffix_dir_statistics); buf_dir_statistics[len_dir_statistics-1]=0;//

    ofstream out_statistics;
    out_statistics.open(buf_dir_statistics, ios::out|ios::trunc);
    out_statistics << statdir.GetFileCount() << endl;
    out_statistics.close();

    char suffix_dir[] = "/image_index.txt";//
    int len_dir = strlen(buf)+strlen(suffix_dir)+1;
    char buf_dir[len_dir];
    snprintf(buf_dir, len_dir, "%s%s", buf, suffix_dir); buf_dir[len_dir-1]=0;//

//    //下面的部分我们把图片总数给加到刚才的文件里
//    std::ifstream read_index;
//    int read_index_length = 0;
//    read_index.open(buf_dir,ios::in|ios::binary);      // open input file
//    read_index.seekg(0, std::ios::end);    // go to the end
//    read_index_length = read_index.tellg();           // report location (this is the length)
//    read_index.seekg(0, std::ios::beg);    // go back to the beginning
//    char read_index_buffer[read_index_length];    // allocate memory for a buffer of appropriate dimension
//    memset(read_index_buffer, 0, read_index_length);//不初始化会出问题的
//    read_index.read(read_index_buffer, read_index_length);       // read the whole file into the buffer
//    read_index.close(); //t完成了使命
//    cout << read_index_buffer << endl;
//
//    ofstream out;
//    out.open(buf_dir, ios::out|ios::trunc);
//    out << statdir.GetFileCount() << endl << read_index_buffer << endl;
//    out.close();
    //然后我们可以开始弄kmeans了。 另外刚才那步其实略多余。

    ifstream in_index;
    ifstream in_statistics;
    ifstream in_SIFT;
    in_index.open(buf_dir, ios::in);
    in_statistics.open(buf_dir_statistics, ios::in);
    char dir_buffer[_MAX_PATH];
    memset(dir_buffer, 0, _MAX_PATH);

    int image_count = 0;//图片总数
    int des_count = 0;
    int des_count_all = 0;
    in_statistics >> image_count;
    in_statistics.close();

    vector<Mat> Descriptor_company;

    //从这个循环开始，我们要一个一个的对付图片。读取他们的descriptor，完成矩阵拼接。
    //我们是不是得弄个足够大的Mat容器？

    int num_of_cluster = 32; // 8类
    Mat bestLabels, centers, clustered;
    Mat des_for_each = Mat::zeros(image_count, 1,CV_32S);//这个用来保存每张图片里的descriptor的数量，后面有用
    int l;
    for(l=0; l< image_count; ++l){
        //下面这两句话是在index文件里的
        in_index >> dir_buffer;//这里错了。这可是那个图片的地址。
        in_index >> des_count;
        des_count_all = des_count_all + des_count;
        des_for_each.at<int>(l,0) = des_count;//这里做个小记录

        //从这里开始我们进入hesaff文件了，但是地址还不对啊
        char dir_front[_MAX_PATH];
        memset(dir_front, 0, _MAX_PATH);//还是初始化一下比较安全啊
        memcpy(dir_front, dir_buffer, sizeof(char)*(strlen(dir_buffer)-4));//把“.jpg”4个字节去掉
        char suffix[] = ".SIFT.txt";
        int len = strlen(dir_front)+strlen(suffix)+1;
        char buf_hesaff[len];
        //memset(buf_go,len,0);
        snprintf(buf_hesaff, len, "%s%s", dir_front, suffix); buf_hesaff[len-1]=0;//
        //
        in_SIFT.open(buf_hesaff, ios::in);

        // mat merge
        Mat p = Mat::zeros(des_count, 128, CV_32F);// for kmeans
        int SIFT_part[des_count][128];
        for(int i = 0; i < des_count; ++i){
        //SIFT的128个分量
            for(int k = 0; k < 128; ++k){
                in_SIFT >> SIFT_part[i][k];
                p.at<float>(i,k) = float(SIFT_part[i][k]);
            }
        }
        if(l == 0){
            Descriptor_company.push_back(p);
        }else{
            Mat miao;//中间？miao是拼接的结果
            vconcat(Descriptor_company[0], p, miao);
            //这里少了一步。我们应该干掉原来的Descriptor_company[0]才对啊。
            miao.convertTo(miao, CV_32F);
            Descriptor_company.pop_back();
            Descriptor_company.push_back(miao);
        }

    }
    in_index.close();
    //CSV输出
    ofstream out_csv;
    out_csv.open("C:/Cassandra/here/kmean.csv",ios::out | ios::trunc);
    exportKeypoints_CSV(Descriptor_company[0], out_csv);
    out_csv.close();

    //Kmeans
    cv::kmeans(Descriptor_company[0], num_of_cluster, bestLabels,
            TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 100, 0.1),
            10, KMEANS_RANDOM_CENTERS, centers);

    //小插曲。这里我们来统计一下每一幅图片里各个词汇的个数。那么我们先做一个mat
    vector<Mat> count_the_des;

    Mat num_by_des = Mat::zeros(image_count,num_of_cluster,CV_32S); // 这个用来判断每张图片里有没有某一个descriptor
    Mat num_by_des_sum = Mat::zeros(1,num_of_cluster,CV_32S);//每个类别里总共的descriptor数？
    Mat des_max_frequence = Mat::zeros(1,image_count,CV_32S); //这个Mat用来记录每张图片里出现频率最高的词汇的词汇频率

    int bestLabel_position = 0;//ii是descriptor的序号标志

    for(int i=0; i < image_count;++i){
        //Frozen是个中间过渡用的Mat
        Mat VW_temp= Mat::zeros(1,num_of_cluster,CV_32S);
        int VW_count_temp = des_for_each.at<int>(i,0);//这一步已经对了。能正确地读出一张图片里的descriptor数。
        for(int j=0; j < VW_count_temp; ++j){
            //VW_count_temp是这张图片里的keypoints的数量
            VW_temp.at<int>(0,(bestLabels.at<int>(bestLabel_position, 0))) = VW_temp.at<int>(0,(bestLabels.at<int>(bestLabel_position, 0))) +1;
            ++bestLabel_position;
        }
//        cout << VW_temp << endl;
        count_the_des.push_back(VW_temp);
    }

    char suffix_tf_idf_dir[] = "/tf_idf.txt";//
    int len_tf_idf_dir = strlen(buf)+strlen(suffix_tf_idf_dir)+1;
    char buf_dir2[len_tf_idf_dir];
    snprintf(buf_dir2, len_tf_idf_dir, "%s%s", buf, suffix_tf_idf_dir); buf_dir2[len_tf_idf_dir-1]=0;//
    std::ofstream tf_idf_out;
    tf_idf_out.open(buf_dir2,ios::trunc);

    tf_idf_out << "Step1: " << endl;
    tf_idf_out << image_count << " " << des_count_all << " " << num_of_cluster << endl;
    for(int i=0; i < image_count; ++i){
        //des_max_frequence.at<int>(0,i) = 0;//其实可以不写这一句
        for(int j=0; j < num_of_cluster; ++j){
            tf_idf_out << count_the_des[i].at<int>(0,j) << " ";
            if(count_the_des[i].at<int>(0,j) > des_max_frequence.at<int>(0,i)){
                    des_max_frequence.at<int>(0,i) = count_the_des[i].at<int>(0,j);
            }
        }
        tf_idf_out << endl;
    }

    tf_idf_out << "Step2: 每个类别是否出现过" << endl;
    for(int j=0; j < image_count ; ++j){
        for(int i=0; i < num_of_cluster; ++i){
            //注意这里的顺序，很容易错的。
            if(count_the_des[j].at<int>(0,i)!=0){
                num_by_des.at<int>(j,i) = 1;
                //num_by_des_sum记录了每个类别都有多少张图片
                num_by_des_sum.at<int>(0,i) = num_by_des_sum.at<int>(0,i) + 1;
            }
            tf_idf_out << num_by_des.at<int>(j,i) << " ";
        }
        tf_idf_out << endl;
    }

    tf_idf_out << "Step3: 每张图片中的最大频度" << endl;
    for(int i=0; i < image_count; ++i){
        tf_idf_out << des_max_frequence.at<int>(0,i) << " ";
    }
    tf_idf_out << endl;
    Mat des_tf = Mat::zeros(image_count,num_of_cluster,CV_64F);
    Mat des_idf = Mat::zeros(1,num_of_cluster,CV_64F);
    Mat des_tf_idf = Mat::zeros(image_count,num_of_cluster,CV_64F);

    tf_idf_out << "Step4: tf" << endl;
    for(int j=0; j < image_count; ++j){
        for(int i=0; i < num_of_cluster; ++i){
            des_tf.at<double>(j,i) = 0.5 + 0.5 * (double(count_the_des[j].at<int>(0,i))/double(des_max_frequence.at<int>(0,j)));
            if(j==0){
            des_idf.at<double>(0,i) = log10((2.0 + double(image_count))/(1.0 + double(num_by_des_sum.at<int>(0,i))));}
            des_tf_idf.at<double>(j,i) = des_tf.at<double>(j,i) * des_idf.at<double>(0,i);// i和j又搞错了哦亲
            tf_idf_out << des_tf.at<double>(j,i) << " ";
        }
        tf_idf_out << endl;
    }
    tf_idf_out << "Step5: idf" << endl;
    tf_idf_out << num_by_des_sum << endl;
    tf_idf_out << des_idf << endl;
    tf_idf_out << "Step6: tf_idf" << endl;
    for(int j=0; j < image_count; ++j){
        for(int i=0; i < num_of_cluster; ++i){
            tf_idf_out << des_tf_idf.at<double>(j,i) << " ";
        }
        tf_idf_out << endl;
    }
    //归一化
    Mat tf_idf_normalized = des_tf_idf.clone();
    mat_normalization(des_tf_idf,tf_idf_normalized);
    vector<Mat> all_des;
    for(int i=0; i < image_count; ++i){
        if(i == 0){
            all_des.push_back(count_the_des[i]);
        }else{
            Mat miao;//中间？miao是拼接的结果
            vconcat(all_des[0], count_the_des[i], miao);
            //这里少了一步。我们应该干掉原来的Descriptor_company[0]才对啊。
            all_des.pop_back();
            all_des.push_back(miao);
        }
    }
    Mat tf_idf_finished;
    Mat old_VW;
    all_des[0].convertTo(old_VW,CV_64F);
    all_des[0].convertTo(tf_idf_finished,CV_64F);
    all_des.pop_back();
    //这里出错了。两个矩阵的数据类型不同。
    //tf_idf_finished : CV_64F
    //tf_idf_normalized :
    tf_idf_finished = tf_idf_finished.mul(tf_idf_normalized);
    tf_idf_out << "tf_idf_normalized:" << endl;
    tf_idf_out << tf_idf_normalized << endl;
    //tf_idf_out << "centers.rows: " << centers.rows << endl;
    //tf_idf_out << "centers.cols: " << centers.cols << endl;

/////////下面是对新图片进行descriptor提取的部分
//    Mat tmptmp;
    //接下来我们读一张新的图片？
    char new_coming_file[] = "C:/Cassandra/orz.jpg";
    Mat img_object = imread(new_coming_file,0);// read it into
    //Mat img_object;
    //img_object_src.convertTo(img_object, CV_32F);
    if((img_object.rows > 500)||(img_object.cols) > 500){
    cv::resize(img_object, img_object, Size(), 0.25, 0.25, INTER_CUBIC);
    }
    if(img_object.empty())
	{
	    cout << "Unable to open image."<<endl;
		return -1;
	}
	imwrite("C:/Cassandra/orz_grey.jpg",img_object);

    int minHessian = 400;
    SurfFeatureDetector detector(minHessian);
    vector<KeyPoint> object_image_KeyPoint;
    detector.detect( img_object, object_image_KeyPoint );
    SiftDescriptorExtractor extractor;
    Mat new_descriptor;//object图片的128维SIFT矩阵
    extractor.compute( img_object, object_image_KeyPoint, new_descriptor );
    g_numberOfPoints = 0;
    //
    char suffix_SIFT[] = ".SIFT.txt";//后缀名
    char suffix_kpts[] = ".kpts.txt";
    // change
    char filename_short[_MAX_PATH];
    memset(filename_short, 0, _MAX_PATH);//还是初始化一下比较安全啊
    memcpy(filename_short, new_coming_file, sizeof(char)*(strlen(new_coming_file)-4));//把“.jpg”4个字节去掉

    int len_SIFT = strlen(filename_short)+strlen(suffix_SIFT)+1;
    char buf_SIFT[len_SIFT];
    int len_kpts = strlen(filename_short)+strlen(suffix_kpts)+1;
    char buf_kpts[len_kpts];
    snprintf(buf_SIFT, len_SIFT, "%s%s", filename_short, suffix_SIFT);buf_SIFT[len_SIFT-1] = 0;//
    snprintf(buf_kpts, len_kpts, "%s%s", filename_short, suffix_kpts);buf_kpts[len_kpts-1] = 0;//
    //
    char suffix_dir_extra[] = "/new_image_index.txt";//
    int len_dir_extra = strlen(buf)+strlen(suffix_dir_extra)+1;
    char buf_dir_extra[len_dir_extra];
    snprintf(buf_dir_extra, len_dir_extra, "%s%s", buf, suffix_dir_extra); buf_dir_extra[len_dir_extra-1]=0;//
    ofstream out_index;
    ofstream out_SIFT;
    ofstream out_kpts;
    out_index.open(buf_dir_extra, ios::out | ios::app);// For: "/image_index.txt"
    out_SIFT.open(buf_SIFT,ios::out|ios::trunc);
    write_SIFT_descriptor(new_descriptor, out_SIFT);
    out_kpts.open(buf_kpts,ios::out|ios::trunc);
    out_index << new_coming_file << " " << new_descriptor.size() << endl;
    write_KeyPoint(object_image_KeyPoint, out_kpts);
    out_index.close();
    out_SIFT.close();
    out_kpts.close();
    //
    ofstream t_extra_output;
    t_extra_output.open("C:/Cassandra/here/new_image_keys.txt",ios::out|ios::trunc);
    //从这里开始，每个descriptor去跟所有的cluster center去计算距离
    //还是写个函数吧
    Mat new_des_cluster;
    Mat object_cluster;
    find_cluster_center(new_descriptor, centers, new_des_cluster,object_cluster);
    t_extra_output << new_des_cluster << endl;
    t_extra_output << "object_cluster.rows: " << object_cluster.rows << " object_cluster.cols: " << object_cluster.cols << endl;
    t_extra_output.close();

//    现在开始，VW的距离比较过程
    Mat New_VW;
    Mat VW_distance = Mat::zeros(1,image_count,CV_64F);
    new_des_cluster.convertTo(New_VW,CV_64F);//double
    double min_store = 0;
    int min_location = 0;
    cout << "old_VW.rows: " << old_VW.rows << "old_VW.cols: " << old_VW.cols << endl;
    for(int i=0; i < image_count ; ++i){
        Mat inter_VW = New_VW.mul(tf_idf_normalized.row(i))-old_VW.row(i).mul(tf_idf_normalized.row(i));
        VW_distance.at<double>(0,i) = inter_VW.dot(inter_VW);
        if(i==0){
            min_store = VW_distance.at<double>(0,i);
        }
        if(VW_distance.at<double>(0,i) < min_store){
            min_location = i;
            min_store = VW_distance.at<double>(0,i);
        }
    }
//    cout << VW_distance << endl;
    cout << "min value: " << min_store << endl;
    cout << "min location: " << min_location << endl;


    ////3/25日改动
    Mat BoW_dis_ranking;
    int top_ranking_limit = 5;// database ranking limit. We compare these images to the query image.
    cv::sortIdx(VW_distance, BoW_dis_ranking, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
    cout << "BoW Ranking: min to max." << endl;
    for(int i=0; i<top_ranking_limit; ++i){
        cout << "Top "<< i << ": " << BoW_dis_ranking.at<int>(0,i) << endl;
    }

    //接下来又要读写字符串了
    vector<string> image_dirs;
    vector<string> image_dirs_SIFT;
    vector<string> image_dirs_kpts;
    vector<Mat> img_scene_mat;
    vector<int> image_descriptor_counts;
    //int len_dir = strlen(buf)+strlen(suffix_dir)+1;
    //char buf_dir[len_dir];
    snprintf(buf_dir, len_dir, "%s%s", buf, suffix_dir); buf_dir[len_dir-1]=0;
    ifstream ransac_image_count;
    ransac_image_count >> image_count;
    ransac_image_count.close();
    ifstream dir_for_ransac;
    ifstream dir_for_SIFT;
    ifstream dir_for_kpts;
    dir_for_ransac.open(buf_dir, ios::in);
    string image_file_string;
    int intra;
    char filename_short_02[_MAX_PATH];


    for(int i=0; i < image_count; ++i){
        dir_for_ransac >> image_file_string;
        dir_for_ransac >> intra;

        //拆开写成两个循环比较好
        if(locate_key_image(i, 0, top_ranking_limit, BoW_dis_ranking)==true){
            image_dirs.push_back(image_file_string);
            image_descriptor_counts.push_back(intra);
        }
        image_file_string.clear();
    }
    dir_for_ransac.close();

    //解决方法。再sort一次就好了。
//    BoW_dis_ranking
    //sort_and_take
    //Mat Key_img_ranking_origin_first_query = BoW_dis_ranking.colRange(0,top_ranking_limit);
    Mat Key_img_ranking_first_query;
    //cout << Key_img_ranking_origin_first_query << endl;
    cv::sortIdx(BoW_dis_ranking.colRange(0,top_ranking_limit),Key_img_ranking_first_query, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
    cout << Key_img_ranking_first_query << endl;
    for(int i=0; i<top_ranking_limit;++i){
        cout << image_dirs[i] << endl;
    }
    vector<string> image_dirs_tmp;
    vector<int> image_descriptor_counts_tmp;
    for(int i=0; i < top_ranking_limit; ++i){
        string dir_tmp = image_dirs[Key_img_ranking_first_query.at<int>(0,i)];
        image_dirs_tmp.push_back(dir_tmp);
        image_descriptor_counts_tmp.push_back(image_descriptor_counts[Key_img_ranking_first_query.at<int>(0,i)]);
    }
    vector <int>().swap(image_descriptor_counts);
    vector <string>().swap(image_dirs);
    for(int i=0; i< top_ranking_limit; ++i){
        image_dirs.push_back(image_dirs_tmp[i]);
        image_descriptor_counts.push_back(image_descriptor_counts_tmp[i]);
    }
    vector <int>().swap(image_descriptor_counts_tmp);
    vector <string>().swap(image_dirs_tmp);

    cout << endl;
    for(int i=0; i<top_ranking_limit;++i){
        cout << image_dirs[i] << " " << image_descriptor_counts[i] << endl;

    }

    //到这里为止都没问题了
    //vector<Mat> ;
    vector<Mat> scene_image_descriptor_first_query;
    vector< vector<KeyPoint> > scene_image_KeyPoint_first_query;
    vector< vector< DMatch > > matches_first_query;
    vector< vector< DMatch > > good_matches_first_query;
    vector<Mat> img_matches_draw_first_query;
    vector< vector<Point2f> > obj_pts_first_query;
    vector< vector<Point2f> > scene_pts_first_query;
    vector<Mat> homograph_for_matches_first_query;
    vector< vector<Point2f> > obj_corners_first_query;
    vector< vector<Point2f> > scene_corners_first_query;
    vector<int> abandoned_image;
    int well_matching_image = 0;
    for(int i=0; i< top_ranking_limit; ++i){
            //
            Mat img_scene_tmp;
            bool has_enough_good_match = true;
            //当确定是这张图片的时候
            char filename_image_dir[_MAX_PATH];
            memset(filename_image_dir, 0, _MAX_PATH);
            image_dirs[i].copy(filename_image_dir,image_dirs[i].length(),0);
            memset(filename_short_02, 0, _MAX_PATH);
            //image_dirs.push_back(image_file_string);
            img_scene_tmp = imread(image_dirs[i], 0);
            cv::resize(img_scene_tmp, img_scene_tmp, Size(), 0.25, 0.25, INTER_CUBIC);
            img_scene_mat.push_back(img_scene_tmp);

            image_dirs[i].copy(filename_short_02,image_dirs[i].length()-4,0);//这里image_file_string.length()代表复制几个字符，0代表复制的位置
            filename_short_02[image_dirs[i].length()-4]= 0;
            int len_SIFT = strlen(filename_short_02)+strlen(suffix_SIFT)+1;
            char buf_SIFT[len_SIFT];
            memset(buf_SIFT,len_SIFT,0);
            snprintf(buf_SIFT, len_SIFT, "%s%s", filename_short_02, suffix_SIFT);buf_SIFT[len_SIFT-1] = 0;
            string image_SIFT_string = buf_SIFT;

            int len_kpts = strlen(filename_short_02)+strlen(suffix_kpts)+1;
            char buf_kpts[len_kpts];
            memset(buf_kpts,len_kpts,0);
            snprintf(buf_kpts, len_kpts, "%s%s", filename_short_02, suffix_kpts);buf_kpts[len_kpts-1] = 0;
            string image_kpts_string = buf_kpts;

            cout << "Target: " << image_dirs[i] << endl;
            image_dirs_SIFT.push_back(image_SIFT_string);
            image_dirs_kpts.push_back(image_kpts_string);
            vector<Mat> scene_image_descriptor_tmp;
            vector<KeyPoint> scene_image_KeyPoint_tmp;
            dir_for_SIFT.open(buf_SIFT, ios::in);
            dir_for_kpts.open(buf_kpts, ios::in);
            int little_marker = 0;
            dir_for_kpts >> little_marker;
            dir_for_SIFT >> little_marker;
            int descriptor_count_single = little_marker;
            for(int j=0; j < descriptor_count_single; ++j){
                Mat descriptor_single_row = Mat::zeros(1,128,CV_32S);
                KeyPoint KeyPoint_local;
                for(int j_02=0; j_02 < 128; ++j_02){
                    dir_for_SIFT >> little_marker;
                    descriptor_single_row.at<int>(0,j_02) = little_marker;
                }
                {
                    dir_for_kpts >> KeyPoint_local.pt.x;
                    dir_for_kpts >> KeyPoint_local.pt.y;
                    dir_for_kpts >> KeyPoint_local.size;
                    dir_for_kpts >> KeyPoint_local.angle;
                    dir_for_kpts >> KeyPoint_local.response;
                    dir_for_kpts >> KeyPoint_local.octave;
                    dir_for_kpts >> KeyPoint_local.class_id;
                }
                scene_image_KeyPoint_tmp.push_back(KeyPoint_local);
                Mat intra_mat;
                //
                if(j==0){
                    scene_image_descriptor_tmp.push_back(descriptor_single_row);
                }else{
                    //中间？miao是拼接的结果
                    vconcat(scene_image_descriptor_tmp[0], descriptor_single_row, intra_mat);
                    scene_image_descriptor_tmp.pop_back();
                    scene_image_descriptor_tmp.push_back(intra_mat);
                    intra_mat.release();
                }
            }
            dir_for_SIFT.close();
            dir_for_kpts.close();

            scene_image_KeyPoint_first_query.push_back(scene_image_KeyPoint_tmp);
            scene_image_descriptor_first_query.push_back(scene_image_descriptor_tmp[0]);
            scene_image_descriptor_tmp.pop_back();
            FlannBasedMatcher matcher;
            vector< DMatch > matches_tmp;
            vector< DMatch > good_matches_tmp;
            Mat descriptor_intra;
            scene_image_descriptor_first_query[i].convertTo(descriptor_intra,CV_32F);
            new_descriptor.convertTo(new_descriptor,CV_32F);
            matcher.match( new_descriptor, descriptor_intra, matches_tmp );

            // Ranking matches on distance.
            Mat match_check = Mat::zeros(1,matches_tmp.size(),CV_64F);
            Mat match_check_Idx;

            //-- Quick calculation of max and min distances between keypoints
            for( int j = 0; j < new_descriptor.rows; j++ ){
                    match_check.at<double>(0,j) = matches_tmp[j].distance;
            }

            cv::sortIdx(match_check,match_check_Idx, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
            double max_dist = match_check.at<double>(0,match_check_Idx.at<int>(0,(matches_tmp.size()-1)));
            double min_dist = match_check.at<double>(0,match_check_Idx.at<int>(0,0));
            cout << "Max dist : " << max_dist << endl;
            cout << "Min dist : " << min_dist << endl;

            if(match_check.at<double>(0,match_check_Idx.at<int>(0,20)) > 280.0){
                cout << "Not enough good matches." << endl;
                has_enough_good_match = false;
            }

            //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )

            int num_of_good_matches = 0;
            for( int j = 0; j < new_descriptor.rows; j++ ){
                if((matches_tmp[match_check_Idx.at<int>(0,j)].distance < 1.5* min_dist) && (matches_tmp[match_check_Idx.at<int>(0,j)].distance < 280.0)){
                    good_matches_tmp.push_back( matches_tmp[j]);
                    num_of_good_matches++;
                }
                if(num_of_good_matches == 30){ // take only top 30
                    break;
                }
                if(j > 100){
                    break;
                }
            }
            cout << "num_of_good_matches: " << num_of_good_matches << endl;
            matches_first_query.push_back(matches_tmp);
            good_matches_first_query.push_back(good_matches_tmp);


            if(has_enough_good_match == true){
                cout << image_dirs[i] << " is a good match to query image. Accepted." << endl;
                Mat img_matches;
                drawMatches( img_object, object_image_KeyPoint, img_scene_tmp, scene_image_KeyPoint_first_query[i],
                            good_matches_tmp, img_matches, Scalar::all(-1), Scalar::all(-1),
                            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

                //-- Localize the object
                vector<Point2f> obj_tmp;
                vector<Point2f> scene_tmp;

                for(int j=0; j < good_matches_tmp.size(); j++){
                //-- Get the keypoints from the good matches
                    obj_tmp.push_back( object_image_KeyPoint[ good_matches_tmp[j].queryIdx ].pt );
                    scene_tmp.push_back( scene_image_KeyPoint_first_query[i][ good_matches_tmp[j].trainIdx ].pt );
                }

                obj_pts_first_query.push_back(obj_tmp);
                scene_pts_first_query.push_back(scene_tmp);

                Mat homograph_tmp = findHomography( obj_tmp, scene_tmp, CV_RANSAC );//
                homograph_for_matches_first_query.push_back(homograph_tmp);
                //-- Get the corners from the image_1 ( the object to be "detected" )
                std::vector<Point2f> obj_corners(4);
                // draw the four corners of the query image.
                obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
                obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );

                std::vector<Point2f> scene_corners(4);
                //--
                perspectiveTransform( obj_corners, scene_corners, homograph_tmp);

                obj_corners_first_query.push_back(obj_corners);
                scene_corners_first_query.push_back(scene_corners);

                //-- Draw lines between the corners (the mapped object in the scene - image_2 )
                line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 255), 4 );
                line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 255), 4 );
                line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 255), 4 );
                line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 255), 4 );

                img_matches_draw_first_query.push_back(img_matches);
                imshow( image_dirs[i], img_matches );
                waitKey(0);
                destroyWindow(image_dirs[i]);
                well_matching_image += 1;
            }else{
                abandoned_image.push_back(i);
                cout << image_dirs[i] << " is NOT a good match to query image. Abandoned." << endl;
            }
            img_scene_tmp.release();



    }
    if(abandoned_image.size() != 0){
        for(size_t i = 0; i < abandoned_image.size(); ++i){
            vector<Mat>::iterator iter_Mat = img_scene_mat.begin() + abandoned_image[i] ;
            img_scene_mat.erase(iter_Mat);

            vector<string>::iterator iter_string = image_dirs.begin() + abandoned_image[i];
            image_dirs.erase(iter_string);

            iter_string = image_dirs_SIFT.begin() + abandoned_image[i];
            image_dirs_SIFT.erase(iter_string);

            iter_string = image_dirs_kpts.begin() + abandoned_image[i];
            image_dirs_kpts.erase(iter_string);
            //还少了两个
            vector< vector< DMatch > >::iterator iter_DMatch = matches_first_query.begin() + abandoned_image[i];
            matches_first_query.erase(iter_DMatch);
            iter_DMatch = good_matches_first_query.begin() + abandoned_image[i];
            good_matches_first_query.erase(iter_DMatch);
        }
    }
    // re-query
    int re_query_limit = 10;
    vector<Mat> VW_intra;
    for(int i=0; i<top_ranking_limit; ++i){
    VW_intra.push_back(count_the_des[BoW_dis_ranking.at<int>(0,i)]);

    }
    Mat VW_averaged = VW_average_operation(VW_intra, num_of_cluster);
    VW_averaged.convertTo(VW_averaged, CV_64F);
    //取30张图片
    //VW保存在哪儿呢？    old_VW
    //

    Mat Requery_key_image = BoW_dis_ranking.colRange(top_ranking_limit, top_ranking_limit + re_query_limit);
    Mat Requery_key_image_Idx;
    cout << BoW_dis_ranking << endl;
    cout << "Requery_key_image" << endl;
    cout << Requery_key_image << endl;
    Mat VW_distance_requery = Mat::zeros(1, re_query_limit, CV_64F);
    for(int i=0; i < Requery_key_image.rows; ++i){
        Mat inter_VW = VW_averaged.mul(tf_idf_normalized.row(Requery_key_image.at<int>(0,i)))
            -old_VW.row(Requery_key_image.at<int>(0,i)).mul(tf_idf_normalized.row(Requery_key_image.at<int>(0,i)));
        VW_distance_requery.at<double>(0,i) = inter_VW.dot(inter_VW);
    }
    cv::sortIdx(VW_distance_requery, Requery_key_image_Idx, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
    cout << "Requery_key_image_Idx" << endl;
    cout << Requery_key_image_Idx << endl;

////

    ifstream dir_for_requery;
    dir_for_requery.open(buf_dir, ios::in);
    vector<string> image_dirs_requery;
    vector<int> image_descriptor_counts_requery;
    // read their dirs
    for(int i=0; i < image_count; ++i){
        dir_for_requery >> image_file_string;
        dir_for_requery >> intra;

        //拆开写成两个循环比较好
        if(locate_key_image(i, 0, re_query_limit, Requery_key_image)==true){
            image_dirs_requery.push_back(image_file_string);
            cout << image_file_string << endl;
            image_descriptor_counts_requery.push_back(intra);
        }
        image_file_string.clear();
    }
    cout << "Requery_key_image: "  << Requery_key_image << endl;
    vector<string> image_dirs_requery_tmp;
    vector<int> image_descriptor_counts_requery_tmp;
    for(int i=0; i < re_query_limit; ++i){
        cout << image_dirs_requery[Requery_key_image_Idx.at<int>(0,i)] << endl;
        string intra_str = image_dirs_requery[Requery_key_image_Idx.at<int>(0,i)];
        image_dirs_requery_tmp.push_back(intra_str);
        int intra_des_count = image_descriptor_counts_requery[Requery_key_image_Idx.at<int>(0,i)];
        image_descriptor_counts_requery_tmp.push_back(intra_des_count);
    }
    vector <string>().swap(image_dirs_requery);
    vector <int>().swap(image_descriptor_counts_requery);

///////////////////////////////////////////////////////

    for(int i=0; i < re_query_limit; ++i){
        image_dirs_requery.push_back(image_dirs_requery_tmp[i]);
        image_descriptor_counts_requery.push_back(image_descriptor_counts_requery_tmp[i]);
        cout << image_dirs_requery[i] << " " << image_descriptor_counts_requery[i] << endl;

    }
    vector <string>().swap(image_dirs_requery_tmp);
    vector <int>().swap(image_descriptor_counts_requery_tmp);

    dir_for_requery.close();


////接下来我们对新的候选图片进行Spatial verification好了。

////到这里为止都没问题了
    vector<Mat> img_scene_mat_requery;
    vector<Mat> scene_image_descriptor_requery;
    vector<string> image_dirs_requery_SIFT;
    vector<string> image_dirs_requery_kpts;
    vector< vector<KeyPoint> > scene_image_KeyPoint_requery;
    vector< vector< DMatch > > matches_requery;
    vector< vector< DMatch > > good_matches_requery;
    vector<Mat> img_matches_draw_requery;
    vector< vector<Point2f> > obj_pts_requery;
    vector< vector<Point2f> > scene_pts_requery;
    vector<Mat> homograph_for_matches_requery;
    vector< vector<Point2f> > obj_corners_requery;
    vector< vector<Point2f> > scene_corners_requery;
    vector<int> abandoned_image_requery;
    int well_matching_image_requery = 0;
    for(int i=0; i< re_query_limit; ++i){
            //
            Mat img_scene_tmp;
            bool has_enough_good_match = true;
            //当确定是这张图片的时候
            char filename_image_dir[_MAX_PATH];
            memset(filename_image_dir, 0, _MAX_PATH);
            image_dirs_requery[i].copy(filename_image_dir,image_dirs_requery[i].length(),0);
            memset(filename_short_02, 0, _MAX_PATH);
            //image_dirs.push_back(image_file_string);
            img_scene_tmp = imread(image_dirs_requery[i], 0);
            cv::resize(img_scene_tmp, img_scene_tmp, Size(), 0.25, 0.25, INTER_CUBIC);
            img_scene_mat_requery.push_back(img_scene_tmp);

            image_dirs_requery[i].copy(filename_short_02,image_dirs_requery[i].length()-4,0);//这里image_file_string.length()代表复制几个字符，0代表复制的位置
            filename_short_02[image_dirs_requery[i].length()-4]= 0;
            int len_SIFT = strlen(filename_short_02)+strlen(suffix_SIFT)+1;
            char buf_SIFT[len_SIFT];
            memset(buf_SIFT,len_SIFT,0);
            snprintf(buf_SIFT, len_SIFT, "%s%s", filename_short_02, suffix_SIFT);buf_SIFT[len_SIFT-1] = 0;
            string image_SIFT_string = buf_SIFT;

            int len_kpts = strlen(filename_short_02)+strlen(suffix_kpts)+1;
            char buf_kpts[len_kpts];
            memset(buf_kpts,len_kpts,0);
            snprintf(buf_kpts, len_kpts, "%s%s", filename_short_02, suffix_kpts);buf_kpts[len_kpts-1] = 0;
            string image_kpts_string = buf_kpts;

            cout << "Target: " << image_dirs_requery[i] << endl;
            image_dirs_requery_SIFT.push_back(image_SIFT_string);
            image_dirs_requery_kpts.push_back(image_kpts_string);
            vector<Mat> scene_image_descriptor_tmp;
            vector<KeyPoint> scene_image_KeyPoint_tmp;
            dir_for_SIFT.open(buf_SIFT, ios::in);
            dir_for_kpts.open(buf_kpts, ios::in);
            int little_marker = 0;
            dir_for_kpts >> little_marker;
            dir_for_SIFT >> little_marker;
            int descriptor_count_single = little_marker;
            for(int j=0; j < descriptor_count_single; ++j){
                Mat descriptor_single_row = Mat::zeros(1,128,CV_32S);
                KeyPoint KeyPoint_local;
                for(int j_02=0; j_02 < 128; ++j_02){
                    dir_for_SIFT >> little_marker;
                    descriptor_single_row.at<int>(0,j_02) = little_marker;
                }
                {
                    dir_for_kpts >> KeyPoint_local.pt.x;
                    dir_for_kpts >> KeyPoint_local.pt.y;
                    dir_for_kpts >> KeyPoint_local.size;
                    dir_for_kpts >> KeyPoint_local.angle;
                    dir_for_kpts >> KeyPoint_local.response;
                    dir_for_kpts >> KeyPoint_local.octave;
                    dir_for_kpts >> KeyPoint_local.class_id;
                }
                scene_image_KeyPoint_tmp.push_back(KeyPoint_local);
                Mat intra_mat;
                //
                if(j==0){
                    scene_image_descriptor_tmp.push_back(descriptor_single_row);
                }else{
                    //中间？miao是拼接的结果
                    vconcat(scene_image_descriptor_tmp[0], descriptor_single_row, intra_mat);
                    scene_image_descriptor_tmp.pop_back();
                    scene_image_descriptor_tmp.push_back(intra_mat);
                    intra_mat.release();
                }
            }
            dir_for_SIFT.close();
            dir_for_kpts.close();

            scene_image_KeyPoint_requery.push_back(scene_image_KeyPoint_tmp);
            scene_image_descriptor_requery.push_back(scene_image_descriptor_tmp[0]);
            scene_image_descriptor_tmp.pop_back();
            FlannBasedMatcher matcher;
            vector< DMatch > matches_tmp;
            vector< DMatch > good_matches_tmp;
            Mat descriptor_intra;
            scene_image_descriptor_requery[i].convertTo(descriptor_intra,CV_32F);
            new_descriptor.convertTo(new_descriptor,CV_32F);
            matcher.match( new_descriptor, descriptor_intra, matches_tmp );

            // Ranking matches on distance.
            Mat match_check = Mat::zeros(1,matches_tmp.size(),CV_64F);
            Mat match_check_Idx;

            //-- Quick calculation of max and min distances between keypoints
            for( int j = 0; j < new_descriptor.rows; j++ ){
                    match_check.at<double>(0,j) = matches_tmp[j].distance;
            }

            cv::sortIdx(match_check,match_check_Idx, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
            double max_dist = match_check.at<double>(0,match_check_Idx.at<int>(0,(matches_tmp.size()-1)));
            double min_dist = match_check.at<double>(0,match_check_Idx.at<int>(0,0));
            cout << "Max dist : " << max_dist << endl;
            cout << "Min dist : " << min_dist << endl;

            if(match_check.at<double>(0,match_check_Idx.at<int>(0,20)) > 280.0){
                cout << "Not enough good matches." << endl;
                has_enough_good_match = false;
            }

            //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )

            int num_of_good_matches = 0;
            for( int j = 0; j < new_descriptor.rows; j++ ){
                if((matches_tmp[match_check_Idx.at<int>(0,j)].distance < 1.5* min_dist) && (matches_tmp[match_check_Idx.at<int>(0,j)].distance < 280.0)){
                    good_matches_tmp.push_back( matches_tmp[j]);
                    num_of_good_matches++;
                }
                if(num_of_good_matches == 30){ // take only top 30
                    break;
                }
                if(j > 100){
                    break;
                }
            }
            cout << "num_of_good_matches: " << num_of_good_matches << endl;
            matches_requery.push_back(matches_tmp);
            good_matches_requery.push_back(good_matches_tmp);


            if(has_enough_good_match == true){
                cout << image_dirs_requery[i] << " is a good match to query image. Accepted." << endl;
                Mat img_matches;
                drawMatches( img_object, object_image_KeyPoint, img_scene_tmp, scene_image_KeyPoint_requery[i],
                            good_matches_tmp, img_matches, Scalar::all(-1), Scalar::all(-1),
                            vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

                //-- Localize the object
                vector<Point2f> obj_tmp;
                vector<Point2f> scene_tmp;

                for(int j=0; j < good_matches_tmp.size(); j++){
                //-- Get the keypoints from the good matches
                    obj_tmp.push_back( object_image_KeyPoint[ good_matches_tmp[j].queryIdx ].pt );
                    scene_tmp.push_back( scene_image_KeyPoint_requery[i][ good_matches_tmp[j].trainIdx ].pt );
                }

                obj_pts_requery.push_back(obj_tmp);
                scene_pts_requery.push_back(scene_tmp);

                Mat homograph_tmp = findHomography( obj_tmp, scene_tmp, CV_RANSAC );//
                homograph_for_matches_requery.push_back(homograph_tmp);
                //-- Get the corners from the image_1 ( the object to be "detected" )
                std::vector<Point2f> obj_corners(4);
                // draw the four corners of the query image.
                obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
                obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );

                std::vector<Point2f> scene_corners(4);
                //--
                perspectiveTransform( obj_corners, scene_corners, homograph_tmp);

                obj_corners_requery.push_back(obj_corners);
                scene_corners_requery.push_back(scene_corners);

                //-- Draw lines between the corners (the mapped object in the scene - image_2 )
                line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 255), 4 );
                line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 255), 4 );
                line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 255), 4 );
                line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 255), 4 );

                img_matches_draw_first_query.push_back(img_matches);
                imshow( image_dirs_requery[i], img_matches );
                waitKey(0);
                destroyWindow(image_dirs_requery[i]);
                well_matching_image += 1;
            }else{
                abandoned_image_requery.push_back(i);
                cout << image_dirs_requery[i] << " is NOT a good match to query image. Abandoned." << endl;
            }
            img_scene_tmp.release();


    }
    if(abandoned_image_requery.size() != 0){
        for(size_t i = 0; i < abandoned_image_requery.size(); ++i){
            vector<Mat>::iterator iter_Mat = img_scene_mat_requery.begin() + abandoned_image_requery[i] ;
            img_scene_mat_requery.erase(iter_Mat);

            vector<string>::iterator iter_string = image_dirs_requery.begin() + abandoned_image_requery[i];
            image_dirs_requery.erase(iter_string);

            iter_string = image_dirs_requery_SIFT.begin() + abandoned_image_requery[i];
            image_dirs_requery_SIFT.erase(iter_string);

            iter_string = image_dirs_requery_kpts.begin() + abandoned_image_requery[i];
            image_dirs_requery_kpts.erase(iter_string);
            //还少了两个
            vector< vector< DMatch > >::iterator iter_DMatch = matches_requery.begin() + abandoned_image_requery[i];
            matches_requery.erase(iter_DMatch);
            iter_DMatch = good_matches_requery.begin() + abandoned_image_requery[i];
            good_matches_requery.erase(iter_DMatch);
        }
    }




    return 0;
}

