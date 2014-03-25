//#include <iostream>
//#include <fstream>
//#include "opencv2\opencv.hpp"
//#include <stdio.h>
//#include <string>
//#include <vector>
//#include <math.h>
//#include <map>
//#include <stdlib.h>
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
//#include <opencv2/calib3d/calib3d.hpp>
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
//    void mat_normalization(Mat &src, Mat &dst)
//    {
//        //int a = src.cols;//K
//        cout << "K = "<<src.cols <<endl;
//        //int b =  src.rows;// image_count
//        cout << "image_count = " <<src.rows << endl;
//        for(int i=0; i < src.rows; ++i){
//            float value_sum = 0;
//            for(int j=0; j < src.cols; ++j){//K
//                value_sum = value_sum + src.at<float>(i,j);
//            }
//            cout << "value_sum[" << i << "]: " <<value_sum <<endl;
//            for(int j=0; j < src.cols;++j){
//                    if(src.at<float>(i,j)==0){
//                        dst.at<float>(i,j) = 0;
//                    }else{
//                        dst.at<float>(i,j) = src.at<float>(i,j)/value_sum;}
//            }
//        }
//    }
//        float calculate_dis(Mat &mat_A, Mat &mat_B)
//    {
//        if((mat_A.rows!=1)||(mat_B.rows!=1)){
//            cout << "Errow: one of the Mat's row number does not equal to 1." << endl;
//            return -1;}
//        if(mat_A.cols!=mat_B.cols){
//            cout << "Error: two Mats should have equal col number." << endl;
//            return -2;
//        }
//        float Mat_distance = 0;
//        for(int i=0; i< mat_A.cols; ++i)
//        {
//            Mat_distance = Mat_distance + pow((mat_A.at<float>(0,i)-mat_B.at<float>(0,i)),2);
//        }
//        float Mat_distance_over = sqrt(Mat_distance);
//        return Mat_distance_over;
//    }
//
//////新图片的descriptor与原cluster center之间进行比对的函数。
//    void find_cluster_center(Mat &descriptor_mat, Mat &cluster_mat, Mat &cluster_output, Mat &cluster_all)
//    {
//        //先统计一共多少个descriptor
//        if(descriptor_mat.cols != 5){
//            return;
//        }
//        //
//        Mat cluster_temp = Mat::zeros(1,cluster_mat.rows,CV_32S);
//        Mat cluster_all_temp = Mat::zeros(descriptor_mat.rows, cluster_mat.rows, CV_32F);
//
//        for(int i=0; i < descriptor_mat.rows; ++i)
//        {
//            float dump = 0;
//            float dump_cal = 0;
//            int cluster_marker = 0;
//            Mat dump_A;
//            Mat dump_B;
//            for(int j =0; j < cluster_mat.rows; ++j)
//            {
//                descriptor_mat.row(i).copyTo(dump_A);
//                //cout << dump_A << endl;
//                cluster_mat.row(j).copyTo(dump_B);
//                //cout << dump_B << endl;
//                dump_cal = calculate_dis(dump_A, dump_B);
//                //cout << dump_cal << endl;
//                cluster_all_temp.at<float>(i,j) = dump_cal;
//                if(dump < dump_cal){
//                    dump = dump_cal;
//                    cluster_marker = j;
//                }
//                dump_A.release();
//                dump_B.release();
//            }
//            cluster_temp.at<int>(0,cluster_marker) = cluster_temp.at<int>(0,cluster_marker) + 1;
//        }
//        cluster_temp.copyTo(cluster_output);
//        cluster_all_temp.copyTo(cluster_all);
//    }
//
//int main(){
//
//    Mat img_scene = imread("C:/Cassandra/here/all_souls_000014.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//    Mat img_object = imread("C:/Cassandra/2313.jpg",CV_LOAD_IMAGE_GRAYSCALE);
//    if( !img_scene.data || !img_object.data ){
//        cout<< " --(!) Error reading images " << endl;
//        return -1;}
////-- Step 1: Detect the keypoints using SURF Detector
//    int minHessian = 400;
//    SurfFeatureDetector detector( minHessian );
//    vector<KeyPoint> keypoints_object, keypoints_scene;
//    detector.detect( img_scene, keypoints_object );
//    detector.detect( img_object, keypoints_scene );
////    for(int i=0; i < keypoints_object.size();++i){
////            KeyPoint GGG = keypoints_object[i];
////        cout << "x: " << GGG.pt.x << " y: " << GGG.pt.y<< endl;
////    }
////-- Step 2: Calculate descriptors (feature vectors)
//    SiftDescriptorExtractor extractor;
////    SurfDescriptorExtractor extractor;
//
//    Mat descriptors_object, descriptors_scene;
//    extractor.compute( img_object, keypoints_object, descriptors_object );
//    extractor.compute( img_scene, keypoints_scene, descriptors_scene );
//    //-- Step 3: Matching descriptor vectors using FLANN matcher
//    FlannBasedMatcher matcher;
//    //find a match
//    std::vector< DMatch > matches;
//    matcher.match( descriptors_object, descriptors_scene, matches );
//    //
////    cout << descriptors_object.rows << " " <<   descriptors_object.cols << endl;
////    for(int i = 0; i < descriptors_object.rows; ++i){
////        cout << descriptors_object.row(i) << endl;
////    }
//
//    double max_dist = 0; double min_dist = 100;
//
//  //-- Quick calculation of max and min distances between keypoints
//    for( int i = 0; i < descriptors_object.rows; i++ )
//    { double dist = matches[i].distance;
//        if( dist < min_dist ) min_dist = dist;
//        if( dist > max_dist ) max_dist = dist;
//    }
//
//    printf("-- Max dist : %f \n", max_dist );
//    printf("-- Min dist : %f \n", min_dist );
//
//  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
//    std::vector< DMatch > good_matches;
//
//    for( int i = 0; i < descriptors_object.rows; i++ ){
//        if( matches[i].distance < 3.5*min_dist ){
//            good_matches.push_back( matches[i]);
//        }
//    }
//
//    Mat img_matches;
//    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
//               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//
//  //-- Localize the object
//    std::vector<Point2f> obj;
//    std::vector<Point2f> scene;
//
//    for( int i = 0; i < good_matches.size(); i++ )
//    {
//    //-- Get the keypoints from the good matches
//        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
//        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
//    }
//
//    Mat H = findHomography( obj, scene, CV_RANSAC );
//
//  //-- Get the corners from the image_1 ( the object to be "detected" )
//    std::vector<Point2f> obj_corners(4);
//    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
//    obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
//    std::vector<Point2f> scene_corners(4);
//
//    perspectiveTransform( obj_corners, scene_corners, H);
//
//  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
//    line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
//    line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//
//  //-- Show detected matches
//    imshow( "Good Matches & Object detection", img_matches );
//
//    waitKey(0);
//    return 0;
//
//}
