//
//  ViewController.m
//  HyperLandmark-iOS
//
//  Created by Le Hoang Vu on 1/25/19.
//  Copyright © 2019 Le Hoang Vu. All rights reserved.
//

#import "ViewController.h"

#import <opencv2/opencv.hpp>
#import <opencv2/videoio/cap_ios.h>

#include <iostream>
#include <vector>
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "ldmarkmodel.h"

#include "opencv+parallel_for_.h"

using namespace std;
using namespace cv;

@interface ViewController () <CvVideoCameraDelegate> {
    ldmarkmodel* _modelt;
    std::vector<cv::Mat>* _currentShape;
}

@property (nonatomic, strong) CvVideoCamera* videoCamera;
@property (weak, nonatomic) IBOutlet UIView *cameraView;


@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    self.videoCamera = [[CvVideoCamera alloc] initWithParentView:self.cameraView];
    self.videoCamera.delegate = self;
    self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionFront;
    self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset640x480;
    self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
    self.videoCamera.defaultFPS = 30;
    self.videoCamera.grayscaleMode = NO;
    
    NSString* haarPath = [NSBundle.mainBundle pathForResource:@"haar_facedetection" ofType:@"xml"];
    NSString* modelPath = [NSBundle.mainBundle pathForResource:@"landmark-model" ofType:@"bin"];
    _modelt = new ldmarkmodel([haarPath UTF8String]);
    
    if(!load_ldmarkmodel([modelPath UTF8String], *_modelt)) {
        std::cout << "Modle Opening Failed." << [modelPath UTF8String] << std::endl;
    }
    
    _currentShape = new std::vector<cv::Mat>(MAX_FACE_NUM);
}

- (IBAction)startCapture:(id)sender {
    [self.videoCamera start];
}

- (void)processImage:(cv::Mat &)image
{
    NSTimeInterval start = [NSDate timeIntervalSinceReferenceDate];
 
    _modelt->track(image, *_currentShape);
    cv::Vec3d eav;
    _modelt->EstimateHeadPose((*_currentShape)[0], eav);
    _modelt->drawPose(image, (*_currentShape)[0], 50);
    
    // 检测张嘴闭嘴
      if (!(*_currentShape)[0].empty()) {
          int numLandmarks = (*_currentShape)[0].cols / 2;
          
          // 获取关键点 62, 57, 66 的坐标
          cv::Point2f point62 = cv::Point2f((*_currentShape)[0].at<float>(62), (*_currentShape)[0].at<float>(62 + numLandmarks));
          cv::Point2f point57 = cv::Point2f((*_currentShape)[0].at<float>(57), (*_currentShape)[0].at<float>(57 + numLandmarks));
          cv::Point2f point66 = cv::Point2f((*_currentShape)[0].at<float>(66), (*_currentShape)[0].at<float>(66 + numLandmarks));
          
          // 计算差值
          float diff66_62 = abs(point66.y - point62.y);
          float diff57_66 = abs(point57.y - point66.y);
          
          // 判断张嘴或闭嘴
          if (diff66_62 > diff57_66) {
              NSLog(@"张嘴");
          } else {
              NSLog(@"闭嘴");
          }
      }
    // 注释掉或删除绘制特征点的代码
//    parallel_for_(cv::Range(0, MAX_FACE_NUM), [&](const cv::Range& range){
//        for (int i = range.start; i < range.end; i++){
//            if (!(*_currentShape)[i].empty()){
//                int numLandmarks = (*_currentShape)[i].cols / 2;
//                for (int j = 0; j < numLandmarks; j++){
//                    int x = (*_currentShape)[i].at<float>(j);
//                    int y = (*_currentShape)[i].at<float>(j + numLandmarks);
//                    cv::circle(image, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1);
//                }
//            }
//        }
//    });
    NSTimeInterval end = [NSDate timeIntervalSinceReferenceDate];
    NSLog(@">>>> FPS: %f", 1.0f/(end - start));
}


@end
