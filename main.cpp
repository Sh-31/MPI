#include <iostream>
#include <vector>
#include <string>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <algorithm>
using namespace cv;
using namespace std;
#define s_int static_cast<int>

void Image_blurring(string image_path, int kernelSize, int argc, char** argv);
void EdageDigtion(string image_path, int argc, char *argv[]);
void ImageSharping(string image_path, int argc, char *argv[]);
void ImageRotation(string image_path , int angle ,int argc, char *argv[]);
void ImageRotation2(string image_path , int angle ,int argc, char *argv[]);
void ImageScaling(string image_path,float ScaleFactor ,int argc, char *argv[]);
void Image_Commpression(string image_path ,int lr, int argc, char *argv[]) ;
void HistogramEqualization(string image_path,int argc, char *argv[]);
void ImageColorSpace(string image_path,int color_space,int argc, char *argv[]);
void GlobalThresholding(string imag_path,int argc, char *argv[]);
void LocalThresholding(string imag_path,int argc, char *argv[]);
void Median(string imag_path,int argc, char *argv[]);
void ImageSummation(string imag_path, string image_path2, int argc, char *argv[]);
void adjustImageBrightness(const string& image_path, float alpha, int argc, char** argv);
void ImageContrast(const string& img, int new_min, int new_max, int argc, char** argv);
void HisgtramMatching(const string& img1, const string& img2, int argc, char** argv);

int main(int argc, char** argv) {
    
        cout << "Welcome to the parallel Image Processing\n" << endl;
        cout << "Please choose the operation you want to perform:\n\n" << endl;
        cout << "1- Image Blurring (Gaussian Blur)\n"
             << "2- Image Edge Detection\n"
             << "3- Image Sharping\n"
             << "4- Image Rotation\n"
             << "5- Image Scaling\n"
             << "6- Histogram Equalization\n"
             << "7- Color Space Conversion\n"
             << "8- Global Thresholding\n"
             << "9- Local Thresholding\n"
             << "10- Image Compression\n"
             << "11- Median\n"
             << "12- Image Summation\n" 
             << "13- Adjust Image Brightness\n"
             << "14- Image Contrast\n"
             << "15- Hisgtram Matching\n" << endl;

        int choice = 1; 
        string image_path;
        string image_path2;

        if (argc >= 2) {
            choice = atoi(argv[1]);
            image_path = argv[2];     
        }

        switch (choice) {
            case 1:
                // Image Blurring (Gaussian Blur)
                int KernelSize;
                KernelSize =  atoi(argv[3]);
                Image_blurring(image_path, KernelSize, argc, argv);
                break;
            case 2:
                // Image Edge Detection
                EdageDigtion(image_path, argc, argv);
                break;

            case 3:
                // Image Sharping
                ImageSharping(image_path, argc, argv);
                break;    
            case 4:
                // Image Rotation
                int angle;
                angle =  atoi(argv[3]);  

                int type;
                type =  atoi(argv[4]);
               
                if (type) 
                    ImageRotation(image_path, angle, argc, argv);
                else 
                    ImageRotation2(image_path, angle, argc, argv);
              
                break;
            case 5:
                // Image Scaling
                float ScaleFactor;
                ScaleFactor = 0.5;
                if (argc > 3) {ScaleFactor =  atoi(argv[3]);}
                ImageScaling(image_path,ScaleFactor,argc,argv);
                break;
            case 6:
                // Histogram Equalization
                HistogramEqualization(image_path,argc,argv);
                break;
            case 7:
                // Color Space Conversion grayscale=6
                int color_space;
                color_space =  atoi(argv[3]);
                ImageColorSpace(image_path, color_space, argc, argv);
                break;
            case 8:
                // Global Thresholding
                GlobalThresholding(image_path,argc,argv);
                break;
            case 9:
                // Local Thresholding
                LocalThresholding(image_path,argc,argv);
                break;
            case 10:
                // Image Compression 
                int lr;
                lr =  atoi(argv[3]);
                Image_Commpression(image_path, lr ,argc, argv);
                break;
            case 11:
                // Median
                Median(image_path,argc,argv);
                break;
            case 12:
                // Image Summation
                if (argc > 3) {image_path2 =  (argv[3]);}
                ImageSummation(image_path,image_path2,argc,argv);   
                break; 
            case 13:
                // adjust Image Brightness
                double D ;
                if (argc > 3) {D =  atof(argv[3]);}
                adjustImageBrightness(image_path,  D,  argc, argv);
                break;
            case 14:
                // Image Contrast
                int new_min, new_max;
                new_min =  atoi(argv[3]);
                new_max =  atoi(argv[4]);
                ImageContrast(image_path, new_min, new_max, argc, argv);
                break; 
            case 15:
                // Hisgtram Matching
                if (argc > 3) {image_path2 =  (argv[3]);}
                HisgtramMatching(image_path,image_path2,argc,argv);   
                break;           
            default:
                break;
        }
        
    return 0;
}

void Image_blurring(String image_path ,int kernelSize, int argc, char** argv){

    int rank, cluster_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mat image;
   
    // Generate a 2D Gaussian kernel
    double sigma = 1;
    Mat kernel1D = getGaussianKernel(kernelSize, sigma, CV_64F);
    Mat kernel2D = kernel1D * kernel1D.t(); 
    kernel2D /= sum(kernel2D)[0];

    int high, width, type, EleSize;
    double start_time,end_time;
    if (rank == 0) {
        image = imread(image_path);
        cout <<"image rows:   "<< image.rows << " image cols: " << image.cols << endl;  
        start_time = MPI_Wtime(); 

        high = image.rows;
        width = image.cols;
        type = image.type();
        EleSize = image.elemSize();
        
        MPI_Bcast(&high, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&EleSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
    } else {
     
    
        MPI_Bcast(&high, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&EleSize, 1, MPI_INT, 0, MPI_COMM_WORLD);


       image = Mat(high, width, type); 
        
    }
   

    int RowForProcess = high / cluster_size;
    int Row_Start = rank * RowForProcess;
    int Row_End = (rank == cluster_size - 1)? high : Row_Start + RowForProcess;
    int total_rows = Row_End - Row_Start;

    Mat LocalImage( total_rows , width, type); 
  
    MPI_Scatter(image.data,    (total_rows * width * EleSize) , MPI_BYTE,
                LocalImage.data,  (total_rows * width * EleSize) , MPI_BYTE,
                0, MPI_COMM_WORLD);

    
    filter2D(LocalImage, LocalImage, -1, kernel2D, Point(-1,-1), 0, BORDER_REPLICATE);

    // imshow("Image at process:" + to_string(rank) , LocalImage);
    // waitKey(0);


    MPI_Gather(LocalImage.data,   (RowForProcess *  LocalImage.cols *  LocalImage.elemSize()), MPI_BYTE,
               image.data,        (RowForProcess *  LocalImage.cols *  LocalImage.elemSize()), MPI_BYTE,
               0, MPI_COMM_WORLD); // to send different size of data we need to use MPI_Gatherv

    // Other way to send data to root process (using Non-blocking communication)

    // MPI_Request send_request;
    // MPI_Isend(LocalImage.data,total_rows * width * EleSize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &send_request);

    // if (rank == 0) { 
       
    //     for (int i = 0; i < cluster_size; i++) {
    //         MPI_Request recv_request;
    //         MPI_Status recv_status;
    //         Row_Start = i * RowForProcess;
    //         Row_End = (i  == cluster_size - 1)? high : Row_Start + RowForProcess;    
    //         total_rows = Row_End - Row_Start;
         
    //         cout << " i : " << i << " Row_Start: " << Row_Start << " Row_End: " << Row_End << " total_rows: " << total_rows << endl;
    //         MPI_Irecv(image.rowRange(Row_Start, Row_End).data, total_rows * width * EleSize , MPI_BYTE, i, 0, MPI_COMM_WORLD, &recv_request);
    //         MPI_Wait(&recv_request, &recv_status);
    //     }
      
    // }
    
    if (rank == 0) {
       
         string windowName = "Blur Image K:" + to_string(kernelSize) + " - P:" + to_string(cluster_size);
         end_time = MPI_Wtime();
         cout << "Number of process: " <<cluster_size<<" Total execution time: " << end_time - start_time << " seconds" << endl;

         imshow(windowName, image);
         waitKey(0);
    }
   

    MPI_Finalize();

}

void EdageDigtion(string image_path, int argc, char *argv[]) {

    int rank, cluster_size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mat image;
    Mat kernel = (Mat_<float>(3, 3) <<  -1, -1, -1, 
                                        -1,  8, -1,
                                        -1, -1, -1);

    int high, width, type, EleSize;
    double start_time, end_time;
    if (rank == 0) {
        image = imread(image_path);
        start_time = MPI_Wtime();
        cout <<"image rows: "<< image.rows << " image cols: " << image.cols << endl;  

        high = image.rows;
        width = image.cols;
        type = image.type();
        EleSize = image.elemSize();
        
        MPI_Bcast(&high, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&EleSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
    } else {
     
    
        MPI_Bcast(&high, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&EleSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
       
    }
     
    Mat edgeImage(high, width, type, Scalar(0, 0, 0));
   
   
    int RowForProcess = high / cluster_size;
    int Row_Start = rank * RowForProcess;
    int Row_End = (rank == cluster_size - 1)? high : Row_Start + RowForProcess;
    int total_rows = Row_End - Row_Start;
    Mat LocalImage( total_rows , width, type); 
    
    MPI_Scatter(image.data,    (total_rows * width * EleSize) , MPI_BYTE,
                LocalImage.data,  (total_rows * width * EleSize) , MPI_BYTE,
                0, MPI_COMM_WORLD);
;

    
    Mat borderedImage , resultImage;
    // copyMakeBorder(myImage, borderedImage, borderSize, borderSize, borderSize, borderSize, BORDER_REPLICATE);
   
    filter2D(LocalImage, resultImage, -1, kernel, Point(-1,-1), 0, BORDER_REPLICATE);
  

    MPI_Gather(resultImage.data,  RowForProcess  *  width * EleSize, MPI_BYTE,
               edgeImage.data,    RowForProcess  *  width * EleSize, MPI_BYTE,
               0, MPI_COMM_WORLD);
    
    // Other way to send data to root process (using Non-blocking communication)

    // MPI_Request send_request;
    // MPI_Isend(resultImage.data,total_rows * width * EleSize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &send_request);

    // if (rank == 0) { 
       
    //     for (int i = 0; i < cluster_size; i++) {
    //         MPI_Request recv_request;
    //         MPI_Status recv_status;
    //         Row_Start = i * RowForProcess;
    //         Row_End = (i  == cluster_size - 1)? high : Row_Start + RowForProcess;    
    //         total_rows = Row_End - Row_Start;
         
    //         cout << " i : " << i << " Row_Start: " << Row_Start << " Row_End: " << Row_End << " total_rows: " << total_rows << endl;
    //         MPI_Irecv(edgeImage.rowRange(Row_Start, Row_End).data, total_rows * width * EleSize , MPI_BYTE, i, 0, MPI_COMM_WORLD, &recv_request);
    //         MPI_Wait(&recv_request, &recv_status);
    //     }
      
    // }

    if (rank == 0) {
         end_time = MPI_Wtime();
        cout << "Number of process: " <<cluster_size<<" Total execution time: " << end_time - start_time << " seconds" << endl;   
         string windowName = "Edage Image ";
         imshow(windowName, edgeImage);
         waitKey(0);
         
    }
   

    MPI_Finalize();

}

void ImageSharping(string image_path, int argc, char *argv[]) {

    int rank, cluster_size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mat image;
    Mat kernel = (Mat_<float>(3,3) << 
                                     0, -1,  0,
                                    -1,  5, -1,
                                     0, -1,  0);


    int high, width, type, EleSize;

    double start_time, end_time;
    if (rank == 0) {
        image = imread(image_path);
        start_time = MPI_Wtime();
        cout <<"image rows: "<< image.rows << " image cols: " << image.cols << endl;  

        high = image.rows;
        width = image.cols;
        type = image.type();
        EleSize = image.elemSize();
        
        MPI_Bcast(&high, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&EleSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
    } else {
     
    
        MPI_Bcast(&high, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&EleSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
       
    }
     
    Mat edgeImage(high, width, type, Scalar(0, 0, 0));
   
   
    int RowForProcess = high / cluster_size;
    int Row_Start = rank * RowForProcess;
    int Row_End = (rank == cluster_size - 1)? high : Row_Start + RowForProcess;
    int total_rows = Row_End - Row_Start;
    Mat LocalImage( total_rows , width, type); 
    
    MPI_Scatter(image.data,    (total_rows * width * EleSize) , MPI_BYTE,
                LocalImage.data,  (total_rows * width * EleSize) , MPI_BYTE,
                0, MPI_COMM_WORLD);
;

    
    Mat  sharpenedImage;
    filter2D(LocalImage, sharpenedImage, -1, kernel, Point(-1,-1), 0, BORDER_REPLICATE);
   
    Mat resultImage;
    if (rank == 0) {resultImage =  Mat(high, width, type, Scalar(0, 0, 0));}
    
    MPI_Gather(sharpenedImage.data,  RowForProcess  *  width * EleSize, MPI_BYTE,
               resultImage.data,    RowForProcess  *  width * EleSize, MPI_BYTE,
               0, MPI_COMM_WORLD);
    
    // Other way to send data to root process (using Non-blocking communication)

    // MPI_Request send_request;
    // MPI_Isend(sharpenedImage.data,total_rows * width * EleSize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &send_request);

    // if (rank == 0) { 
       
    //     for (int i = 0; i < cluster_size; i++) {
    //         MPI_Request recv_request;
    //         MPI_Status recv_status;
    //         Row_Start = i * RowForProcess;
    //         Row_End = (i  == cluster_size - 1)? high : Row_Start + RowForProcess;    
    //         total_rows = Row_End - Row_Start;
         
    //         cout << " i : " << i << " Row_Start: " << Row_Start << " Row_End: " << Row_End << " total_rows: " << total_rows << endl;
    //         MPI_Irecv(resultImage.rowRange(Row_Start, Row_End).data, total_rows * width * EleSize , MPI_BYTE, i, 0, MPI_COMM_WORLD, &recv_request);
    //         MPI_Wait(&recv_request, &recv_status);
    //     }
      
    // }

    if (rank == 0) {
            end_time = MPI_Wtime();
            cout << "Number of process: " <<cluster_size<<" Total execution time: " << end_time - start_time << " seconds" << endl;
         string windowName = "Sharp Image ";
         imshow(windowName, resultImage);
         waitKey(0);
         
    }
   

    MPI_Finalize();

}


Mat rotateImage(const Mat& src , Mat& rot , Rect2f& bbox,const int &angle) {

    // https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c
    // get rotation matrix for rotating the image around its center in pixel coordinates

    cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
    rot = cv::getRotationMatrix2D(center, angle, 1.0);
    // determine bounding rectangle, cente0r not relevant
    bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 -  src.cols/2.0;
    rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;

    Mat dst;
    warpAffine(src, dst, rot, bbox.size());


    return dst ;
}

Rect2f getRotatedBoundingBox(const Mat& src,const int &angle) {
    cv::Point2f center((double)(src.cols-1)/2.0, (double)(src.rows-1)/2.0);
    RotatedRect rotatedRect(center, src.size(), angle);
    return rotatedRect.boundingRect2f();
}

void ImageRotation(string image_path , int angle ,int argc, char *argv[])  {

    int rank, cluster_size;
    double start_time, end_time; 

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mat image, ResultImage;
    if (rank == 0) {
        start_time = MPI_Wtime(); 

        image = imread(image_path, IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Error: Could not read the image." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

    }

    int high, width, type, EleSize;
    if (rank == 0) {
        high = image.rows;
        width = image.cols;
        type = image.type();
        EleSize = image.elemSize();
    }

    MPI_Bcast(&high, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&EleSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  
   int RowForProcess, Row_Start, Row_End, Col_Start, Col_End;
   if (angle < 0 && angle > -270 || angle > 90 && angle < 271){
        
        RowForProcess = high / cluster_size;
        Row_Start = ( cluster_size - rank) * RowForProcess;
        Row_End = Row_Start + RowForProcess;
        Col_Start = 0;
        Col_End = width;
        
    }else{
        RowForProcess = high / cluster_size;
        Row_Start = rank * RowForProcess;
        Row_End = Row_Start + RowForProcess;
        Col_Start = 0;
        Col_End = width;
    }

    /*
    if Image (100, 100) and cluster_size = 4
        rank 0: Rows: 0 - 24  - Cols: 0 - 99 
        rank 1: Rows: 25 - 49 - Cols: 0 - 99
        rank 2: Rows: 50 - 74 - Cols: 0 - 99
        rank 3: Rows: 75 - 99 - Cols: 0 - 99

    Each process have 4 corners:
        rank 0: (0, 0), (0, 99), (24, 0), (24, 99)  
        rank 1: (25, 0), (25, 99), (49, 0), (49, 99) 
        rank 2: (50, 0), (50, 99), (74, 0), (74, 99) 
        rank 3: (75, 0), (75, 99), (99, 0), (99, 99)
        --------------------------------------------
        rank n: (start_row, col_start), (start_row, col_end), (end_row, col_start), (end_row, col_end)
    */
    
    Mat LocalImage(RowForProcess, width, type);

    MPI_Scatter(image.data, (Row_End - Row_Start) * width * EleSize, MPI_BYTE,
                LocalImage.data, (Row_End - Row_Start) * width * EleSize, MPI_BYTE,
                0, MPI_COMM_WORLD);         

  
    Mat rot;
    Rect2f bbox;  
    Mat rotatedImage = rotateImage(LocalImage, rot, bbox, angle);


    Mat Corners(4, 1, CV_32FC2);
    // x -> col, y -> row

    
    Corners.at<Vec2f>(0) = Vec2f(Col_Start, Row_Start);
    Corners.at<Vec2f>(1) = Vec2f(Col_Start, Row_End);
    Corners.at<Vec2f>(2) = Vec2f(Col_End, Row_Start);
    Corners.at<Vec2f>(3) = Vec2f(Col_End, Row_End);


    for (int i = 0; i < 4; i++) {
        cout <<"Rank: "<< rank <<" org Corner: " << i << " x : " << Corners.at<Point2f>(i, 0).x << " y : "<< Corners.at<Point2f>(i, 0).y << endl;
    }


    Mat TransformedCorners;
    transform(Corners, TransformedCorners, rot); // be careful with the float precision errors

    for (int i = 0; i < 4; i++) {
        cout <<"Rank: "<< rank <<" New Corner: " << i << " x : " << TransformedCorners.at<Point2f>(i, 0).x << " y : "<< TransformedCorners.at<Point2f>(i, 0).y << endl;
    }


    MPI_Request send_request1 , send_request2;
    MPI_Isend(&rotatedImage.rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &send_request1);
    MPI_Isend(&rotatedImage.cols, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &send_request1);
    MPI_Isend(rotatedImage.data, rotatedImage.rows * rotatedImage.cols * EleSize, MPI_BYTE, 0, 0, MPI_COMM_WORLD,  &send_request1);
    MPI_Isend(TransformedCorners.data, 8, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &send_request2);

    

    if (rank == 0){

        Rect2f bbox = getRotatedBoundingBox(image, angle);
        int  rotatedImageWidth = static_cast<int>(bbox.width);
        int  rotatedImageHeight = static_cast<int>(bbox.height);
      
        Mat ResultImage(rotatedImageHeight, rotatedImageWidth,  type , Scalar(0,0,0) ) , BurrCorners(4 , 1 , 13);
        vector<vector<bool>> MaskImage(rotatedImageHeight, vector<bool>(rotatedImageWidth, 0)); // for just the Sharp Angle
   

       for (int i = 0; i < cluster_size; i++){

        MPI_Request recv_request;
        MPI_Status recv_status ;
        int rwidth, rhigh;
        MPI_Irecv(&rhigh, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &recv_request);   MPI_Wait(&recv_request, &recv_status);
        MPI_Irecv(&rwidth, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &recv_request);  MPI_Wait(&recv_request, &recv_status);

        Mat BurrImage(rhigh, rwidth, type);
        MPI_Irecv(BurrImage.data,rhigh * rwidth * EleSize, MPI_BYTE, i, 0, MPI_COMM_WORLD, &recv_request); MPI_Wait(&recv_request, &recv_status);

        MPI_Irecv(BurrCorners.data, 8, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &recv_request); MPI_Wait(&recv_request, &recv_status);
        
        int row_start, row_end, col_start, col_end;
        // 90 degree rotation 
      
         row_end   = abs(BurrCorners.at<Point2f>(0, 0).y);
         row_start = abs(BurrCorners.at<Point2f>(3, 0).y);
         col_start = abs(BurrCorners.at<Point2f>(0, 0).x);
         col_end   = abs(BurrCorners.at<Point2f>(3, 0).x);
         if (row_start >= row_end) swap(row_start, row_end);
         if (col_start >= col_end) swap(col_start, col_end);

        cout << "row_start: " << row_start << " row_end: " << row_end << " col_start: " << col_start << " col_end: " << col_end << endl;
       
        for (int i = row_start; i < row_end ; i++) {
            for (int j = col_start; j < col_end ; j++) {
              
               if (MaskImage[i][j] != 0 && BurrImage.at<Vec3b>(i - row_start, j - col_start) == Vec3b(0,0,0)  ) continue; 

               ResultImage.at<Vec3b>(i, j) = BurrImage.at<Vec3b>(i - row_start, j - col_start);
               MaskImage[i][j] = 1;
               
                
            }
        }
           imshow("ResultImage Image After Adding P:" + to_string(rank) , ResultImage);
           waitKey(0);
         
      

       } 
        
        end_time = MPI_Wtime();
        cout << "Number of processes: " << cluster_size << " Total execution time: " << end_time - start_time << " seconds" << endl;
        imshow("ResultImage: ", ResultImage);
        waitKey(0);

}
    

    MPI_Finalize();

}

void ImageRotation2(string image_path, int angle, int argc, char* argv[]) {

    int rank, cluster_size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mat image, LocalImage;
    int high, width, type, EleSize;
 
    if (rank == 0) {
        start_time = MPI_Wtime();

        image = imread(image_path, IMREAD_COLOR);
        if (image.empty()) {
            cerr << "Error: Could not read the image." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        high = image.rows;
        width = image.cols;
        type = image.type();
        EleSize = image.elemSize();
    }

    // Broadcast image properties to all processes
    MPI_Bcast(&high, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&EleSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for local image
    LocalImage = Mat::zeros(high, width, 0);

    if (rank == 0) {
        vector<Mat> channels; 
        split(image, channels);
        LocalImage = channels[0];
        for (int i = 1; i < cluster_size; ++i) {
            MPI_Send(channels[i].data, channels[i].rows * channels[i].cols , MPI_BYTE, i, 0, MPI_COMM_WORLD);
        }
    }
    else {
        MPI_Recv(LocalImage.data, LocalImage.rows * LocalImage.cols , MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    Point2f center(LocalImage.cols / 2.0, LocalImage.rows / 2.0);
    Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);

   
    Mat rotatedImage;
    warpAffine(LocalImage, rotatedImage, rotationMatrix, image.size());
  
    // Send the rotated image to the root process
   
    MPI_Request send_request;
    MPI_Isend(rotatedImage.data, high * width , MPI_BYTE, 0, 0, MPI_COMM_WORLD, &send_request);


    if (rank == 0) {
        

        Mat ResultImage = Mat::zeros(rotatedImage.rows, rotatedImage.cols, type);
        Mat r, g, b;

        // Receive rotated color channels from processes
        for (int i = 0; i < cluster_size; ++i) {
            MPI_Request recv_request;
            MPI_Status recv_status;
            MPI_Irecv(rotatedImage.data, high * width , MPI_BYTE, i, 0, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&recv_request, &recv_status);

            if (i == 0) b = rotatedImage.clone(); 
            if (i == 1) g = rotatedImage.clone(); 
            if (i == 2) r = rotatedImage.clone(); 
        }

       
        for (int i = 0; i < ResultImage.rows; i++) {
            for (int j = 0; j < ResultImage.cols; j++) {
                ResultImage.at<Vec3b>(i, j) = Vec3b(b.at<uchar>(i, j), g.at<uchar>(i, j), r.at<uchar>(i, j)); // Note the order: BGR
            }
        }

                
    
        end_time = MPI_Wtime();
        cout << "Number of processes: " << cluster_size << " Total execution time: " << end_time - start_time << " seconds" << endl;

        imshow("Result Image", ResultImage);
        waitKey(0);
    }

    MPI_Finalize();
}

void ImageScaling(string image_path,float ScaleFactor ,int argc, char *argv[]){
    int rank, cluster_size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);

    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    Mat image;
    image = imread(image_path, IMREAD_COLOR);

    if (image.empty()) {
        cerr << "Error: Could not read the image." << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int RowForProcess = image.rows / cluster_size;
    int start_row = rank * RowForProcess;
    int end_row = (rank == cluster_size - 1) ? image.rows : start_row + RowForProcess;
    // Be careful if last process has less rows than or higer the other processes when scaling the after image will be different

    Mat ScaledImage;
    resize(image.rowRange(start_row, end_row), ScaledImage, cv::Size(), ScaleFactor, ScaleFactor);

    MPI_Request send_request;
    MPI_Isend(ScaledImage.data, ScaledImage.total() * ScaledImage.elemSize(), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &send_request);

    if (rank == 0) {
        Mat ResultImage(image.rows * ScaleFactor, image.cols * ScaleFactor, image.type());

        for (int i = 0; i < cluster_size; i++) {

        int RowForProcess = (image.rows / cluster_size) * ScaleFactor;
        int start_row = i * RowForProcess; 
        int end_row = (i == cluster_size - 1) ? (image.rows * ScaleFactor) : start_row + RowForProcess;
        int total_rows = end_row - start_row;

        MPI_Request recv_request;
        MPI_Irecv(ResultImage.rowRange(start_row, end_row).data, 
                  (total_rows * ResultImage.cols * ResultImage.elemSize()), 
                   MPI_BYTE, i, 0, MPI_COMM_WORLD, &recv_request);

        MPI_Status recv_status;
        MPI_Wait(&recv_request, &recv_status);

        }

        end_time = MPI_Wtime();
        cout << "Number of processes: " << cluster_size << " Total execution time: " << end_time - start_time << " seconds" << endl;
        cv::imshow("Scaled Image (" + to_string(ResultImage.rows) + ", " + to_string(ResultImage.cols) + ")", ResultImage);
        cv::waitKey(0);
    }

    MPI_Finalize();

}

void  Image_Commpression(string image_path ,int lr, int argc, char** argv) {
    int rank, cluster_size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);

    int high, width, type, ElementSize;
    Mat image;

    if (rank == 0) {
        start_time = MPI_Wtime();
        image = imread(image_path, IMREAD_COLOR);

        if (image.empty()) {
            cerr << "Error: Could not read the image." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        high = image.rows;
        width = image.cols;
        type = image.type();
        ElementSize = image.elemSize();
    }

    MPI_Bcast(&high, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ElementSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int RowForProcess = high / cluster_size;
    Mat LocalImage(RowForProcess, width, type);

    MPI_Scatter(image.data, s_int(RowForProcess * width * ElementSize), MPI_BYTE,
                LocalImage.data, s_int(RowForProcess * width * ElementSize), MPI_BYTE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < LocalImage.rows; i++) {
        for (int j = 0; j < LocalImage.cols; j++) {
            auto pixel = LocalImage.at<Vec3b>(i, j);
            for (int k = 0; k < 3; k++) {
                pixel[k] >>= lr; // Shift each color channel by lr bits
            }
            LocalImage.at<Vec3b>(i, j) = pixel;
        }
    }

    MPI_Gather(LocalImage.data, s_int(RowForProcess * width * ElementSize), MPI_BYTE,
               image.data, s_int(RowForProcess * width * ElementSize), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        end_time = MPI_Wtime();
        cout << "Number of processes: " << cluster_size << " Total execution time: " << end_time - start_time << " seconds" << endl;
        imshow("new image", image);
        waitKey(0);
    }
    MPI_Finalize();

}
void HistogramEqualization(string image_path,int argc, char** argv) {
    int rank, cluster_size;
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);

    int high , width , type ,ElementSize;
    Mat image;
    vector<int> fre(256,  0) , cum(256 , 0), temcum(256, 0);


    if (rank == 0) {
        start_time = MPI_Wtime();
        image = imread(image_path, cv::IMREAD_GRAYSCALE);

        cout << "Pixal size"<< image.rows * image.cols << endl; 
        // imshow("Original Image", image);
        // waitKey(0);
        

          if (image.empty()) {
                cerr << "Error: Could not read the image." << endl;
               MPI_Abort(MPI_COMM_WORLD, 1);
            }
        high = image.rows;
        width = image.cols;
        type = image.type();
        ElementSize = image.elemSize();

    }

    MPI_Bcast(&high, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ElementSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int RowForProcess = high / cluster_size;
    double ratio = ((double) 255 / (double)(high * width));
    
    Mat LocalImage(RowForProcess, width, image.type());

    MPI_Scatter(image.data, (RowForProcess * width * ElementSize), MPI_BYTE, 
                LocalImage.data, (RowForProcess * width * ElementSize), MPI_BYTE, 0, MPI_COMM_WORLD);
   

    for (int i = 0; i < LocalImage.rows; i++) {
        for (int j = 0; j < LocalImage.cols; j++) {
            int pixel = LocalImage.at<uchar>(i, j);
            fre[pixel]++;
        }
    }

 
    if (rank != 0) {
        MPI_Request send_request;
        MPI_Isend(fre.data(), 256, MPI_INT, 0, 0, MPI_COMM_WORLD, &send_request);

    }else{

        vector<int> temp(256, 0);

        for (int i = 1; i < cluster_size; i++) {

            MPI_Request recv_request;
            MPI_Irecv(temp.data(), 256, MPI_INT, i, 0, MPI_COMM_WORLD, &recv_request);

            MPI_Status recv_status;
            MPI_Wait(&recv_request, &recv_status);

            for (int j = 0; j < 256; j++) {
                fre[j] += temp[j];
            }

        }

        cum[0] = fre[0];
        for (int i = 1; i < 256; i++) {
            cum[i] = cum[i - 1] + fre[i];
        }

       
    }

    MPI_Bcast(cum.data(), 256, MPI_INT, 0, MPI_COMM_WORLD);

    
    for (int i = 0; i < LocalImage.rows; i++) {
        for (int j = 0; j < LocalImage.cols; j++) {
            int pixel = LocalImage.at<uchar>(i, j);
            LocalImage.at<uchar>(i, j) = cum[pixel] * ratio;
        }
    }

    MPI_Gather(LocalImage.data, s_int(RowForProcess * width * ElementSize), MPI_BYTE, 
               image.data, s_int(RowForProcess * width * ElementSize), MPI_BYTE, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        end_time = MPI_Wtime();
        cout << "Number of processes: " << cluster_size << " Total execution time: " << end_time - start_time << " seconds" << endl;
        imshow("Histogram Equalization", image);
        waitKey(0);
    }
    MPI_Finalize();

}
void ImageColorSpace(string image_path,int color_space,int argc, char *argv[]) {
    int rank, cluster_size;
    double start_time, end_time; 

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mat image;
    if (rank == 0) {
        start_time = MPI_Wtime(); 
        image = imread(image_path, IMREAD_COLOR);
        
        if (image.empty()) {
            cerr << "Error: Could not read the image." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int rows, cols , type , ele_size;
    
    if (rank == 0) {
        rows = image.rows;
        cols = image.cols;
        type = image.type();
        ele_size = image.elemSize();
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ele_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
    
    int RowForProcess = rows / cluster_size;
    int Row_Start = rank * RowForProcess;
    int Row_End = (rank  == cluster_size - 1)? rows : Row_Start + RowForProcess;    
    int total_rows = Row_End - Row_Start;

    Mat local_image(total_rows, cols, type);

    MPI_Scatter(image.data,  total_rows * cols * ele_size, MPI_BYTE,
                local_image.data,  total_rows * cols * ele_size, MPI_BYTE,
                0, MPI_COMM_WORLD);

    cout << "Color space: " << color_space << endl;

    Mat LocalConvImage;
    cvtColor(local_image, LocalConvImage, color_space);
    // imshow("LocalConvImage P:" + to_string(rank) , LocalConvImage);
    // waitKey(0);

    Mat ResultImage(rows, cols, LocalConvImage.type());

    ele_size = (LocalConvImage.elemSize());

    MPI_Gather(LocalConvImage.data,  RowForProcess * cols * ele_size, MPI_BYTE,
               ResultImage.data, RowForProcess * cols * ele_size, MPI_BYTE,
               0, MPI_COMM_WORLD); // to send different size of buffer must use MPI_Gatherv

    // MPI_Request send_request;
    // MPI_Isend(LocalConvImage.data, LocalConvImage.total(), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &send_request);

    // if (rank == 0) { 
    //     int t = 0;
    //     for (int i = 0; i < cluster_size; i++) {
    //         MPI_Request recv_request;
    //         MPI_Status recv_status;
    //         Row_Start = i * RowForProcess;
    //         Row_End = (i  == cluster_size - 1)? rows : Row_Start + RowForProcess;    
    //         total_rows = Row_End - Row_Start;
         
    //         cout << " i : " << i << " Row_Start: " << Row_Start << " Row_End: " << Row_End << " total_rows: " << total_rows << endl;
    //         MPI_Irecv(ResultImage.rowRange(Row_Start, Row_End ).data, total_rows * cols * ele_size , MPI_BYTE, i, 0, MPI_COMM_WORLD, &recv_request);
    //         MPI_Wait(&recv_request, &recv_status);
    //         // imshow("ResultImage Image After Adding P:" + to_string(i) , ResultImage);
    //         // waitKey(0);
    //     }
      
    // }

    
    if (rank == 0) {

        end_time = MPI_Wtime();
        cout << "Number of processes: " << cluster_size << " Total execution time: " << end_time - start_time << " seconds" << endl;
        imshow("Color Space Image", ResultImage);
        waitKey(0);
    }



    MPI_Finalize();
}
void GlobalThresholding(string imag_path,int argc, char *argv[]) {

    int rank, cluster_size;
    double start_time, end_time; 
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   
    Mat image;

    if (rank == 0) {
        start_time = MPI_Wtime(); 
        image = imread(imag_path, IMREAD_COLOR);
        
        cout <<"image rows: "<< image.rows << " image cols: " << image.cols << endl;  
        MPI_Bcast(&image.rows, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&image.cols, 1 , MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast(&image.rows, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&image.cols, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        image = Mat( image.rows, image.cols, CV_8UC3);
        cout << "Rank " << rank << " received image" << endl;
    }

    Mat myImage( (image.rows / cluster_size) , image.cols, image.type()); 

    MPI_Scatter(image.data,   s_int ((image.rows / cluster_size) * image.cols * image.elemSize()) , MPI_BYTE,
                myImage.data, s_int ((image.rows / cluster_size) * image.cols * image.elemSize()) , MPI_BYTE,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < myImage.rows; ++i) {
        for (int j = 0; j < myImage.cols; ++j) {
            Vec3b pixel = myImage.at<Vec3b>(i, j);
            for (int c = 0; c < myImage.channels(); ++c) {
                if (pixel[c] < 128) {
                     pixel[c] = 0;
                } else {
                    pixel[c] = 255; 
                }
            }
            myImage.at<cv::Vec3b>(i, j) = pixel;
        }
    }

    MPI_Gather(myImage.data, s_int (myImage.rows *  myImage.cols  * myImage.elemSize()), MPI_BYTE,
               image.data,       s_int (myImage.rows  * myImage.cols *  myImage.elemSize()), MPI_BYTE,
               0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        end_time = MPI_Wtime(); 
        cout << "Number of process: " <<cluster_size<<" Total execution time: " << end_time - start_time << " seconds" << endl;


        string windowName = "GlobalThresholding";
        imshow(windowName, image);
        waitKey(0);
       
    }

    MPI_Finalize();

   
}

void  LocalThresholding(string image_path,int argc, char *argv[]) {
    int rank, cluster_size;
    double start_time, end_time; 

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   
    Mat image;
    if (rank == 0) {
        start_time = MPI_Wtime(); 
        image = imread(image_path, IMREAD_GRAYSCALE);

        if (image.empty()) {
            cerr << "Error: Could not read the image." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1); // Terminates MPI execution environment
        }
    }


    int rows, cols;
    if (rank == 0) {
        rows = image.rows;
        cols = image.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);


    int rows_per_process = rows / cluster_size;


    Mat local_image(rows_per_process, cols, CV_8UC1);
    MPI_Scatter(image.data, rows_per_process * cols, MPI_BYTE,
                local_image.data, rows_per_process * cols, MPI_BYTE,
                0, MPI_COMM_WORLD);


    Mat local_thresholded;
    adaptiveThreshold(local_image, local_thresholded, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 2);


    Mat thresholded_image;
    if (rank == 0) {
        thresholded_image.create(rows, cols, CV_8UC1);
    }

    MPI_Gather(local_thresholded.data, rows_per_process * cols, MPI_BYTE,
               thresholded_image.data, rows_per_process * cols, MPI_BYTE,
               0, MPI_COMM_WORLD);


    if (rank == 0) {
        end_time = MPI_Wtime();

        cout << "Number of process: " <<cluster_size<<" Total execution time: " << end_time - start_time << " seconds" << endl;

        imshow("Thresholded Image", thresholded_image);
        waitKey(0);
    }

    MPI_Finalize();

}
void  Median (string image_path,int argc, char *argv[]) {

    int rank, cluster_size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mat image;
    double start_time, end_time;
    if (rank == 0) {
        image = imread(image_path);
        start_time = MPI_Wtime();
        if (image.empty()) {
            cerr << "Error: Could not read the image." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

     
        cout <<"image rows: "<< image.rows << " image cols: " << image.cols << endl;  
      
        MPI_Bcast(&image.rows, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&image.cols, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        
    } else {
     
        MPI_Bcast(&image.rows, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&image.cols, 1 , MPI_INT, 0, MPI_COMM_WORLD);

        image = Mat( image.rows, image.cols, CV_8UC3);

        cout << "Rank " << rank << " received image" << endl;
    }
    
    

    Mat myImage( (image.rows / cluster_size) , image.cols, image.type()); 

    MPI_Scatter(image.data,    ((image.rows / cluster_size) * image.cols * image.elemSize()) , MPI_BYTE,
                myImage.data,  ((image.rows / cluster_size) * image.cols * image.elemSize()) , MPI_BYTE,
                0, MPI_COMM_WORLD);


 
    Mat borderedImage , resultImage;
   
    medianBlur(myImage, resultImage, 3);


    MPI_Gather(resultImage.data,  (resultImage.rows *  resultImage.cols  * resultImage.elemSize()), MPI_BYTE,
               image.data,        (resultImage.rows * resultImage.cols *  resultImage.elemSize()), MPI_BYTE,
               0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        end_time = MPI_Wtime();
        cout << "Number of processes: " << cluster_size << " Total execution time: " << end_time - start_time << " seconds" << endl;
         string windowName = "Median Image ";
         imshow(windowName, image);
         waitKey(0);
    }
   

    MPI_Finalize();
 
}

void ImageSummation(string image_path,string image_path2, int argc, char *argv[]){
    int rank, cluster_size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mat Image1, Imaga2;
    int Ihiget, IIhiget , Iwidth , IIwidth , EleSize1 , EleSize2 ,  ImageType1 , ImageType2;
    double start_time, end_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
        Image1 = imread(image_path);
        Imaga2 = imread(image_path2);
        Ihiget = Image1.rows;
        IIhiget = Imaga2.rows;
        Iwidth = Image1.cols;
        IIwidth = Imaga2.cols;
        EleSize1 = Image1.elemSize();
        EleSize2 = Imaga2.elemSize();
        ImageType1 = Image1.type();
        ImageType2 = Imaga2.type();

   
      
        MPI_Bcast(&Ihiget, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&Iwidth, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&IIhiget, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&IIwidth, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&EleSize1, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&EleSize2, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ImageType1, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ImageType2, 1 , MPI_INT, 0, MPI_COMM_WORLD);

        
       
    } else {
        
        MPI_Bcast(&Ihiget, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&Iwidth, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&IIhiget, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&IIwidth, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&EleSize1, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&EleSize2, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ImageType1, 1 , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ImageType2, 1 , MPI_INT, 0, MPI_COMM_WORLD);


    }

    int RowForPro_1 = Ihiget / cluster_size;
    int RowStart_1  = rank * RowForPro_1;
    int RowEnd_1 = (rank == cluster_size - 1)? Ihiget: RowStart_1 + RowForPro_1;
    int totalrows1 = RowEnd_1 - RowStart_1;

    int RowForPro_2 = IIhiget / cluster_size;
    int RowStart_2  = rank * RowForPro_2;
    int RowEnd_2 = (rank == cluster_size - 1)? IIhiget: RowStart_2 + RowForPro_2;
    int totalrows2 = RowEnd_2 - RowStart_2;

    Mat LocalImage1 ( totalrows1 , Iwidth  , ImageType1); 
    Mat LocalImage2 ( totalrows2 , IIwidth , ImageType2); 

    MPI_Scatter(Image1.data,       (totalrows1 * Iwidth  * EleSize1) , MPI_BYTE,
                LocalImage1.data,  (totalrows1 * Iwidth * EleSize1) , MPI_BYTE,
                0, MPI_COMM_WORLD);

    
    MPI_Scatter(Imaga2.data,       (totalrows2 * IIwidth * EleSize2) , MPI_BYTE,
                LocalImage2.data,  (totalrows2 * IIwidth* EleSize2) , MPI_BYTE,
                0, MPI_COMM_WORLD);


    LocalImage1.convertTo(LocalImage1, CV_32F);
    LocalImage2.convertTo(LocalImage2, CV_32F);
    Mat sumImage (Ihiget , Iwidth , CV_64F);

    add(LocalImage1, LocalImage2, sumImage);
     
    Mat RestImage (Ihiget, Iwidth, sumImage.type());
    MPI_Request send_request;

    MPI_Isend(sumImage.data, sumImage.total() * sumImage.elemSize(), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &send_request);

    if (rank == 0) { 
       
        for (int i = 0; i < cluster_size; i++) {
            MPI_Request recv_request;
            MPI_Status recv_status;
            int Row_Start = i * RowForPro_1;
            int Row_End = (i  == cluster_size - 1)? Ihiget : Row_Start + RowForPro_1;    
            int total_rows = Row_End - Row_Start;
         
            cout << " i : " << i << " Row_Start: " << Row_Start << " Row_End: " << Row_End << " total_rows: " << total_rows << endl;
            MPI_Irecv(RestImage.rowRange(Row_Start, Row_End).data, total_rows * Iwidth * RestImage.elemSize() , MPI_BYTE, i, 0, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&recv_request, &recv_status);
        }
            double maxVal;
            minMaxLoc(RestImage, nullptr, &maxVal);

            // normalize
            RestImage = (RestImage / maxVal) * 255; 
        
            RestImage.convertTo(RestImage, CV_8U);

      
    }

    if (rank == 0) {
        end_time = MPI_Wtime();
        cout << "Number of processes: " << cluster_size << " Total execution time: " << end_time - start_time << " seconds" << endl;
         string windowName = "Summation Result: ";
         imshow(windowName, RestImage);
         waitKey(0);
    }
   

    MPI_Finalize();   
}


void adjustImageBrightness(const string& image_path, float alpha, int argc, char** argv) {
    int rank, cluster_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mat image;
    int height, width, type, elemSize;
    double start_time, end_time;
    if (rank == 0) {
        image = imread(image_path);
        start_time = MPI_Wtime();
        cout << "Image rows: " << image.rows << ", Image cols: " << image.cols << endl;

         height = image.rows;
         width = image.cols;
         type = image.type();
         elemSize = image.elemSize();
        MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&elemSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
       
        MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&elemSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        
    }

    int rows_per_process = height / cluster_size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == cluster_size - 1) ? height : start_row + rows_per_process;
    int total_rows = end_row - start_row;

    Mat LocalImage(total_rows, width, type);
    Mat ResultImage(height, width, type);

    MPI_Scatter(image.data, total_rows * width * elemSize, MPI_BYTE,
                LocalImage.data, total_rows * width * elemSize, MPI_BYTE,
                0, MPI_COMM_WORLD);

    LocalImage.convertTo(LocalImage, -1, alpha); // Adjust brightness

    MPI_Request send_request;
    MPI_Isend(LocalImage.data,total_rows * width * elemSize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &send_request);

    if (rank == 0) { 
       
        for (int i = 0; i < cluster_size; i++) {
            MPI_Request recv_request;
            MPI_Status recv_status;
            int RowForProcess = height/ cluster_size;
            int  Row_Start = i * RowForProcess;
            int Row_End = (i  == cluster_size - 1)? height : Row_Start + RowForProcess;    
            int total_rows = Row_End - Row_Start;
         
            cout << " i : " << i << " Row_Start: " << Row_Start << " Row_End: " << Row_End << " total_rows: " << total_rows << endl;
            MPI_Irecv(ResultImage.rowRange(Row_Start, Row_End).data, total_rows * width * elemSize  , MPI_BYTE, i, 0, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&recv_request, &recv_status);
        }
      
    }

   
    if (rank == 0) {
        end_time = MPI_Wtime();
        cout << "Number of processes: " << cluster_size << " Total execution time: " << end_time - start_time << " seconds" << endl;
        string window_name = "Adjusted Image Alpha:" + to_string(alpha) + " - P:" + to_string(cluster_size);
        imshow(window_name, ResultImage);
        waitKey(0);
    }

    MPI_Finalize();
}


void ImageContrast(const string& img, int new_min, int new_max, int argc, char** argv) {
    int rank, cluster_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mat image;
    int height, width, type, elemSize;
    double minVal, maxVal;
    double start_time, end_time;
    if (rank == 0) {
        image = imread(img);
        start_time = MPI_Wtime();
        height = image.rows;
        width = image.cols;
        type = image.type();
        elemSize = image.elemSize();
        minMaxLoc(image, &minVal, &maxVal);
    }

    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&elemSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&minVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rows_per_process = height / cluster_size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == cluster_size - 1) ? height : start_row + rows_per_process;
    int total_rows = end_row - start_row;

    Mat LocalImage(total_rows, width, type);
    Mat ResultImage(height, width, type);

    MPI_Scatter(image.data, total_rows * width * elemSize, MPI_BYTE,
                LocalImage.data, total_rows * width * elemSize, MPI_BYTE,
                0, MPI_COMM_WORLD);

    for (int i = 0; i < LocalImage.rows; i++) {
        for (int j = 0; j < LocalImage.cols; j++) {
            for (int c = 0; c < LocalImage.channels(); c++) {
                int pixel = LocalImage.at<Vec3b>(i, j)[c];
                double new_pixel = ((pixel - minVal) / (maxVal - minVal)) * (new_max - new_min) + new_min;
                new_pixel = max(0.0, min(255.0, new_pixel));
                LocalImage.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(new_pixel);
            }
        }
    }

    
    MPI_Gather(LocalImage.data, rows_per_process * width * elemSize, MPI_BYTE,
               ResultImage.data, rows_per_process * width * elemSize, MPI_BYTE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        end_time = MPI_Wtime();
        cout << "Number of processes: " << cluster_size << " Total execution time: " << end_time - start_time << " seconds" << endl;
        string window_name = "Contrast Image";
        imshow(window_name, ResultImage);
        waitKey(0);
    }

    MPI_Finalize();
}

void HisgtramMatching(const string& img1, const string& img2, int argc, char** argv){

    // image 1 -> Source  -  image 2 -> template 
    int rank, cluster_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    Mat image1, image2;
    int height, width, type, elemSize;

    vector<int> fre1(256, 0), cum1(256, 0) , fre2(256, 0), cum2(256, 0) ,temp1 (256, 0) ,temp2(256, 0), lookup(256, 0);


    Mat image;
    double start_time, end_time;
    if (rank == 0) {
        image1 = imread(img1, IMREAD_GRAYSCALE);
        image2 = imread(img2, IMREAD_GRAYSCALE);
        start_time = MPI_Wtime();
        if (image1.empty() || image2.empty()) {
            cerr << "Error: Could not read the image." << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        height = image1.rows;
        width = image1.cols;
        type = image1.type();
        elemSize = image1.elemSize();
    }

    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&elemSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double ratio = ((double) 255 / (double)(height* width));

    int rows_per_process = height / cluster_size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == cluster_size - 1) ? height : start_row + rows_per_process;
    int total_rows = end_row - start_row;


    Mat LocalImage1(total_rows, width, type);
    Mat LocalImage2(total_rows, width, type);

    MPI_Scatter(image1.data, total_rows * width * elemSize, MPI_BYTE,
                LocalImage1.data, total_rows * width * elemSize, MPI_BYTE,
                0, MPI_COMM_WORLD);

    MPI_Scatter(image2.data, total_rows * width * elemSize, MPI_BYTE, 
                LocalImage2.data, total_rows * width * elemSize, MPI_BYTE, 
                0, MPI_COMM_WORLD);           

 

    // Local Histogram
    for (int i = 0; i < LocalImage2.rows; i++) {
        for (int j = 0; j < LocalImage2.cols; j++) {
            int pixel1 = LocalImage1.at<uchar>(i, j);
            int pixel2 = LocalImage2.at<uchar>(i, j);
            fre1[pixel1]++;
            fre2[pixel2]++;
        }
    }

    if (rank != 0) {
        MPI_Request send_request;
        MPI_Isend(fre1.data(), 256, MPI_INT, 0, 0, MPI_COMM_WORLD, &send_request);
        MPI_Isend(fre2.data(), 256, MPI_INT, 0, 0, MPI_COMM_WORLD, &send_request);

    }else{

        for (int i = 1; i < cluster_size; i++) {

            MPI_Request recv_request;
            MPI_Status recv_status;
            MPI_Irecv(temp1.data(), 256, MPI_INT, i, 0, MPI_COMM_WORLD, &recv_request);   MPI_Wait(&recv_request, &recv_status);
            MPI_Irecv(temp2.data(), 256, MPI_INT, i, 0, MPI_COMM_WORLD, &recv_request);   MPI_Wait(&recv_request, &recv_status);

            // Golobal Histogram
            for (int j = 0; j < 256; j++) {
                fre1[j] += temp1[j];
                fre2[j] += temp2[j];
            }
        }

        cum1[0] = fre1[0];
        cum2[0] = fre2[0];
        for (int i = 1; i < 256; i++) {
            cum1[i] = cum1[i - 1] + fre1[i];
            cum2[i] = cum2[i - 1] + fre2[i];
        }

       for(int i=0;i<255;i++){
        cout << "cum1[" << i << "] = " << cum1[i] << " cum2[" << i << "] = " << cum2[i] << endl;
       }
    }

    MPI_Bcast(cum1.data(), 256, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cum2.data(), 256, MPI_INT, 0, MPI_COMM_WORLD);
    
    fre1.assign(256, 0);
    fre2.assign(256, 0);

    int pixel1, pixel2, new_pixel1, new_pixel2;
    for (int i = 0; i < LocalImage2.rows; i++) {
        for (int j = 0; j < LocalImage2.cols; j++) {
            pixel1 = LocalImage1.at<uchar>(i, j);
            pixel2 = LocalImage2.at<uchar>(i, j);

            LocalImage1.at<uchar>(i, j) = cum1[pixel1] * ratio;
            LocalImage2.at<uchar>(i, j) = cum2[pixel2] * ratio;

            // New Histogram after image equalization

            new_pixel1 = LocalImage1.at<uchar>(i, j);
            new_pixel2 = LocalImage2.at<uchar>(i, j);

            fre1[new_pixel1]++;
            fre2[new_pixel2]++;
        }
    }
    

    if (rank != 0) {
        MPI_Request send_request;
        MPI_Isend(fre1.data(), 256, MPI_INT, 0, 0, MPI_COMM_WORLD, &send_request);
        MPI_Isend(fre2.data(), 256, MPI_INT, 0, 0, MPI_COMM_WORLD, &send_request);

    } else {

        cout << " After Histogram Elasction" << endl;
        for (int i = 1; i < cluster_size; i++) {

            MPI_Request recv_request;
            MPI_Status recv_status;
            MPI_Irecv(temp1.data(), 256, MPI_INT, i, 0, MPI_COMM_WORLD, &recv_request);   MPI_Wait(&recv_request, &recv_status);
            MPI_Irecv(temp2.data(), 256, MPI_INT, i, 0, MPI_COMM_WORLD, &recv_request);   MPI_Wait(&recv_request, &recv_status);

            // New Golobal Histogram
            for (int j = 0; j < 256; j++) {
                fre1[j] += temp1[j];
                fre2[j] += temp2[j];
            }   
        }

        // Normalize  histograms
        for (int i = 0; i < 256; i++) {
            long double total = height * width;
            long double tempfreq =  ((long double) fre1[i] ) / (total);
            long double tempfreq2 = ((long double) fre2[i] ) / (total);

            fre1[i] = int(tempfreq   * 255);
            fre2[i] = int(tempfreq2  * 255);
            cout << "fre1[" << i << "] = " << fre1[i] << " fre2[" << i << "] = " << fre2[i] << endl;
           
        }
     
        cum1[0] = fre1[0];
        cum2[0] = fre2[0];
        for (int i = 1; i < 256; i++) {
            cum1[i] = cum1[i - 1] + fre1[i];
            cum2[i] = cum2[i - 1] + fre2[i];
        }

       // Normalize the cumulative histograms
        for (int i = 0; i < 256; i++) {
            long double total1 =  ((long double)cum1[i] / cum1[255]);
            long double total2 =  ((long double)cum2[i] / cum2[255]);
            cum1[i] = int(total1 * 255);
            cum2[i] = int(total2 * 255);
            cout << " T1: "<< total1 << " T2: " << total2 << " cum1[" << i << "] = " << cum1[i] << " cum2[" << i << "] = " << cum2[i] << endl;
        }

        for (int i = 0 ; i < 256 ; ++i){
            int j = 255;
            while (cum2[j] > cum1[i] && j > 0) {
                j--;
                lookup[i] = j;
            }
        } 
    }
    
    MPI_Bcast(lookup.data(), 256, MPI_INT, 0, MPI_COMM_WORLD);   

    for (int i = 0; i < LocalImage1.rows; i++) {
        for (int j = 0; j < LocalImage1.cols; j++) {
            int pixel = LocalImage1.at<uchar>(i, j);
            LocalImage1.at<uchar>(i, j) = lookup[pixel];
        }
    }

    Mat ResultImage(height, width, LocalImage1.type());

    MPI_Gather(LocalImage1.data, rows_per_process * width * elemSize, MPI_BYTE,
               ResultImage.data, rows_per_process * width * elemSize, MPI_BYTE,
               0, MPI_COMM_WORLD);


    if (rank == 0) {
        end_time = MPI_Wtime();
        cout << "Number of processes: " << cluster_size << " Total execution time: " << end_time - start_time << " seconds" << endl;

        string window_name = "Histogram Matching";
        imshow("Source", image1);
        imshow("Template", image2);
        imshow(window_name, ResultImage);
        waitKey(0);
    }
   
 MPI_Finalize();
   
}