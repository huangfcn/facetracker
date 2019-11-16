# Description

	This project is very similar to my another project multiple-object-tracking, the only difference 
	is a general YOLO3 detector in that project replaced by the MTCNN model (detecting faces only). 

	Face detection and tracking is one of the most widely used application of deep-learning. 
	Face detection model only detects faces visible on current frame. To track obstacle faces and moving
	faces, we need to solve a multiple-object-tracking problem. This could be done in two steps:
  
	1, Detecting moving faces in each frame
	2, Tracking historical faces with some tracking algorithms
  
	An assignment problem is used to associate the faces detected by detectors and tracked by trackers.
  
	We can found some introduction of this framework here,
	https://towardsdatascience.com/computer-vision-for-tracking-8220759eee85
  
	Another example in more detail with matlab code (detecors and trackers may different),
	https://www.mathworks.com/help/vision/examples/motion-based-multiple-object-tracking.html

	Here I implemented a highly efficient and scalable C++ framework to combine the state of art 
	deep-learning based face detectors (MTCNN here) and correlation filters based trackers 
	(KCF, Kalman Filters also implemented). The assignment problem is solved by hungarian algorithm.
  
# Detectors: MTCNN 

	MTCNN is a specific model used to detect face. This model is implemted using the technology described 
	in my another project 'dnnsimd' and running purely on CPU with SIMD acceleration (auto-vectorization).

# Trackers: Kalman Filter and KCF

	Kalman filter is fast but less accurate. KCF is accurate but much slower. 
	They are implemnted with exactly same interface, so we can easily switch from one to another 
	in the project.

# Live Camera Capture: OpenCV

	OpenCV is used to capture live video frames and used for image preprocessing.

# Misc
	
	1, Only x64/Release environment variables are properly set
	2, MTCNN supporting library (conv.c) can only be compiled by gcc to enable SIMD acceleration, 
	   a Visual Studio compatible conv.o is provided and compiled by gcc with '-O3 -march=native'.