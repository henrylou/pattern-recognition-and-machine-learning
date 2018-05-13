# Project 2: Human Face Detection using Boosting

### 1. Objective
Boosting is a general method for improving the accuracy of any given learning algo- rithm. Specifically, one can use it to combine weak learners, each performing only slightly better than random guess, to form an arbitrarily good hypothesis. In this project, you are required to implement an AdaBoost and RealBoost algorithms for frontal human face detection.

### 2. Data
• Training data: Face and non-face images of the size of 16x16 pixels are given.   
• Testing data: Three photos taken at the class are used for testing.  
• Hard negatives: Three background images are taken without faces.  

### 3. Tasks  
1. Construct weak classifiers: Load the predefined set of Haar filters. Compute the features by applying each Haar filter to the integral images of the positive and negative populations. Determine the polarity and threshold classifier with lowest weighted error. Note, as the samples change their weights over time, the histograms and threshold θ will change.  
2. Implement AdaBoost: Implement the AdaBoost algorithm to boost the weak classifiers. Construct the strong classifier as an weighted ensemble of T weak classifiers. Perform non-maximum suppression and hard negatives mining.     
3. Implement RealBoost: Implement the RealBoost algorithm using the top T = 10, 50, 100 features. Compute the histograms of negative and positive populations and the corresponding ROC curves. 
