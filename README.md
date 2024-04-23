# POINT-SLAM: A Fusion of ORB-SLAM3 and SuperPoint

POINT-SLAM is a fusion of ORB-SLAM3 with SuperPoint feature detection. SuperPoint feature detection was made as a neural network based method to detect keypoints on images. We integrated this with ORB-SLAM3 in order to investigate if SuperPoint's keypoints would make the entire ORB-SLAM3 system more robust.

Euroc MH05 Results:
![Results of ground truth, ORB-SLAM3, and PointSLAM overlayed](https://github.com/amons02/POINT-SLAM/blob/main/MH05_GT_Comparison.png?raw=true)

## Building:

All systems were built and tested on Ubuntu 22.04 in WSL.

1. Install all ORB-SLAM3 dependencies. This can be done by navigating to the [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) repository and following all instructions under prerequisites.
2. Install all SuperGlue dependencies. This can be done by navigating to the [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) repository and following all instructions under dependencies.
3. If on linux, run "chmod +x build.sh".
4. Run build.sh

## Credit:

This project was created by Anirudh Attaluri, Yash Dixit, Kartikeya Gupta, and Allen Mons.

Thank you to all creators and contributors for the projects we built this on. Special credit to UZ-SLAMLab for ORB-SLAM3, MagicLeap for SuperGlue, aPR0T0 for recent ORB-SLAM3 build fixes, and MichaelGrupp for evo which was used to evaluate our performance.
