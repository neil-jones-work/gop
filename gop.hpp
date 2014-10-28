#ifndef __GOP_H__
#define __GOP_H__


#include <opencv2/opencv.hpp>
#include <pthread.h>
#include "contour/structuredforest.h"
#include "proposals/proposal.h"

class GeodesicObjectProposal{
public:
    GeodesicObjectProposal();

    static GeodesicObjectProposal &getInstance(){
        static GeodesicObjectProposal instance;
        return instance;
    }

    void loadDetector(const char *filename);

    void getSegments(const cv::Mat src, std::vector<cv::Mat> &segments, const int top_segments = 500);

private:

    MultiScaleStructuredForest detector;
    ProposalSettings prop_settings;
    pthread_mutex_t mutex;
    int nSeeds, nSegmentsPerSeed;

};

#endif
