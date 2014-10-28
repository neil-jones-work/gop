#include "gop.hpp"


GeodesicObjectProposal::GeodesicObjectProposal(){

    pthread_mutex_init(&this->mutex, NULL);
    this->nSeeds = 200;
    this->nSegmentsPerSeed = 10;

    prop_settings.max_iou = 0.8;
    // Foreground/background proposals
    std::vector<int> vbg = {0,15};
    prop_settings.unaries.push_back( ProposalSettings::UnarySettings( this->nSeeds, this->nSegmentsPerSeed, seedUnary(), backgroundUnary(vbg) ) );
    // Pure background proposals
    std::vector<int> allbg = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    prop_settings.unaries.push_back( ProposalSettings::UnarySettings( 0, this->nSegmentsPerSeed, zeroUnary(), backgroundUnary(allbg), 0.1, 1  ) );

}

void GeodesicObjectProposal::loadDetector(const char *filename){
    detector.load(filename);
}

void GeodesicObjectProposal::getSegments(const cv::Mat src, std::vector<cv::Mat> &segments, const int top_segments){
    /* Create the proposlas */
    Proposal prop( prop_settings );

    // swap BGR2RGB
    cv::Mat rgb;
    cv::cvtColor(src, rgb, CV_BGR2RGB);

    Image8u im(rgb.data, rgb.cols, rgb.rows, rgb.channels());

    pthread_mutex_lock(&this->mutex);
    static int nIters = 10;
    // Create an over-segmentation
    std::shared_ptr<ImageOverSegmentation> s = geodesicKMeans( im, detector, top_segments,  nIters);
    RMatrixXb p = prop.propose( *s );

    // If you just want boxes use
//    RMatrixXi boxes = s->maskToBox( p );

    // To use the proposals use the over segmentation s.s() and p.row(n)
    // you can get the binary segmentation mask using the following lines
    segments.reserve(top_segments);

    for (int idx = 0; idx < p.rows(); ++idx){

        cv::Mat segment(src.rows, src.cols, CV_8UC1);
#pragma omp parallel for
        for( int j=0; j<s->s().rows(); ++j ){
            unsigned char *ptr = segment.ptr<unsigned char>(j);
            for( int i=0; i<s->s().cols(); ++i )
                 ptr[i] = p( idx, s->s()(j,i) );
        }
        segments.push_back(segment);
    }
    segments.resize(segments.size());

    pthread_mutex_unlock(&this->mutex);
}
