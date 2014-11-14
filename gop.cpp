#include "gop.hpp"




GeodesicObjectProposal::GeodesicObjectProposal(){

    pthread_mutex_init(&this->mutex, NULL);
    this->nSeeds = 200; // N_S
    this->nSegmentsPerSeed = 10; // N_T

    this->prop_settings.max_iou = 0.9;

    bool learn = true;
    if (!learn){
        // Foreground/background proposals
        std::vector<int> vbg = {0,15};
        this->prop_settings.unaries.push_back( ProposalSettings::UnarySettings( this->nSeeds, this->nSegmentsPerSeed, seedUnary(), backgroundUnary(vbg) ) );
        // Pure background proposals
        std::vector<int> allbg = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        this->prop_settings.unaries.push_back( ProposalSettings::UnarySettings( 0, this->nSegmentsPerSeed, zeroUnary(), backgroundUnary(allbg), 0.1, 1  ) );
    }
    else{

        // Load the seed function
        std::shared_ptr<LearnedSeed> seed = std::make_shared<LearnedSeed>();

        seed->load( "/usr/local/share/we-cv-sdk/pretrained_model/gop/seed_final.dat" );
        prop_settings.foreground_seeds = seed;

        // Load the foreground/background proposals
        for( int i=0; i<3; i++ ){
            prop_settings.unaries.push_back( ProposalSettings::UnarySettings( this->nSeeds,
                                                                              this->nSegmentsPerSeed,
                                                                              binaryLearnedUnary(("/usr/local/share/we-cv-sdk/pretrained_model/gop/masks_final_" + std::to_string(i)+ "_fg.dat").c_str()),
                                                                              binaryLearnedUnary(("/usr/local/share/we-cv-sdk/pretrained_model/gop/masks_final_" + std::to_string(i)+ "_bg.dat").c_str()) ) );
        }

        // Pure background proposals
        std::vector<int> allbg = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        prop_settings.unaries.push_back( ProposalSettings::UnarySettings( 0, this->nSegmentsPerSeed, zeroUnary(), backgroundUnary(allbg), 0.1, 1  ) );
    }
    std::cout<<"asfa"<<std::endl;
    std::cout.flush();
}

void GeodesicObjectProposal::loadDetector(const char *filename){
    detector.load(filename);
}

void GeodesicObjectProposal::loadSeeds(const char *filename){
    // Load the seed function
    std::shared_ptr<LearnedSeed> seed = std::make_shared<LearnedSeed>();
    seed->load( filename );
    this->prop_settings.foreground_seeds = seed;
}

void GeodesicObjectProposal::loadMask(const char *fg_mask_filename, const char *bg_mask_filename){
    this->prop_settings.unaries.push_back( ProposalSettings::UnarySettings( this->nSeeds, this->nSegmentsPerSeed, binaryLearnedUnary(fg_mask_filename), binaryLearnedUnary(bg_mask_filename) ) );
}


void GeodesicObjectProposal::setPureBackgroundProposals(){
    // Pure background proposals
    std::vector<int> allbg = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    this->prop_settings.unaries.push_back( ProposalSettings::UnarySettings( 0, this->nSegmentsPerSeed, zeroUnary(), backgroundUnary(allbg), 0.1, 1  ) );
}

void GeodesicObjectProposal::getSegments(const cv::Mat src, std::vector<cv::Mat> &segments, const int nSuperPixels, const int top_segments){


    /* Create the proposlas */
    Proposal prop( this->prop_settings );

    // swap BGR2RGB
    cv::Mat rgb;
    cv::cvtColor(src, rgb, CV_BGR2RGB);

    Image8u im(rgb.cols, rgb.rows, rgb.channels()); // we should copy the image data
    memcpy(im.data(), rgb.data, sizeof(unsigned char) *3 * rgb.rows * rgb.cols);
    pthread_mutex_lock(&this->mutex);

    static int nIters = 10;
    // Create an over-segmentation
    std::shared_ptr<ImageOverSegmentation> s = geodesicKMeans( im, detector, nSuperPixels,  nIters);

    RMatrixXb p = prop.propose( *s );

    segments.reserve(top_segments);
    for (int idx = 0; idx < std::min(top_segments, (int)p.rows()); ++idx){
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




void GeodesicObjectProposal::apply(const cv::Mat src, std::vector<cv::Mat> &segments, const int nSuperPixels, const int top_segments){

    /* Create the proposlas */
    Proposal prop( this->prop_settings );

    // swap BGR2RGB
    cv::Mat rgb;
    cv::cvtColor(src, rgb, CV_BGR2RGB);

    Image8u im(rgb.cols, rgb.rows, rgb.channels()); // we should copy the image data
    memcpy(im.data(), rgb.data, sizeof(unsigned char) *3 * rgb.rows * rgb.cols);
    pthread_mutex_lock(&this->mutex);

    static int nIters = 10;
    // Create an over-segmentation
    std::shared_ptr<ImageOverSegmentation> s = geodesicKMeans( im, detector, nSuperPixels,  nIters);

    RMatrixXb p = prop.propose( *s );

    segments.reserve(top_segments);
    for (int idx = 0; idx < std::min(top_segments, (int)p.rows()); ++idx){
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
