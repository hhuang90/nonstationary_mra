#pragma once

#include "class_partition.hpp"

using namespace arma;

class Approximation
/*
Declare the class of Data, which contains all the information about the data
*/
{
    public:

        //Parameters in the stationary covariance function
        double sill, range, nugget;

        //Parameters in the nonstatioary covariance function
        double sill_scale, range_scale, nugget_scale;

        //log-likelihood;
        double loglikelihood;

        //Cholesky factor of covariance matrix of knots in each level, dimension: the total number of regions
        mat *RChol;

        //w tilde, dimension: the total number of regions
        mat *wTilde;

        //A tilde, dimension: the total number of regions
        cube *ATilde;

        //KCholTimesCurrentw, dimension: the total number of regions
        vec *KCholTimesCurrentw;

        //KCholTimesCurrentA, dimension: the total number of regions
        cube *KCholTimesCurrentA;

        //BTilde, dimension: [the number of regions in the finest level]*[the number of levels-1]*[matrix size]
        double ***BTilde;

        //Posterior mean
        vec *posteriorPredictionMean;
        vec *posteriorPredictionVariance;

        //Constructor function for assigning sill and range
        Approximation(const double &sill, const double &range, const double &nugget,
                      const double &sill_scale, const double &range_scale, const double &nugget_scale);

        //Destructor
        ~Approximation();

        //Calculate prior and posterior quantities
        void likelihood();

        //Make predictions
        void predict();
        
        //Dump prediction results
        void dump_prediction_result();

    private:
        void create_prior();

        void posterior_inference();

        void prior_allocate_memory();

        void get_all_ancestors(unsigned long* ancestorArray, unsigned long currentRegion, const unsigned long &nAncestors);

        void get_conditional_covariance_matrix(const int &ancestorLevel, double **ancestorNow, double **ancestorBefore, double* nowBefore, const int &numKnotsNow, const int &numKnotsBefore, const int &numKnotsAncestor);

        void get_wTilde_and_ATilde_in_prior(const int &nKnots, const int &currentLevel, const unsigned long &iRegion, vec *Sicy, mat **SicB);

        void loop_regions_before_finest_level_in_prior(double ***KCholTimesw);

        void loop_regions_at_finest_level_in_prior(double ***KCholTimesw);

        void aggregate_A(mat &A, double *tempMemory, const unsigned long &indexA, const unsigned long &indexStart, const unsigned long &indexEnd, unsigned long iRegion, bool &supervisor, bool &synchronizeFlag);

        void aggregate_w_and_A(vec &w, mat &A, double *tempMemory, const int &jLevel, const unsigned long &indexA, const unsigned long &indexStart, const unsigned long &indexEnd, unsigned long iRegion, bool &supervisor, bool &synchronizeFlag);

        void get_wTilde_and_ATilde_in_posterior(vec &w, mat &A, double *tempMemory, const unsigned long &nKnots, cube &KCholTimesCurrentA, const vec &KCholTimesCurrentw, const unsigned long &iRegion, const int &currentLevel, bool &supervisor, bool &synchronizeFlag);

        void synchronizeIndicesRegionsForEachWorker();

        void load_ATilde_from_disk(const unsigned long &indexChildrenStart, const unsigned long &indexChildrenEnd, bool secondLastLevel, unsigned long* nKnotsAtFinestLevel, unsigned long offset, unsigned long size1, unsigned long size2);

        void free_wTilde_and_ATilde(const unsigned long &indexStartFreeingATilde, const unsigned long &indexEndFreeingATilde);

        template<typename T> void receive(T &object, double *tempMemory, int tag, unsigned long iRegion);

        template<typename T> void send(const T &object, int tag, unsigned long iRegion);
};