#include <iostream>
#include <armadillo>
#include <mpi.h>
#include <omp.h>
#include <mkl.h>

#include "class_data.hpp"
#include "class_nonstationary_covariance.hpp"
#include "class_approximation.hpp"
#include "class_partition.hpp"
#include "evaluate_covariance.hpp"
#include "constants.hpp"
#include <string.h>

void Approximation::loop_regions_at_finest_level_in_prior(double ***KCholTimesw)
{
    int maxOpenMPThreads=omp_get_max_threads();
    int currentLevel = NUM_LEVELS_M - 1;
    double loglikelihoodThisLevel = 0;
    
    unsigned long maxNumKnotsAtFinestLevel = 0;
    //Find the maximum number of knots at the finest level
    #pragma omp parallel for reduction(max:maxNumKnotsAtFinestLevel) schedule(dynamic,1)
    for(unsigned long iRegion = partition->nRegionsInTotal-partition->nRegionsAtEachLevel[NUM_LEVELS_M-1]; iRegion < partition->nRegionsInTotal; iRegion++)
    {
        if( !WORKING_REGION_FLAG[iRegion] ) continue;
        maxNumKnotsAtFinestLevel =  std::max(maxNumKnotsAtFinestLevel, partition->nKnotsAtFinestLevel[iRegion-partition->nRegionsInTotal+partition->nRegionsAtEachLevel[NUM_LEVELS_M-1]]);
    }

    /////// Define and allocate variables

    //Allocate memory for ATilde
    if(!SAVE_TO_DISK_FLAG)
        for(unsigned long iRegion = partition->indexStartFinestLevel; iRegion < partition->nRegionsInTotal; iRegion++)
            ATilde[iRegion].set_size(partition->nKnots, partition->nKnots, (NUM_LEVELS_M-1)*NUM_LEVELS_M/2);
    
    //Allocate memory for wTilde
    for(unsigned long iRegion = partition->indexStartFinestLevel; iRegion < partition->nRegionsInTotal; iRegion++)
        wTilde[iRegion].set_size(partition->nKnots,NUM_LEVELS_M-1);


    //Define indices of all the ancestors for the current region
    unsigned long **ancestorArray = new unsigned long* [maxOpenMPThreads];
    for(int i = 0; i < maxOpenMPThreads; i++) ancestorArray[i] = new unsigned long [currentLevel];

    //Auxilliary variables for matrix computations
    int one=1;
    char L='L', N='N';
    int nKnots = partition->nKnots;

    //Loop the regions 
    #pragma omp parallel for reduction(+:loglikelihoodThisLevel) num_threads(maxOpenMPThreads) schedule(dynamic,1)
    for(unsigned long iRegion = REGION_START[NUM_LEVELS_M-1]; iRegion < REGION_END[NUM_LEVELS_M-1] + 1; iRegion++)
    {
        int status;
        int rankOpenMP = omp_get_thread_num();
        unsigned long indexRegionAtThisLevel = iRegion-partition->nRegionsInTotal+partition->nRegionsAtEachLevel[NUM_LEVELS_M-1];

        mat **SicB;
        vec *Sicy;
        double *RCholThisLevel;
        double **covObsKnots;
        int nObsThisRegion;

        //Assign the number of knots in the current region accordingly
        int nKnotsInCurrentRegion;
        if(OBS_AS_KNOTS_FLAG)
            nKnotsInCurrentRegion = partition->nKnotsAtFinestLevel[indexRegionAtThisLevel];
        else
        {
            nKnotsInCurrentRegion = NUM_KNOTS_FINEST;
            nObsThisRegion = partition->nObservationsAtFinestLevel[indexRegionAtThisLevel];
        }

        if( (nKnotsInCurrentRegion == 0 && OBS_AS_KNOTS_FLAG) || (nObsThisRegion == 0 && !OBS_AS_KNOTS_FLAG)  ) 
        {
            ATilde[iRegion].zeros();
            wTilde[iRegion].zeros();
            if(CALCULATION_MODE=="prediction")
            {
                if(VARIOUS_J_FINEST_LEVEL_FLAG)
                {
                    ancestorArray[rankOpenMP][currentLevel-1] = partition->parent[iRegion - partition->indexStartFinestLevel];
                    get_all_ancestors(ancestorArray[rankOpenMP], ancestorArray[rankOpenMP][currentLevel-1], currentLevel-1);
                }
                else
                    get_all_ancestors(ancestorArray[rankOpenMP], iRegion, currentLevel);
            }
        }
        else
        {
            //Find the indices of all the ancestors
            if(VARIOUS_J_FINEST_LEVEL_FLAG)
            {
                ancestorArray[rankOpenMP][currentLevel-1] = partition->parent[iRegion - partition->indexStartFinestLevel];
                get_all_ancestors(ancestorArray[rankOpenMP], ancestorArray[rankOpenMP][currentLevel-1], currentLevel-1);
            }
            else
                get_all_ancestors(ancestorArray[rankOpenMP], iRegion, currentLevel);

            //Allocate memory for variables in the current region
            mat **covarianceMatrix = new mat* [currentLevel+1];

            double **covarianceMatrixMemory = new double* [currentLevel+1];
            for(int jLevel = 0; jLevel < currentLevel; jLevel++)
                covarianceMatrixMemory[jLevel] = new double [nKnotsInCurrentRegion*partition->nKnots];
            covarianceMatrixMemory[currentLevel] =  new double [nKnotsInCurrentRegion*nKnotsInCurrentRegion];

            SicB = new mat* [currentLevel];
            for(int jLevel = 0; jLevel < currentLevel; jLevel++)
                SicB[jLevel] = new mat(partition->nKnots,nKnotsInCurrentRegion);

            if(CALCULATION_MODE=="prediction") KCholTimesw[iRegion] = new double* [currentLevel];

            //Loop for all ancestors to get the cross-covariance matrix between knots 
            for(int jLevel = 0; jLevel < currentLevel; jLevel++)
            {
                //Calculate the covariances between current region and the ancestors
                covarianceMatrix[jLevel] =  new mat(covarianceMatrixMemory[jLevel],nKnotsInCurrentRegion,nKnots,false,true);

                if(NONSTATIONARY_FLAG)
                    evaluate_nonstationary_cross_covariance(covarianceMatrixMemory[jLevel],
                    partition->knotsLon[iRegion], partition->knotsLat[iRegion],
                    partition->knotsX[iRegion], partition->knotsY[iRegion], partition->knotsZ[iRegion],
                    partition->knotsLon[ancestorArray[rankOpenMP][jLevel]], partition->knotsLat[ancestorArray[rankOpenMP][jLevel]],
                    partition->knotsX[ancestorArray[rankOpenMP][jLevel]], partition->knotsY[ancestorArray[rankOpenMP][jLevel]], partition->knotsZ[ancestorArray[rankOpenMP][jLevel]],
                    nKnotsInCurrentRegion, nKnots, sill_scale, range_scale, nugget_scale);
                else
                    evaluate_cross_covariance(covarianceMatrixMemory[jLevel], 
                    partition->knotsLon[iRegion], partition->knotsLat[iRegion],
                    partition->knotsX[iRegion], partition->knotsY[iRegion], partition->knotsZ[iRegion],
                    partition->knotsLon[ancestorArray[rankOpenMP][jLevel]], partition->knotsLat[ancestorArray[rankOpenMP][jLevel]],
                    partition->knotsX[ancestorArray[rankOpenMP][jLevel]], partition->knotsY[ancestorArray[rankOpenMP][jLevel]], partition->knotsZ[ancestorArray[rankOpenMP][jLevel]],
                    nKnotsInCurrentRegion, nKnots, sill, range, nugget);
            }

            //Get the current variance-covariance matrix
            covarianceMatrix[currentLevel] =  new mat(covarianceMatrixMemory[currentLevel],nKnotsInCurrentRegion,nKnotsInCurrentRegion,false,true);

            if(NONSTATIONARY_FLAG)
                evaluate_nonstationary_variance_covariance(covarianceMatrixMemory[currentLevel],
                partition->knotsLon[iRegion], partition->knotsLat[iRegion], 
                partition->knotsX[iRegion], partition->knotsY[iRegion], partition->knotsZ[iRegion], 
                nKnotsInCurrentRegion, sill_scale, range_scale, nugget_scale);
            else
                evaluate_variance_covariance(covarianceMatrixMemory[currentLevel],
                partition->knotsLon[iRegion], partition->knotsLat[iRegion], 
                partition->knotsX[iRegion], partition->knotsY[iRegion], partition->knotsZ[iRegion], 
                nKnotsInCurrentRegion, sill, range, nugget);
            
            //Loop for all ancestors to get the conditional cross-covariance matrix
            for(int jLevel = 0; jLevel < currentLevel; jLevel++)
            {
                get_conditional_covariance_matrix(jLevel, covarianceMatrixMemory, KCholTimesw[ancestorArray[rankOpenMP][jLevel]], covarianceMatrixMemory[jLevel], nKnotsInCurrentRegion,nKnots,nKnots);
                
                //Copy covarianceMatrix[jLevel] to SicB[jLevel] for later use
                memcpy(SicB[jLevel]->memptr(),covarianceMatrixMemory[jLevel],nKnotsInCurrentRegion*nKnots*sizeof(double));
                
                //Reassign covarianceMatrix[rankOpenMP][jLevel] = RChol[ancestorArray[rankOpenMP][jLevel]]^(-1)*covarianceMatrix[rankOpenMP][jLevel]
                dtrtrs_(&L,&N,&N,&nKnots,&nKnotsInCurrentRegion,RChol[ancestorArray[rankOpenMP][jLevel]].memptr(),&nKnots,covarianceMatrixMemory[jLevel],&nKnots,&status);

                if(CALCULATION_MODE=="prediction" || !OBS_AS_KNOTS_FLAG) KCholTimesw[iRegion][jLevel]=covarianceMatrixMemory[jLevel];
            }
            //Get the conditional variance-covariance matrix of this region
            get_conditional_covariance_matrix(currentLevel, covarianceMatrixMemory, covarianceMatrixMemory, covarianceMatrixMemory[currentLevel], nKnotsInCurrentRegion,nKnotsInCurrentRegion,nKnots);      
            
            //Cholesky factorization, covarianceMatrix[rankOpenMP][currentLevel] = RCholThisLevel*RCholThisLevel^T
            RCholThisLevel = covarianceMatrixMemory[currentLevel];
            dpotrf_(&L,&nKnotsInCurrentRegion,RCholThisLevel,&nKnotsInCurrentRegion,&status);   

            //Allocate and compute covObsKnots if Observations are not used as knots
            // if(!OBS_AS_KNOTS_FLAG)
            // {
            //     covObsKnots = new double* [NUM_LEVELS_M];

            //     for(int jLevel = 0; jLevel < currentLevel; jLevel++)
            //         covObsKnots[jLevel] = new double [nKnots * partition->nObservationsAtFinestLevel[indexRegionAtThisLevel]];
            //     covObsKnots[currentLevel] = new double [nKnotsInCurrentRegion * partition->nObservationsAtFinestLevel[indexRegionAtThisLevel]]

            //     //Loop for all ancestors to get the cross-covariance matrix between observations and knots 

            //     if(NONSTATIONARY_FLAG)
            //     {
            //         for(int jLevel = 0; jLevel < currentLevel; jLevel++)
            //             evaluate_nonstationary_cross_covariance(covObsKnots[jLevel],
            //                 partition->observationLon[iRegion], partition->observationLat[iRegion],
            //                 partition->observationX[iRegion], partition->observationY[iRegion], partition->observationZ[iRegion],
            //                 partition->knotsLon[ancestorArray[rankOpenMP][jLevel]], partition->knotsLat[ancestorArray[rankOpenMP][jLevel]],
            //                 partition->knotsX[ancestorArray[rankOpenMP][jLevel]], partition->knotsY[ancestorArray[rankOpenMP][jLevel]], partition->knotsZ[ancestorArray[rankOpenMP][jLevel]],
            //                 nObsThisRegion, nKnots, sill_scale, range_scale, nugget_scale);
                    
            //         jLevel = currentLevel;
            //         evaluate_nonstationary_cross_covariance(covObsKnots[jLevel],
            //             partition->observationLon[iRegion], partition->observationLat[iRegion],
            //             partition->observationX[iRegion], partition->observationY[iRegion], partition->observationZ[iRegion],
            //             partition->knotsLon[iRegion], partition->knotsLat[iRegion],
            //             partition->knotsX[iRegion], partition->knotsY[iRegion], partition->knotsZ[iRegion],
            //             nObsThisRegion, nKnotsInCurrentRegion, sill_scale, range_scale, nugget_scale);
            //     }
            //     else
            //     {
            //         for(int jLevel = 0; jLevel < currentLevel; jLevel++)
            //             evaluate_cross_covariance(covObsKnots[jLevel], partition->observationLon[iRegion], partition->observationLat[iRegion], partition->knotsLon[ancestorArray[rankOpenMP][jLevel]], partition->knotsLat[ancestorArray[rankOpenMP][jLevel]], nObsThisRegion, nKnots, sill, range, nugget);

            //         jLevel = currentLevel;
            //         evaluate_cross_covariance(covObsKnots[jLevel], partition->observationLon[iRegion], partition->observationLat[iRegion], partition->knotsLon[iRegion], partition->knotsLat[iRegion], nObsThisRegion, nKnotsInCurrentRegion, sill, range, nugget);
            //     }

                
            //     //Loop for all ancestors to get the conditional cross-covariance matrix
            //     for(int jLevel = 0; jLevel < currentLevel; jLevel++)
            //     {
            //         get_conditional_covariance_matrix(jLevel, covObsKnots, KCholTimesw[ancestorArray[rankOpenMP][jLevel]], covObsKnots[jLevel], nObsThisRegion, nKnots,nKnots);
                
            //         dtrtrs_(&L,&N,&N,&nKnots,&nObsThisRegion,RChol[ancestorArray[rankOpenMP][jLevel]].memptr(),&nKnots,covObsKnots[jLevel],&nKnots,&status);
            //     }

            //     jLevel = currentLevel;
            //     get_conditional_covariance_matrix(jLevel, covObsKnots, KCholTimesw[iRegion], covObsKnots[jLevel], nObsThisRegion, nKnotsInCurrentRegion, nKnots);
                
            //     //covObsKnots = var(knots|knots before)^{-1/2} * cov(obs,knots | knots before)
            //     dtrtrs_(&L,&N,&N,&nKnotsInCurrentRegion,&nObsThisRegion,RCholThisLevel,&nKnotsInCurrentRegion,covObsKnots[jLevel],&nKnotsInCurrentRegion,&status);

            // }

            //Compute Sicy = RCholThisLevel^(-1)*partition->knotsResidual[indexRegionAtThisLevel]
            if(OBS_AS_KNOTS_FLAG)
            {
                Sicy = new vec(partition->knotsResidual[indexRegionAtThisLevel],nKnotsInCurrentRegion,true,true);
                dtrtrs_(&L,&N,&N,&nKnotsInCurrentRegion,&one,RCholThisLevel,&nKnotsInCurrentRegion,Sicy->memptr(),&nKnotsInCurrentRegion,&status);
            }
            else
            {
                // Sicy = new vec(partition->residual[indexRegionAtThisLevel],nObsThisRegion,true,true);
            }

            //Compute SicB[jLevel] = RCholThisLevel^(-1)*covarianceMatrix[jLevel]
            for(int jLevel = 0; jLevel < currentLevel; jLevel++)
            {      
                inplace_trans(*SicB[jLevel]);
                dtrtrs_(&L,&N,&N,&nKnotsInCurrentRegion,&nKnots,RCholThisLevel,&nKnotsInCurrentRegion,SicB[jLevel]->memptr(),&nKnotsInCurrentRegion,&status);
            }

            get_wTilde_and_ATilde_in_prior(nKnots, currentLevel, iRegion, Sicy, SicB);

            if(CALCULATION_MODE!="prediction")
            {
                double loglikelihoodToAdd = 0; int pos = 0;
                for(int i = 0; i < nKnotsInCurrentRegion; i++)
                {
                    loglikelihoodToAdd += (*Sicy)[i]*(*Sicy)[i] + 2*log(RCholThisLevel[pos]);
                    pos += nKnotsInCurrentRegion + 1;
                }

                loglikelihoodThisLevel += loglikelihoodToAdd;

                for(int jLevel = 0; jLevel < currentLevel; jLevel++)
                {
                    delete SicB[jLevel];
                    delete[] covarianceMatrixMemory[jLevel];
                    delete covarianceMatrix[jLevel];
                }

                delete[] covObsKnots;
            }

            delete[] covarianceMatrixMemory;
            delete[] covarianceMatrix;
        }

        if(CALCULATION_MODE=="prediction")
        {
            int nPredictionsInCurrentRegion = partition->nPredictionsAtFinestLevel[indexRegionAtThisLevel];
            if(nPredictionsInCurrentRegion == 0) continue;

            //Allocate memory for variables in the current region
            mat **covarianceMatrix = new mat* [currentLevel+1];

            double **covarianceMatrixMemory = new double* [currentLevel];
            for(int jLevel = 0; jLevel < currentLevel; jLevel++)
                covarianceMatrixMemory[jLevel] = new double [nPredictionsInCurrentRegion*nKnots];

            double **posteriorPredictionKCholTimesw = new double* [currentLevel+1];
            for(int jLevel = 0; jLevel < currentLevel; jLevel++)
                posteriorPredictionKCholTimesw[jLevel]= new double [nPredictionsInCurrentRegion*nKnots];

            if(nKnotsInCurrentRegion > 0)
                posteriorPredictionKCholTimesw[currentLevel]= new double [nPredictionsInCurrentRegion*nKnotsInCurrentRegion];            

            for(int jLevel = 0; jLevel < currentLevel; jLevel++)
            {
                //Calculate the covariances between prediction locations in the current region and knots in the ancestor regions
                covarianceMatrix[jLevel] =  new mat(covarianceMatrixMemory[jLevel],nKnots,nPredictionsInCurrentRegion,false,true);

                if(NONSTATIONARY_FLAG)
                    evaluate_nonstationary_cross_covariance(covarianceMatrixMemory[jLevel], 
                    partition->predictionLon[indexRegionAtThisLevel], partition->predictionLat[indexRegionAtThisLevel],
                    partition->predictionX[indexRegionAtThisLevel], partition->predictionY[indexRegionAtThisLevel],
                    partition->predictionZ[indexRegionAtThisLevel],
                    partition->knotsLon[ancestorArray[rankOpenMP][jLevel]], partition->knotsLat[ancestorArray[rankOpenMP][jLevel]],
                    partition->knotsX[ancestorArray[rankOpenMP][jLevel]], partition->knotsY[ancestorArray[rankOpenMP][jLevel]], partition->knotsZ[ancestorArray[rankOpenMP][jLevel]],
                    nPredictionsInCurrentRegion, nKnots, sill_scale, range_scale, nugget_scale);
                else
                    evaluate_cross_covariance(covarianceMatrixMemory[jLevel],
                    partition->predictionLon[indexRegionAtThisLevel], partition->predictionLat[indexRegionAtThisLevel],
                    partition->predictionX[indexRegionAtThisLevel], partition->predictionY[indexRegionAtThisLevel],
                    partition->predictionZ[indexRegionAtThisLevel],
                    partition->knotsLon[ancestorArray[rankOpenMP][jLevel]], partition->knotsLat[ancestorArray[rankOpenMP][jLevel]],
                    partition->knotsX[ancestorArray[rankOpenMP][jLevel]], partition->knotsY[ancestorArray[rankOpenMP][jLevel]], partition->knotsZ[ancestorArray[rankOpenMP][jLevel]],
                    nPredictionsInCurrentRegion, nKnots, sill, range, nugget);
                
                get_conditional_covariance_matrix(jLevel, posteriorPredictionKCholTimesw, KCholTimesw[ancestorArray[rankOpenMP][jLevel]], covarianceMatrixMemory[jLevel],nPredictionsInCurrentRegion,nKnots,nKnots);

                //posteriorPredictionKCholTimesw[jLevel] = RChol[ancestorArray[rankOpenMP][jLevel]]^(-1)*covarianceMatrix[rankOpenMP][jLevel]
                memcpy(posteriorPredictionKCholTimesw[jLevel],covarianceMatrixMemory[jLevel],nKnots*nPredictionsInCurrentRegion*sizeof(double));
                dtrtrs_(&L,&N,&N,&nKnots,&nPredictionsInCurrentRegion,RChol[ancestorArray[rankOpenMP][jLevel]].memptr(),&nKnots,posteriorPredictionKCholTimesw[jLevel],&nKnots,&status);

            }

            if(nKnotsInCurrentRegion > 0)
            {
                //Calculate the covariances between prediction locations and the knots in the current region
                if(NONSTATIONARY_FLAG)
                {
                    evaluate_nonstationary_cross_covariance(posteriorPredictionKCholTimesw[currentLevel],  partition->predictionLon[indexRegionAtThisLevel], partition->predictionLat[indexRegionAtThisLevel], 
                    partition->predictionX[indexRegionAtThisLevel], partition->predictionY[indexRegionAtThisLevel], 
                    partition->predictionZ[indexRegionAtThisLevel],
                    partition->knotsLon[iRegion], partition->knotsLat[iRegion], 
                    partition->knotsX[iRegion], partition->knotsY[iRegion], partition->knotsZ[iRegion], 
                    nPredictionsInCurrentRegion, nKnotsInCurrentRegion, sill_scale, range_scale, nugget_scale);
                }else
                {
                    evaluate_cross_covariance(posteriorPredictionKCholTimesw[currentLevel],
                    partition->predictionLon[indexRegionAtThisLevel], partition->predictionLat[indexRegionAtThisLevel], 
                    partition->predictionX[indexRegionAtThisLevel], partition->predictionY[indexRegionAtThisLevel], 
                    partition->predictionZ[indexRegionAtThisLevel],
                    partition->knotsLon[iRegion], partition->knotsLat[iRegion], 
                    partition->knotsX[iRegion], partition->knotsY[iRegion], partition->knotsZ[iRegion], 
                    nPredictionsInCurrentRegion, nKnotsInCurrentRegion, sill, range, nugget);
                }

                get_conditional_covariance_matrix(currentLevel, posteriorPredictionKCholTimesw, KCholTimesw[iRegion], posteriorPredictionKCholTimesw[currentLevel],nPredictionsInCurrentRegion,nKnotsInCurrentRegion,nKnots);

                //posteriorPredictionKCholTimesw[currentLevel] := RCholThisLevel^(-1)*posteriorPredictionKCholTimesw[currentLevel]
                dtrtrs_(&L,&N,&N,&nKnotsInCurrentRegion,&nPredictionsInCurrentRegion,RCholThisLevel,&nKnotsInCurrentRegion,posteriorPredictionKCholTimesw[currentLevel],&nKnotsInCurrentRegion,&status);
            }

            //Get the variance-covariance matrix of the prediction locations in the current region
            mat *varianceCovarianceMatrix = new mat(nPredictionsInCurrentRegion,nPredictionsInCurrentRegion);

            if(NONSTATIONARY_FLAG)
                evaluate_nonstationary_variance_covariance((*varianceCovarianceMatrix).memptr(),
                partition->predictionLon[indexRegionAtThisLevel], partition->predictionLat[indexRegionAtThisLevel],
                partition->predictionX[indexRegionAtThisLevel], partition->predictionY[indexRegionAtThisLevel],
                partition->predictionZ[indexRegionAtThisLevel],
                nPredictionsInCurrentRegion, sill_scale, range_scale, nugget_scale);
            else
                evaluate_variance_covariance((*varianceCovarianceMatrix).memptr(),
                partition->predictionLon[indexRegionAtThisLevel], partition->predictionLat[indexRegionAtThisLevel],
                partition->predictionX[indexRegionAtThisLevel], partition->predictionY[indexRegionAtThisLevel],
                partition->predictionZ[indexRegionAtThisLevel],
                nPredictionsInCurrentRegion, sill, range, nugget);

            get_conditional_covariance_matrix(currentLevel, posteriorPredictionKCholTimesw, posteriorPredictionKCholTimesw, (*varianceCovarianceMatrix).memptr(), nPredictionsInCurrentRegion, nPredictionsInCurrentRegion, nKnots);

            posteriorPredictionMean[indexRegionAtThisLevel].set_size(nPredictionsInCurrentRegion);

            if(nKnotsInCurrentRegion > 0)
            {
                cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,nPredictionsInCurrentRegion,nPredictionsInCurrentRegion,nKnotsInCurrentRegion,-1.0,posteriorPredictionKCholTimesw[currentLevel],nKnotsInCurrentRegion,posteriorPredictionKCholTimesw[currentLevel],nKnotsInCurrentRegion,1.0,(*varianceCovarianceMatrix).memptr(),nPredictionsInCurrentRegion);

                cblas_dgemv(CblasColMajor, CblasTrans, nKnotsInCurrentRegion, nPredictionsInCurrentRegion, 1.0, posteriorPredictionKCholTimesw[currentLevel], nKnotsInCurrentRegion, Sicy->memptr(), 1, 0.0, posteriorPredictionMean[indexRegionAtThisLevel].memptr(), 1);
            }
            else
                memset(posteriorPredictionMean[indexRegionAtThisLevel].memptr(),0,sizeof(double)*nPredictionsInCurrentRegion);

            posteriorPredictionVariance[indexRegionAtThisLevel] = (*varianceCovarianceMatrix).diag();

            //BTilde[indexRegionAtThisLevel][jLevel] := covarianceMatrix[jLevel]-posteriorPredictionKCholTimesw[currentLevel].t()*SicB[jLevel]
            BTilde[indexRegionAtThisLevel] = covarianceMatrixMemory;

            if(nKnotsInCurrentRegion > 0)
                for(int jLevel = 0; jLevel < currentLevel; jLevel++)
                    cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,nKnots,nPredictionsInCurrentRegion,nKnotsInCurrentRegion,-1.0,SicB[jLevel]->memptr(),nKnotsInCurrentRegion,posteriorPredictionKCholTimesw[currentLevel],nKnotsInCurrentRegion,1.0,BTilde[indexRegionAtThisLevel][jLevel],nKnots);

            delete varianceCovarianceMatrix;
            for(int jLevel = 0; jLevel < currentLevel; jLevel++)
                delete[] posteriorPredictionKCholTimesw[jLevel];

            if(nKnotsInCurrentRegion > 0)
                delete[] posteriorPredictionKCholTimesw[currentLevel];
        }

        if(nKnotsInCurrentRegion > 0)
        {
            delete[] covObsKnots;
            delete[] RCholThisLevel;   
            delete Sicy;
        }
    }

    if(CALCULATION_MODE!="prediction") loglikelihood += loglikelihoodThisLevel;
    
    //cout<<"Lvl:"<<currentLevel<<"\tWK:"<<WORKER<<"\t"<<loglikelihoodThisLevel<<endl;

    //Free memory for ancestorArray
    for(int i = 0; i < maxOpenMPThreads; i++) delete[] ancestorArray[i];
    delete[] ancestorArray; ancestorArray = NULL;
}