#include <iostream>
#include <armadillo>
#include <mpi.h>
#include <omp.h>
#include <mkl.h>

#include "class_data.hpp"
#include "class_approximation.hpp"
#include "class_partition.hpp"
#include "evaluate_covariance.hpp"
#include "constants.hpp"
#include <string.h>

void Approximation::posterior_inference()
{
    if(WORKER == 0)
    {
        gettimeofday(&timeNow, NULL);
        cout<<"===========================> Processor 1: calculating posterior quantities starts. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n";
    }

    //Set WORKING_REGION_FLAG to false for all regions before the second finest level
    memset(WORKING_REGION_FLAG,false,partition->nRegionsInTotal-partition->nRegionsAtEachLevel[NUM_LEVELS_M-1]-partition->nRegionsAtEachLevel[NUM_LEVELS_M-2]);

    //Allocate memory for KCholTimesCurrentw (vector size: partition->nKnots)
    KCholTimesCurrentw = new vec [partition->nRegionsInTotal];

    //Allocate memory for KCholTimesCurrentA (cube size: first dimension: partition->nKnots, second dimesnion: partition->nKnots, third dimension or the number of slices: (partition->nRegionsInTotal-1)*partition->nRegionsInTotal/2)
    KCholTimesCurrentA = new cube [partition->nRegionsInTotal-partition->nRegionsAtEachLevel[NUM_LEVELS_M-1]];

    //Allocate memory for temporary variables
    int maxOpenMPThreads=omp_get_max_threads();
    vec* w = new vec [maxOpenMPThreads];
    mat* A = new mat [maxOpenMPThreads];
    double **tempMemory = new double* [maxOpenMPThreads];
    for(int i = 0; i < maxOpenMPThreads; i++)
    {
            w[i].set_size(partition->nKnots);
            A[i].set_size(partition->nKnots,partition->nKnots);
            tempMemory[i] =  new double [partition->nKnots*partition->nKnots];
    }

    //Temporary variables if load ATilde from disk 
    int size=(NUM_LEVELS_M-1)*NUM_LEVELS_M/2;
    unsigned long offset = partition->nRegionsInTotal-partition->nRegionsAtEachLevel[NUM_LEVELS_M-1];

    //The second finest level
    int currentLevel = NUM_LEVELS_M-2;
    double loglikelihoodThisLevel = 0;

    #pragma omp parallel for num_threads(maxOpenMPThreads) reduction(+:loglikelihoodThisLevel) schedule(dynamic,1)
    for(unsigned long iRegion = REGION_START[currentLevel]; iRegion < REGION_END[currentLevel] + 1; iRegion++)
    {   
        //Skip regions that are not dealt with by this worker
        if( !WORKING_REGION_FLAG[iRegion] ) continue;

        int rankOpenMP = omp_get_thread_num();

        double loglikelihoodToAdd;
        bool supervisor = true;
        bool synchronizeFlag = false;

        unsigned long indexRegionAtThisLevel = iRegion - partition->indexStartSecondFinestLevel;

        //Deal with this region
        if(VARIOUS_J_FINEST_LEVEL_FLAG)
        {
            if(SAVE_TO_DISK_FLAG) load_ATilde_from_disk(partition->childrenStart[indexRegionAtThisLevel], partition->childrenEnd[indexRegionAtThisLevel], currentLevel == NUM_LEVELS_M-2, partition->nKnotsAtFinestLevel, offset, partition->nKnots, size);

            aggregate_w_and_A(w[rankOpenMP], A[rankOpenMP], tempMemory[rankOpenMP], currentLevel, (currentLevel+1)*(currentLevel+2)/2-1, partition->childrenStart[indexRegionAtThisLevel], partition->childrenEnd[indexRegionAtThisLevel], iRegion, supervisor, synchronizeFlag);
        }
        else
        {
            if(SAVE_TO_DISK_FLAG) load_ATilde_from_disk(iRegion*NUM_PARTITIONS_J+1, iRegion*NUM_PARTITIONS_J+NUM_PARTITIONS_J, currentLevel == NUM_LEVELS_M-2, partition->nKnotsAtFinestLevel, offset, partition->nKnots, size);

            aggregate_w_and_A(w[rankOpenMP], A[rankOpenMP], tempMemory[rankOpenMP], currentLevel, (currentLevel+1)*(currentLevel+2)/2-1, iRegion*NUM_PARTITIONS_J+1, iRegion*NUM_PARTITIONS_J+NUM_PARTITIONS_J, iRegion, supervisor, synchronizeFlag);
        }

        if(supervisor)
        {
            if(CALCULATION_MODE != "prediction")
                loglikelihoodToAdd = -2*sum(log(RChol[iRegion].diag()));

            //Compute the posterior conditional variance-covariance matrix
            RChol[iRegion] = chol( RChol[iRegion]*RChol[iRegion].t()+A[rankOpenMP],"lower");

            //Compute KCholTimesCurrentw that is the Cholesky factor of the inverse of the conditional posterior variance-covariance matrix times w
            KCholTimesCurrentw[iRegion].set_size(partition->nKnots);
            KCholTimesCurrentw[iRegion] = solve(trimatl(RChol[iRegion]),w[rankOpenMP]);

            //Compute KCholTimesCurrentA that is the Cholesky factor of the inverse of the conditional posterior variance-covariance matrix times A
            KCholTimesCurrentA[iRegion].set_size(partition->nKnots,partition->nKnots,currentLevel);
        }

        //Get wTilde and ATilde
        if(currentLevel != 0)
            get_wTilde_and_ATilde_in_posterior(w[rankOpenMP],A[rankOpenMP], tempMemory[rankOpenMP], partition->nKnots, KCholTimesCurrentA[iRegion], KCholTimesCurrentw[iRegion], iRegion, currentLevel, supervisor, synchronizeFlag);

        if(CALCULATION_MODE != "prediction")
            KCholTimesCurrentA[iRegion].reset();

        //Free wTilde and ATilde of the children to save memory
        if(VARIOUS_J_FINEST_LEVEL_FLAG)
            free_wTilde_and_ATilde(partition->childrenStart[indexRegionAtThisLevel], partition->childrenEnd[indexRegionAtThisLevel]);
        else
            free_wTilde_and_ATilde(iRegion*NUM_PARTITIONS_J+1, iRegion*NUM_PARTITIONS_J+NUM_PARTITIONS_J);

        if(!supervisor)
        {
            INDICES_REGIONS_AT_CURRENT_LEVEL.erase(iRegion);
            WORKING_REGION_FLAG[iRegion] = false;
        }
        
        if(supervisor && CALCULATION_MODE != "prediction")
            loglikelihoodToAdd += 2*sum(log(RChol[iRegion].diag()))-dot(KCholTimesCurrentw[iRegion],KCholTimesCurrentw[iRegion]);
        // cout<<"!!!! "<<sum(log(RChol[iRegion].diag()))<<"\t"<<dot(KCholTimesCurrentw[iRegion],KCholTimesCurrentw[iRegion])<<"\n";

        if(supervisor && CALCULATION_MODE != "prediction")
            loglikelihoodThisLevel += loglikelihoodToAdd;

        if(CALCULATION_MODE != "prediction") 
            KCholTimesCurrentw[iRegion].reset();

        if(supervisor && SAVE_TO_DISK_FLAG)
        {
            //save ATilde of the current region to disk
            std::string fileName=TMP_DIRECTORY+"/"+std::to_string(iRegion)+".bin";
            ATilde[iRegion].save(fileName,arma_binary);
            ATilde[iRegion].reset();
        }
    }

    if(currentLevel > 0)
        synchronizeIndicesRegionsForEachWorker();

    loglikelihood += loglikelihoodThisLevel;
    // cout<<"Lvl:"<<currentLevel<<"\tWK:"<<WORKER<<"\t"<<loglikelihoodThisLevel<<endl;

    timeval timeNow;
    gettimeofday(&timeNow, NULL);

    MPI_Barrier(WORLD);
    if(WORKER == 0) cout<<"Posterior: Level "<<currentLevel+1<<" is complete. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n";

    //From the third finest to the coarsest level
    for(int currentLevel = NUM_LEVELS_M-3; currentLevel > -1 ; currentLevel--)
    {
        double loglikelihoodThisLevel = 0;

        #pragma omp parallel for num_threads(maxOpenMPThreads) reduction(+:loglikelihoodThisLevel) schedule(dynamic,1)
        for(unsigned long iRegion = REGION_START[currentLevel]; iRegion < REGION_END[currentLevel] + 1; iRegion++)
        {   
            //Skip regions that are not dealt with by this worker
            if( !WORKING_REGION_FLAG[iRegion] ) continue;

            int rankOpenMP = omp_get_thread_num();

            double loglikelihoodToAdd;
            bool supervisor = true;
            bool synchronizeFlag = false;

            //Deal with this region
            if(SAVE_TO_DISK_FLAG) load_ATilde_from_disk(iRegion*NUM_PARTITIONS_J+1, iRegion*NUM_PARTITIONS_J+NUM_PARTITIONS_J, currentLevel == NUM_LEVELS_M-2, partition->nKnotsAtFinestLevel, offset, partition->nKnots, size);

            aggregate_w_and_A(w[rankOpenMP], A[rankOpenMP], tempMemory[rankOpenMP], currentLevel, (currentLevel+1)*(currentLevel+2)/2-1, iRegion*NUM_PARTITIONS_J+1, iRegion*NUM_PARTITIONS_J+NUM_PARTITIONS_J, iRegion, supervisor, synchronizeFlag);

            if(supervisor)
            {
                if(CALCULATION_MODE != "prediction")
                    loglikelihoodToAdd = -2*sum(log(RChol[iRegion].diag()));

                //Compute the posterior conditional variance-covariance matrix
                RChol[iRegion] = chol( RChol[iRegion]*RChol[iRegion].t()+A[rankOpenMP],"lower");

                //Compute KCholTimesCurrentw that is the Cholesky factor of the inverse of the conditional posterior variance-covariance matrix times w
                KCholTimesCurrentw[iRegion].set_size(partition->nKnots);
                KCholTimesCurrentw[iRegion] = solve(trimatl(RChol[iRegion]),w[rankOpenMP]);

                //Compute KCholTimesCurrentA that is the Cholesky factor of the inverse of the conditional posterior variance-covariance matrix times A
                KCholTimesCurrentA[iRegion].set_size(partition->nKnots,partition->nKnots,currentLevel);
            }

            //Get wTilde and ATilde
            if(currentLevel != 0)
                get_wTilde_and_ATilde_in_posterior(w[rankOpenMP],A[rankOpenMP], tempMemory[rankOpenMP], partition->nKnots, KCholTimesCurrentA[iRegion], KCholTimesCurrentw[iRegion], iRegion, currentLevel, supervisor, synchronizeFlag);

            if(CALCULATION_MODE != "prediction")
                KCholTimesCurrentA[iRegion].reset();

            //Free wTilde and ATilde of the children to save memory
            free_wTilde_and_ATilde(iRegion*NUM_PARTITIONS_J+1, iRegion*NUM_PARTITIONS_J+NUM_PARTITIONS_J);
            if(!supervisor)
            {
                INDICES_REGIONS_AT_CURRENT_LEVEL.erase(iRegion);
                WORKING_REGION_FLAG[iRegion] = false;
            }
            
            if(supervisor && CALCULATION_MODE != "prediction")
                loglikelihoodToAdd += 2*sum(log(RChol[iRegion].diag()))-dot(KCholTimesCurrentw[iRegion],KCholTimesCurrentw[iRegion]);
            
            if(supervisor && CALCULATION_MODE != "prediction")
                loglikelihoodThisLevel += loglikelihoodToAdd;

            if(CALCULATION_MODE != "prediction") 
                KCholTimesCurrentw[iRegion].reset();

            if(supervisor && SAVE_TO_DISK_FLAG)
            {
                //save ATilde of the current region to disk
                std::string fileName=TMP_DIRECTORY+"/"+std::to_string(iRegion)+".bin";
                ATilde[iRegion].save(fileName,arma_binary);
                ATilde[iRegion].reset();
            }
        }

        if(currentLevel > 0)
            synchronizeIndicesRegionsForEachWorker();

        loglikelihood += loglikelihoodThisLevel;
        // cout<<"Lvl:"<<currentLevel<<"\tWK:"<<WORKER<<"\t"<<loglikelihoodThisLevel<<endl;

        timeval timeNow;
        gettimeofday(&timeNow, NULL);

        MPI_Barrier(WORLD);
        if(WORKER == 0) cout<<"Posterior: Level "<<currentLevel+1<<" is complete. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n";

    }
  
    MPI_Request request;

    if(CALCULATION_MODE!="prediction")
    {
        if(WORKER != 0)
            MPI_Isend(&loglikelihood,1,MPI_DOUBLE,0,MPI_TAG_LIKELIHOOD,WORLD, &request);
        else
        {
            double likelihoodComponent;
            for(int i = 1; i < MPI_SIZE; i++)
            {
                MPI_Recv(&likelihoodComponent,1,MPI_DOUBLE,i,MPI_TAG_LIKELIHOOD,WORLD,MPI_STATUS_IGNORE);
                loglikelihood += likelihoodComponent;
            }
            cout<<"The obtained loglikelihood is: "<<-0.5*(loglikelihood+NUM_OBSERVATIONS*log(2*3.14159265359))<<endl;
        }
    }
    

    //Deallocate temporary memory
    for(int i = 0; i < maxOpenMPThreads; i++)
    {
        w[i].reset();
        A[i].reset();
        delete[] tempMemory[i];
    }
    delete[] w;
    delete[] A;
    delete[] tempMemory;

    if(WORKER == 0)
    {
        gettimeofday(&timeNow, NULL);
        cout<<"===========================>  Processor 1: calculating posterior quantities is complete. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n\n";
    }
}