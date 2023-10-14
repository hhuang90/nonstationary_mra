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


void Approximation::predict()
{
    if(WORKER == 0)
    {
        gettimeofday(&timeNow, NULL);
        cout<<"===========================> Processor 1: predicting starts. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n";
    }
			

    int maxOpenMPThreads=omp_get_max_threads();
    int currentLevel = NUM_LEVELS_M-1;
    unsigned long indexRegionAtThisLevel = 0;
    int nKnots = partition->nKnots;
    char L='L', N='N';

    //Define indices of all the ancestors for the current region
    unsigned long **ancestorArray = new unsigned long* [maxOpenMPThreads];
    for(int i = 0; i < maxOpenMPThreads; i++) ancestorArray[i] = new unsigned long [currentLevel];

    //Recalculate WORKERS_FOR_EACH_REGION
    if(NUM_LEVELS_M>2)
    {
        for(unsigned long iRegion = partition->indexStartSecondFinestLevel; iRegion < partition->indexStartFinestLevel; iRegion++)
        {
            unsigned long ancestor = (iRegion-1)/NUM_PARTITIONS_J;
            while(ancestor>0)
            {
                WORKERS_FOR_EACH_REGION[ancestor].insert(WORKERS_FOR_EACH_REGION[iRegion].begin(),WORKERS_FOR_EACH_REGION[iRegion].end());
                ancestor = (ancestor-1)/NUM_PARTITIONS_J;
            }
        }
    }

    //Synchronize RChol, KCholTimesCurrentw and KCholTimesCurrentA
    if(WORKER==0)
    {
        for(int iWorker = 1; iWorker < MPI_SIZE; iWorker++)
        {
            MPI_Send(KCholTimesCurrentw[0].memptr(),partition->nKnots,MPI_DOUBLE,iWorker,MPI_TAG_VEC,WORLD);
            MPI_Send(RChol[0].memptr(),partition->nKnots*partition->nKnots,MPI_DOUBLE,iWorker,MPI_TAG_MAT,WORLD);
        }
    }
    else
    {
        KCholTimesCurrentw[0].set_size(partition->nKnots);
        MPI_Recv(KCholTimesCurrentw[0].memptr(),partition->nKnots,MPI_DOUBLE,0,MPI_TAG_VEC,WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(RChol[0].memptr(),partition->nKnots*partition->nKnots,MPI_DOUBLE,0,MPI_TAG_MAT,WORLD,MPI_STATUS_IGNORE);
    }

    unsigned long regionStart = 1, regionEnd = NUM_PARTITIONS_J+1;
    for(int currentLevel = 1; currentLevel < NUM_LEVELS_M-1; currentLevel++)
    {
        #pragma omp parallel for schedule(dynamic,1)
        for(unsigned long iRegion = regionStart; iRegion < regionEnd; iRegion++)
        {
            if(WORKERS_FOR_EACH_REGION[iRegion].size() > 1)
            {
                std::set<unsigned long>::iterator workerToSend = WORKERS_FOR_EACH_REGION[iRegion].begin();
                
                if(WORKER==*workerToSend)
                {
                    //Supervisor, send KCholTimesCurrentw[iRegion] to other workers
                    for(workerToSend++; workerToSend != WORKERS_FOR_EACH_REGION[iRegion].end(); workerToSend++)
                    {
                        MPI_Send(KCholTimesCurrentw[iRegion].memptr(),partition->nKnots,MPI_DOUBLE,*workerToSend,MPI_TAG_VEC,WORLD);
                        MPI_Send(KCholTimesCurrentA[iRegion].memptr(),partition->nKnots*partition->nKnots*currentLevel,MPI_DOUBLE,*workerToSend,MPI_TAG_MAT,WORLD);
                        MPI_Send(RChol[iRegion].memptr(),partition->nKnots*partition->nKnots,MPI_DOUBLE,*workerToSend,MPI_TAG_MAT,WORLD);
                    }
                }else
                {
                    for(workerToSend++; workerToSend != WORKERS_FOR_EACH_REGION[iRegion].end(); workerToSend++)
                    {
                        if(WORKER!=*workerToSend) continue;
                        //Not supervisor, receive KCholTimesCurrentw[iRegion] from supervisor
                        KCholTimesCurrentw[iRegion].set_size(partition->nKnots);
                        KCholTimesCurrentA[iRegion].set_size(partition->nKnots,partition->nKnots,currentLevel);
                        RChol[iRegion].set_size(partition->nKnots,partition->nKnots);
                        
                        MPI_Recv(KCholTimesCurrentw[iRegion].memptr(),partition->nKnots,MPI_DOUBLE,*WORKERS_FOR_EACH_REGION[iRegion].begin(),MPI_TAG_VEC,WORLD,MPI_STATUS_IGNORE);
                        MPI_Recv(KCholTimesCurrentA[iRegion].memptr(),partition->nKnots*partition->nKnots*currentLevel,MPI_DOUBLE,*WORKERS_FOR_EACH_REGION[iRegion].begin(),MPI_TAG_MAT,WORLD,MPI_STATUS_IGNORE);
                        MPI_Recv(RChol[iRegion].memptr(),partition->nKnots*partition->nKnots,MPI_DOUBLE,*WORKERS_FOR_EACH_REGION[iRegion].begin(),MPI_TAG_MAT,WORLD,MPI_STATUS_IGNORE);
                        break;
                    }
                }
            }
        }
        regionStart = regionStart*NUM_PARTITIONS_J+1;
        regionEnd = regionEnd*NUM_PARTITIONS_J+1;
    }

    #pragma omp parallel for schedule(dynamic,1)
    for(unsigned long iRegion = REGION_START[NUM_LEVELS_M-1]; iRegion < REGION_END[NUM_LEVELS_M-1] + 1; iRegion++)
    {
        int rankOpenMP = omp_get_thread_num();
        int status;
        unsigned long indexRegionAtThisLevel = iRegion-partition->indexStartFinestLevel;
        int nPredictionsInCurrentRegion = partition->nPredictionsAtFinestLevel[indexRegionAtThisLevel];
        if(nPredictionsInCurrentRegion == 0) continue;

        //Find the indices of all the ancestors
        if(VARIOUS_J_FINEST_LEVEL_FLAG)
        {
            ancestorArray[rankOpenMP][currentLevel-1] = partition->parent[iRegion - partition->indexStartFinestLevel];
            get_all_ancestors(ancestorArray[rankOpenMP], ancestorArray[rankOpenMP][currentLevel-1], currentLevel-1);
        }
        else
            get_all_ancestors(ancestorArray[rankOpenMP], iRegion, currentLevel);
        
        int jLevel = currentLevel-1;
        mat* KcBTilde;
        mat** tmpBTilde = new mat* [jLevel];
        
        //KcBTilde := RChol[ancestorArray[rankOpenMP][jLevel]])^(-1)*BTilde[indexRegionAtThisLevel][jLevel]
        KcBTilde = new mat(BTilde[indexRegionAtThisLevel][jLevel],nKnots,nPredictionsInCurrentRegion,true,true);
        dtrtrs_(&L,&N,&N,&nKnots,&nPredictionsInCurrentRegion,RChol[ancestorArray[rankOpenMP][jLevel]].memptr(),&nKnots,KcBTilde->memptr(),&nKnots,&status);
        
        for(int kLevelBeforejLevel = 0 ; kLevelBeforejLevel < jLevel; kLevelBeforejLevel++)
        {
            //tmpBTilde[kLevelBeforejLevel] := BTilde[indexRegionAtThisLevel][kLevelBeforejLevel].t()-KCholTimesCurrentA[ancestorArray[rankOpenMP][currentLevel-1]].slice(kLevelBeforejLevel).t()*KcBTilde
            tmpBTilde[kLevelBeforejLevel] = new mat(BTilde[indexRegionAtThisLevel][kLevelBeforejLevel],nKnots,nPredictionsInCurrentRegion,false,true);
            cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,nKnots,nPredictionsInCurrentRegion,nKnots,-1.0,KCholTimesCurrentA[ancestorArray[rankOpenMP][currentLevel-1]].slice(kLevelBeforejLevel).memptr(),nKnots,KcBTilde->memptr(),nKnots,1.0,tmpBTilde[kLevelBeforejLevel]->memptr(),nKnots);
        }
        delete KcBTilde;
        
        for(int jLevel = currentLevel-2; jLevel > 0; jLevel--)
        {
            //KcBTilde := RChol[ancestorArray[rankOpenMP][jLevel]])^(-1)*tmpBTilde[jLevel])
            KcBTilde = new mat(tmpBTilde[jLevel]->memptr(),nKnots,nPredictionsInCurrentRegion,true,true);
            dtrtrs_(&L,&N,&N,&nKnots,&nPredictionsInCurrentRegion,RChol[ancestorArray[rankOpenMP][jLevel]].memptr(),&nKnots,KcBTilde->memptr(),&nKnots,&status);
            
            for(int kLevelBeforejLevel = 0 ; kLevelBeforejLevel < jLevel; kLevelBeforejLevel++)
            {
                //tmpBTilde[kLevelBeforejLevel] := tmpBTilde[kLevelBeforejLevel]-KCholTimesCurrentA[ancestorArray[rankOpenMP][jLevel]].slice(kLevelBeforejLevel).t()*KcBTilde
                cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,nKnots,nPredictionsInCurrentRegion,nKnots,-1.0,KCholTimesCurrentA[ancestorArray[rankOpenMP][jLevel]].slice(kLevelBeforejLevel).memptr(),nKnots,KcBTilde->memptr(),nKnots,1.0,tmpBTilde[kLevelBeforejLevel]->memptr(),nKnots);
            }

            delete KcBTilde;
        }
        
        for(jLevel = 0; jLevel < currentLevel-1; jLevel++)
        {
            //KcBTilde := RChol[ancestorArray[rankOpenMP][jLevel]])^(-1)*tmpBTilde[jLevel])
            KcBTilde = new mat(tmpBTilde[jLevel]->memptr(),nKnots,nPredictionsInCurrentRegion,false,true);
            
            dtrtrs_(&L,&N,&N,&nKnots,&nPredictionsInCurrentRegion,RChol[ancestorArray[rankOpenMP][jLevel]].memptr(),&nKnots,KcBTilde->memptr(),&nKnots,&status);
            
            //posteriorPredictionMean[indexRegionAtThisLevel] += KcBTilde.t()*KCholTimesCurrentw[ancestorArray[rankOpenMP][jLevel]]
            cblas_dgemv(CblasColMajor,CblasTrans,nKnots,nPredictionsInCurrentRegion,1.0,KcBTilde->memptr(),nKnots,KCholTimesCurrentw[ancestorArray[rankOpenMP][jLevel]].memptr(),1,1.0,posteriorPredictionMean[indexRegionAtThisLevel].memptr(),1);

            //posteriorPredictionVariance[indexRegionAtThisLevel] += columnSum(KcBTilde.^2);
            posteriorPredictionVariance[indexRegionAtThisLevel] += sum(square(*KcBTilde),0).t();
        }

        jLevel=currentLevel-1;
        
        //KcBTilde := RChol[ancestorArray[rankOpenMP][jLevel]])^(-1)*BTilde[indexRegionAtThisLevel][jLevel])
        KcBTilde = new mat(BTilde[indexRegionAtThisLevel][jLevel],nKnots,nPredictionsInCurrentRegion,false,true);
        dtrtrs_(&L,&N,&N,&nKnots,&nPredictionsInCurrentRegion,RChol[ancestorArray[rankOpenMP][jLevel]].memptr(),&nKnots,KcBTilde->memptr(),&nKnots,&status);

        //posteriorPredictionMean[indexRegionAtThisLevel] += KcBTilde.t()*KCholTimesCurrentw[ancestorArray[rankOpenMP][jLevel]]
        cblas_dgemv(CblasColMajor,CblasTrans,nKnots,nPredictionsInCurrentRegion,1.0,KcBTilde->memptr(),nKnots,KCholTimesCurrentw[ancestorArray[rankOpenMP][jLevel]].memptr(),1,1.0,posteriorPredictionMean[indexRegionAtThisLevel].memptr(),1);
        
        //posteriorPredictionVariance[indexRegionAtThisLevel] += columnSum(KcBTilde.^2);
        posteriorPredictionVariance[indexRegionAtThisLevel] += sum(square(*KcBTilde),0).t();
        
        delete KcBTilde;
        for(int jLevel = 0 ; jLevel < currentLevel-1; jLevel++)
        {
            delete[] BTilde[indexRegionAtThisLevel][jLevel];
            delete tmpBTilde[jLevel];
        }
        delete[] BTilde[indexRegionAtThisLevel][currentLevel-1];
        delete[] BTilde[indexRegionAtThisLevel];
        delete[] tmpBTilde;
    }

    //Free memory for temporary variables
    for(int i = 0; i < maxOpenMPThreads; i++) delete[] ancestorArray[i];
    delete[] ancestorArray;

    if(WORKER == 0)
    {
        gettimeofday(&timeNow, NULL);
        cout<<"===========================> Processor 1: predicting is complete. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n\n";
    }
}
