#include <iostream>
#include <fstream>
#include <armadillo>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "sys/stat.h"
#include <unordered_set>

#include "class_partition.hpp"
#include "constants.hpp"

/////////////////// Below are private member functions /////////////////

/*   
Creats knots in the current region
Input: 
    lonMin, lonMax, latMin, latMax: the boudary of the current region
    nKnotsLon, nKnotsLat: the number of knots in the current region in each direction
Output: 
    currentKnotsLon: the array of the coordinates in longitude for each knot
    currentKnotsLat: the array of the coordinates in latitude for each knot
*/
void Partition::create_knots(double *&currentKnotsLon, double *&currentKnotsLat, const double &lonMin, const double &lonMax, const int &nKnotsLon, const double &latMin, const double &latMax, const int &nKnotsLat)
{
    double offsetLon = (lonMax-lonMin)*OFFSET;
    double offsetLat = (latMax-latMin)*OFFSET;

    double lonStart = lonMin+offsetLon;
    double lonEnd = lonMax-offsetLon;
    double latStart = latMin+offsetLat;
    double latEnd = latMax-offsetLat;

    double lonIncrement = ( nKnotsLon==1 ? 0 : (lonEnd-lonStart)/(nKnotsLon-1));
    double latIncrement = ( nKnotsLat==1 ? 0 : (latEnd-latStart)/(nKnotsLat-1));

    #pragma omp parallel for
    for(int iKnotLon = 0; iKnotLon < nKnotsLon; iKnotLon++)
        for(int jKnotLat = 0; jKnotLat < nKnotsLat; jKnotLat++)
        {
            currentKnotsLon[iKnotLon*nKnotsLat+jKnotLat] = lonEnd-(nKnotsLon-iKnotLon-1)*lonIncrement;//xStart+iKnotX*xIncrement;
            currentKnotsLat[iKnotLon*nKnotsLat+jKnotLat] = latEnd-(nKnotsLat-jKnotLat-1)*latIncrement;//yStart+jKnotY*yIncrement;
        }
}

/*   
Creats partitions in the current region
*/
void Partition::create_partitions(unsigned long region, double *partitionLonMin, double *partitionLonMax, double *partitionLatMin, double *partitionLatMax)
{
    double lonMin=partitionLonMin[region];
    double lonMax=partitionLonMax[region];
    double latMin=partitionLatMin[region];
    double latMax=partitionLatMax[region];

    if(NUM_PARTITIONS_J==2)
    {
        if((lonMax-lonMin)>=(latMax-latMin))
        {
            double lonMid = (lonMax+lonMin)/2;
            partitionLonMin[region*2+1]=lonMin; partitionLonMin[region*2+2]=lonMid;
            partitionLonMax[region*2+1]=lonMid; partitionLonMax[region*2+2]=lonMax;

            partitionLatMin[region*2+1]=latMin; partitionLatMin[region*2+2]=latMin;
            partitionLatMax[region*2+1]=latMax; partitionLatMax[region*2+2]=latMax;
        }else
        {
            double latMid = (latMax+latMin)/2;
            partitionLonMin[region*2+1]=lonMin; partitionLonMin[region*2+2]=lonMin;
            partitionLonMax[region*2+1]=lonMax; partitionLonMax[region*2+2]=lonMax;

            partitionLatMin[region*2+1]=latMin; partitionLatMin[region*2+2]=latMid;
            partitionLatMax[region*2+1]=latMid; partitionLatMax[region*2+2]=latMax;
        }
    }else
    {
        double lonMid = (lonMax+lonMin)/2;
        double latMid = (latMax+latMin)/2;

        partitionLonMin[region*4+1]=lonMin; partitionLonMin[region*4+2]=lonMid; partitionLonMin[region*4+3]=lonMin; partitionLonMin[region*4+4]=lonMid;

        partitionLonMax[region*4+1]=lonMid; partitionLonMax[region*4+2]=lonMax; partitionLonMax[region*4+3]=lonMid; partitionLonMax[region*4+4]=lonMax;

        partitionLatMin[region*4+1]=latMin; partitionLatMin[region*4+2]=latMin; partitionLatMin[region*4+3]=latMid; partitionLatMin[region*4+4]=latMid;

        partitionLatMax[region*4+1]=latMid; partitionLatMax[region*4+2]=latMid; partitionLatMax[region*4+3]=latMax; partitionLatMax[region*4+4]=latMax;
    }
}

/*
Build partitions by user-provided knots files
*/
void Partition::build_partition_by_user_files()
{
    nKnots = NUM_KNOTS_r;

    if(WORKER == 0)
	{
		gettimeofday(&timeNow, NULL);
		cout<<"===========================> Processor 1: reading knots from "<<KNOTS_FILE_NAME<<" starts. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n";
	}

    //Check knots file exists
    struct stat fileStatus;
    if(stat(KNOTS_FILE_NAME.c_str(),&fileStatus) == -1 || !S_ISREG(fileStatus.st_mode))
    {
        if(WORKER == 0) cout<<"Program exits with an error: the knots file "<<KNOTS_FILE_NAME<<" does not exist.\n";
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    //Open the knots file by KNOTS_FILE_NAME
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD,KNOTS_FILE_NAME.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&file);

    //Read knots
    //lon and lat of all knots
    double *lon, *lat;
    unsigned long *region_index;
    
    //Get the number of locations, nLocations   
    MPI_Status* readStatus = new MPI_Status;
    int count;

    unsigned long nKnotsInTotal = 0;
    MPI_File_read(file,(void*)&nKnotsInTotal,1,MPI_UNSIGNED_LONG,readStatus);
    MPI_Get_count(readStatus,MPI_UNSIGNED_LONG,&count);

    if(count!=1)
    {
        if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<KNOTS_FILE_NAME<<" for the number of locations. Double check the data file format."<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    lon = new double [nKnotsInTotal];
    lat = new double [nKnotsInTotal];
    region_index = new unsigned long [nKnotsInTotal];

    //Get longitude
    int interval=INT_MAX;
    unsigned long positionStart = 0, positionEnd = positionStart + interval;
    while(positionStart<nKnotsInTotal)
    {
        if(positionEnd>nKnotsInTotal) interval = nKnotsInTotal - positionStart;
        
        MPI_File_read(file,(void*)&lon[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<KNOTS_FILE_NAME<<" for x. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Get latitude
    interval=INT_MAX;
    positionStart = 0; positionEnd = positionStart + interval;
    while(positionStart<nKnotsInTotal)
    {
        if(positionEnd>nKnotsInTotal) interval = nKnotsInTotal - positionStart;
        
        MPI_File_read(file,(void*)&lat[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<KNOTS_FILE_NAME<<" for y. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Get region_index
    interval=INT_MAX;
    positionStart = 0; positionEnd = positionStart + interval;
    while(positionStart<nKnotsInTotal)
    {
        if(positionEnd>nKnotsInTotal) interval = nKnotsInTotal - positionStart;
        
        MPI_File_read(file,(void*)&region_index[positionStart],interval,MPI_UNSIGNED_LONG,readStatus);
        MPI_Get_count(readStatus,MPI_UNSIGNED_LONG,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<KNOTS_FILE_NAME<<" for region indices. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }


    //Close the knots file
    MPI_File_close(&file);

    if(WORKER == 0)
	{
		gettimeofday(&timeNow, NULL);
		cout<<"===========================> Processor 1: reading knots from "<<KNOTS_FILE_NAME<<" is complete. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n";
	}

    #pragma omp parallel for schedule(dynamic,1)
    for(unsigned long iRegion = 0; iRegion < nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M-1]; iRegion++)
    {
        if(WORKING_REGION_FLAG[iRegion])
        {
            knotsLon[iRegion] =  new double [nKnots];
            knotsLat[iRegion] =  new double [nKnots];

            if(CHORDAL_DISTANCE_FLAG)
            {
                knotsX[iRegion] = new double [nKnots];
                knotsY[iRegion] = new double [nKnots];
                knotsZ[iRegion] = new double [nKnots];
            }
        }
    }

    if(!OBS_AS_KNOTS_FLAG)
    {
        #pragma omp parallel for schedule(dynamic,1)
        for(unsigned long iRegion = nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M-1]; iRegion < nRegionsInTotal; iRegion++)
        {
            if(WORKING_REGION_FLAG[iRegion])
            {
                knotsLon[iRegion] =  new double [NUM_KNOTS_FINEST];
                knotsLat[iRegion] =  new double [NUM_KNOTS_FINEST];

                if(CHORDAL_DISTANCE_FLAG)
                {
                    knotsX[iRegion] = new double [NUM_KNOTS_FINEST];
                    knotsY[iRegion] = new double [NUM_KNOTS_FINEST];
                    knotsZ[iRegion] = new double [NUM_KNOTS_FINEST];
                }
            }
        }
    }

    for(unsigned long i = 0; i < nKnotsInTotal;)
    {
        if(region_index[i] < nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M-1])
        {
            if(WORKING_REGION_FLAG[region_index[i]])
            {
                for(unsigned long j = i; j < i + NUM_KNOTS_r; j++)
                {
                    knotsLon[region_index[j]][j-i] = lon[j];
                    knotsLat[region_index[j]][j-i] = lat[j];
                    
                    if(CHORDAL_DISTANCE_FLAG)
                    {
                        double cosLat = cos(lat[j] * M_PI/180);
                        double cosLon = cos(lon[j] * M_PI/180);
                        double sinLat = sin(lat[j] * M_PI/180);
                        double sinLon = sin(lon[j] * M_PI/180);

                        knotsX[region_index[j]][j-i] = cosLat * cosLon;
                        knotsY[region_index[j]][j-i] = cosLat * sinLon;
                        knotsZ[region_index[j]][j-i] = sinLat;
                    }
                }
            }
            i += NUM_KNOTS_r;
        }else
        {
            if(!OBS_AS_KNOTS_FLAG)
            {
                if(WORKING_REGION_FLAG[region_index[i]])
                {
                    for(unsigned long j = i; j < i + NUM_KNOTS_FINEST; j++)
                    {
                        knotsLon[region_index[j]][j-i] = lon[j];
                        knotsLat[region_index[j]][j-i] = lat[j];
                        
                        if(CHORDAL_DISTANCE_FLAG)
                        {
                            double cosLat = cos(lat[j] * M_PI/180);
                            double cosLon = cos(lon[j] * M_PI/180);
                            double sinLat = sin(lat[j] * M_PI/180);
                            double sinLon = sin(lon[j] * M_PI/180);

                            knotsX[region_index[j]][j-i] = cosLat * cosLon;
                            knotsY[region_index[j]][j-i] = cosLat * sinLon;
                            knotsZ[region_index[j]][j-i] = sinLat;
                        }
                    }
                }
                i += NUM_KNOTS_FINEST;
            }
            else break;
        }   
    }

    //Get coordinates of observations/knots and predictions in the regions in the finest level
    int maxOpenMPThreads=omp_get_max_threads();
    
    #pragma omp parallel for num_threads(maxOpenMPThreads) schedule(dynamic,1)
    for(unsigned long iRegion = indexStartFinestLevel; iRegion < nRegionsInTotal; iRegion++)
    {
        unsigned long indexStartThisLevel = iRegion - indexStartFinestLevel;

        double *tmpLon, *tmpLat, *tmpX, *tmpY, *tmpZ, *tmpVal;
        unsigned long *nFinestLevel;
        if(OBS_AS_KNOTS_FLAG)
            nFinestLevel = nKnotsAtFinestLevel;
        else
            nFinestLevel = nObservationsAtFinestLevel;

        std::set<unsigned long> indexObservationsInThisRegion;
        std::set<unsigned long> indexPredictionsInThisRegion;
        
        if(WORKING_REGION_FLAG[iRegion])
        {
            indexObservationsInThisRegion.clear();

            for(unsigned long jObservation = 0; jObservation < NUM_OBSERVATIONS; jObservation++)
            {
                if(data->observationRegion[jObservation] == iRegion)
                    indexObservationsInThisRegion.insert(jObservation);
            }

            nFinestLevel[indexStartThisLevel] = indexObservationsInThisRegion.size();

            if(nFinestLevel[indexStartThisLevel] > 0)
            {
                if(OBS_AS_KNOTS_FLAG)
                {
                    knotsLon[iRegion] = new double [nFinestLevel[indexStartThisLevel]];            
                    knotsLat[iRegion] = new double [nFinestLevel[indexStartThisLevel]];
                    knotsResidual[indexStartThisLevel] = new double [nFinestLevel[indexStartThisLevel]];

                    if(CHORDAL_DISTANCE_FLAG)
                    {
                        knotsX[iRegion] = new double [nFinestLevel[indexStartThisLevel]];
                        knotsY[iRegion] = new double [nFinestLevel[indexStartThisLevel]];      
                        knotsZ[iRegion] = new double [nFinestLevel[indexStartThisLevel]];      

                        tmpX = knotsX[iRegion]; 
                        tmpY = knotsY[iRegion];
                        tmpZ = knotsZ[iRegion];
                    }

                    tmpLon = knotsLon[iRegion];
                    tmpLat = knotsLat[iRegion];
                    tmpVal = knotsResidual[indexStartThisLevel];
                }
                else
                {
                    observationLon[indexStartThisLevel] = new double [nFinestLevel[indexStartThisLevel]];            
                    observationLat[indexStartThisLevel] = new double [nFinestLevel[indexStartThisLevel]];
                    residual[indexStartThisLevel] = new double [nFinestLevel[indexStartThisLevel]];

                    if(CHORDAL_DISTANCE_FLAG)
                    {
                        observationX[indexStartThisLevel] = new double [nFinestLevel[indexStartThisLevel]];
                        observationY[indexStartThisLevel] = new double [nFinestLevel[indexStartThisLevel]];      
                        observationZ[indexStartThisLevel] = new double [nFinestLevel[indexStartThisLevel]];   
                        tmpX = observationX[indexStartThisLevel]; 
                        tmpY = observationY[indexStartThisLevel];
                        tmpZ = observationZ[indexStartThisLevel];   
                    }

                    tmpLon = observationLon[indexStartThisLevel];
                    tmpLat = observationLat[indexStartThisLevel];
                    tmpVal = residual[indexStartThisLevel];
                }

                unsigned long tmpIndex=0;
                for(std::set<unsigned long>::iterator jObservation = indexObservationsInThisRegion.begin();
                jObservation != indexObservationsInThisRegion.end(); jObservation++)
                {
                    tmpLon[tmpIndex] = data->observationLon[*jObservation];
                    tmpLat[tmpIndex] = data->observationLat[*jObservation];
                    tmpVal[tmpIndex] = data->observationResiduals[*jObservation];

                    if(CHORDAL_DISTANCE_FLAG)
                    {
                        double cosLon = cos(tmpLon[tmpIndex] * M_PI/180);
                        double cosLat = cos(tmpLat[tmpIndex] * M_PI/180);
                        double sinLon = sin(tmpLon[tmpIndex] * M_PI/180);
                        double sinLat = sin(tmpLat[tmpIndex] * M_PI/180);
                        
                        tmpX[tmpIndex] = cosLat * cosLon;
                        tmpY[tmpIndex] = cosLat * sinLon;
                        tmpZ[tmpIndex] = sinLat;
                    }
                    tmpIndex++;
                }
            }
            
            if(CALCULATION_MODE=="prediction")
            {
                indexPredictionsInThisRegion.clear();
                for(unsigned long jPrediction = 0; jPrediction < NUM_PREDICTIONS; jPrediction++)
                {
                    if(data->predictionRegion[jPrediction] == iRegion)
                        indexPredictionsInThisRegion.insert(jPrediction);
                }

                nPredictionsAtFinestLevel[indexStartThisLevel] = indexPredictionsInThisRegion.size();

                if(nPredictionsAtFinestLevel[indexStartThisLevel] > 0)
                {
                    predictionLon[indexStartThisLevel] = new double [nPredictionsAtFinestLevel[indexStartThisLevel]];
                    predictionLat[indexStartThisLevel] = new double [nPredictionsAtFinestLevel[indexStartThisLevel]];

                    if(CHORDAL_DISTANCE_FLAG)
                    {
                        predictionX[indexStartThisLevel] = new double [nPredictionsAtFinestLevel[indexStartThisLevel]];
                        predictionY[indexStartThisLevel] = new double [nPredictionsAtFinestLevel[indexStartThisLevel]];
                        predictionZ[indexStartThisLevel] = new double [nPredictionsAtFinestLevel[indexStartThisLevel]];
                    }

                    unsigned long tmpIndex=0;
                    for(std::set<unsigned long>::iterator jPrediction = indexPredictionsInThisRegion.begin();
                    jPrediction != indexPredictionsInThisRegion.end(); jPrediction++)
                    {
                        predictionLon[indexStartThisLevel][tmpIndex] = data->predictionLon[*jPrediction];
                        predictionLat[indexStartThisLevel][tmpIndex] = data->predictionLat[*jPrediction];

                        if(CHORDAL_DISTANCE_FLAG)
                        {
                            double cosLon = cos(predictionLon[indexStartThisLevel][tmpIndex] * M_PI/180);
                            double cosLat = cos(predictionLat[indexStartThisLevel][tmpIndex] * M_PI/180);
                            double sinLon = sin(predictionLon[indexStartThisLevel][tmpIndex] * M_PI/180);
                            double sinLat = sin(predictionLat[indexStartThisLevel][tmpIndex] * M_PI/180);
                        
                            predictionX[indexStartThisLevel][tmpIndex] = cosLat * cosLon;
                            predictionY[indexStartThisLevel][tmpIndex] = cosLat * sinLon;
                            predictionZ[indexStartThisLevel][tmpIndex] = sinLat;
                        }
                        tmpIndex++;
                    }
                }
            }
        }    

        indexObservationsInThisRegion.clear();
        indexPredictionsInThisRegion.clear();
    }
}

/*
Build partitions by this program
*/
void Partition::build_partition_by_program()
{
    if(WORKER == 0)
	{
		gettimeofday(&timeNow, NULL);
		cout<<"===========================> Processor 1: building hierarchical grid starts. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n";
	}

    //The number of knots in each direction
    int nKnotsLon = ceil(sqrt(NUM_KNOTS_r));
    int nKnotsLat = (int)(NUM_KNOTS_r/nKnotsLon);
    nKnots = nKnotsLon*nKnotsLat;

    //Array of coordinates for the partition in each region with dimension [nRegionsInTotal]
    double *partitionLonMin, *partitionLatMin, *partitionLonMax, *partitionLatMax;
    partitionLonMin = new double [nRegionsInTotal];
    partitionLonMax = new double [nRegionsInTotal];
    partitionLatMin = new double [nRegionsInTotal];
    partitionLatMax = new double [nRegionsInTotal];

    partitionLonMin[0] = data->domainBoundaries[0];
    partitionLonMax[0] = data->domainBoundaries[1];
    partitionLatMin[0] = data->domainBoundaries[2];
    partitionLatMax[0] = data->domainBoundaries[3];


    for(int iLevel = 0; iLevel < NUM_LEVELS_M-1; iLevel++)
    {
        #pragma omp parallel for
        for(unsigned long jRegion = REGION_START[iLevel]; jRegion < REGION_END[iLevel] + 1; jRegion++)
            create_partitions(jRegion, partitionLonMin, partitionLonMax, partitionLatMin, partitionLatMax);
    }

    #pragma omp parallel for schedule(dynamic,1)
    for(unsigned long iRegion = 0; iRegion < nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M-1]; iRegion++)
    {
        if(WORKING_REGION_FLAG[iRegion])
        {
            knotsLon[iRegion] = new double [nKnots];
            knotsLat[iRegion] = new double [nKnots];
            create_knots(knotsLon[iRegion], knotsLat[iRegion], partitionLonMin[iRegion], partitionLonMax[iRegion], nKnotsLon, partitionLatMin[iRegion], partitionLatMax[iRegion], nKnotsLat);

            if(CHORDAL_DISTANCE_FLAG)
            {
                knotsX[iRegion] = new double [nKnots];
                knotsY[iRegion] = new double [nKnots];
                knotsZ[iRegion] = new double [nKnots];

                #pragma omp parallel for
                for(int iKnot= 0; iKnot < nKnots; iKnot++)
                {
                    double cosLat = cos(knotsLat[iRegion][iKnot] * M_PI/180);
                    double cosLon = cos(knotsLon[iRegion][iKnot] * M_PI/180);
                    double sinLat = sin(knotsLat[iRegion][iKnot] * M_PI/180);
                    double sinLon = sin(knotsLon[iRegion][iKnot] * M_PI/180);

                    knotsX[iRegion][iKnot] = cosLat * cosLon;
                    knotsY[iRegion][iKnot] = cosLat * sinLon;
                    knotsZ[iRegion][iKnot] = sinLat;
                }
            }
        }
    }

    //Get coordinates of the knots and predictions at the regions in the finest level
    int maxOpenMPThreads=omp_get_max_threads();

    #pragma omp parallel for num_threads(maxOpenMPThreads) schedule(dynamic,1)
    for(unsigned long iRegion = nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M-1]; iRegion < nRegionsInTotal; iRegion++)
    {
        std::set<unsigned long> *indexObservationsInThisRegion = new std::set<unsigned long>;
        std::set<unsigned long> *indexPredictionsInThisRegion = new std::set<unsigned long>;

        unsigned long indexStartThisLevel = iRegion+nRegionsAtEachLevel[NUM_LEVELS_M-1]-nRegionsInTotal;
        if(WORKING_REGION_FLAG[iRegion])
        {
            double lonMin=partitionLonMin[iRegion];
            double lonMax=partitionLonMax[iRegion];
            double latMin=partitionLatMin[iRegion];
            double latMax=partitionLatMax[iRegion];

            indexObservationsInThisRegion->clear();

            for(unsigned long jObservation = 0; jObservation < NUM_OBSERVATIONS; jObservation++)
            {
                if(data->observationLon[jObservation] >= lonMin && data->observationLon[jObservation] < lonMax && data->observationLat[jObservation] >= latMin && data->observationLat[jObservation] < latMax)
                {
                    
                    bool skip = false;
                    //Skip if the observation location coincides with knots in levels above
                    unsigned long ancestor = iRegion;
                    for(int kLevel = 0; kLevel < NUM_LEVELS_M-1; kLevel++)
                    {
                        ancestor = (ancestor-1)/NUM_PARTITIONS_J;
                        for(int pKnot = 0; pKnot < nKnots; pKnot++)
                        {
                            if(knotsLon[ancestor][pKnot] == data->observationLon[jObservation] && knotsLat[ancestor][pKnot] == data->observationLat[jObservation])
                            {
                                skip = true;
                                break;
                            }
                        }
                        if(skip) break;
                    }
                    
                    //Add this observation
                    if(!skip) indexObservationsInThisRegion->insert(jObservation);
                }
            }

            nKnotsAtFinestLevel[indexStartThisLevel] = indexObservationsInThisRegion->size();
            knotsLon[iRegion] = new double [nKnotsAtFinestLevel[indexStartThisLevel]];            
            knotsLat[iRegion] = new double [nKnotsAtFinestLevel[indexStartThisLevel]];
            if(CHORDAL_DISTANCE_FLAG)
            {
                knotsX[iRegion] = new double [nKnotsAtFinestLevel[indexStartThisLevel]];     
                knotsY[iRegion] = new double [nKnotsAtFinestLevel[indexStartThisLevel]];     
                knotsZ[iRegion] = new double [nKnotsAtFinestLevel[indexStartThisLevel]];     
            }

            knotsResidual[indexStartThisLevel] = new double [nKnotsAtFinestLevel[indexStartThisLevel]];
       
            unsigned long tmpIndex=0;
            for(std::set<unsigned long>::iterator jObservation = indexObservationsInThisRegion->begin(); jObservation != indexObservationsInThisRegion->end(); jObservation++)
            {
                knotsLon[iRegion][tmpIndex] = data->observationLon[*jObservation];
                knotsLat[iRegion][tmpIndex] = data->observationLat[*jObservation];
                knotsResidual[indexStartThisLevel][tmpIndex] = data->observationResiduals[*jObservation];

                if(CHORDAL_DISTANCE_FLAG)
                {
                    double cosLat = cos(data->observationLat[*jObservation] * M_PI/180);
                    double cosLon = cos(data->observationLon[*jObservation] * M_PI/180);
                    double sinLat = sin(data->observationLat[*jObservation] * M_PI/180);
                    double sinLon = sin(data->observationLon[*jObservation] * M_PI/180);

                    knotsX[iRegion][tmpIndex] = cosLat * cosLon;
                    knotsY[iRegion][tmpIndex] = cosLat * sinLon;
                    knotsZ[iRegion][tmpIndex] = sinLat;
                }

                tmpIndex++;
            }

            if(CALCULATION_MODE=="prediction")
            {
                indexPredictionsInThisRegion->clear();
                for(unsigned long jPrediction = 0; jPrediction < NUM_PREDICTIONS; jPrediction++)
                {
                    if(data->predictionLon[jPrediction] >= lonMin && data->predictionLon[jPrediction] < lonMax && data->predictionLat[jPrediction] >= latMin && data->predictionLat[jPrediction] < latMax)
                        indexPredictionsInThisRegion->insert(jPrediction);
                }

                nPredictionsAtFinestLevel[indexStartThisLevel] = indexPredictionsInThisRegion->size();
                predictionLon[indexStartThisLevel] = new double [nPredictionsAtFinestLevel[indexStartThisLevel]];            
                predictionLat[indexStartThisLevel] = new double [nPredictionsAtFinestLevel[indexStartThisLevel]];
                if(CHORDAL_DISTANCE_FLAG)
                {
                    predictionX[indexStartThisLevel] = new double [nPredictionsAtFinestLevel[indexStartThisLevel]];     
                    predictionY[indexStartThisLevel] = new double [nPredictionsAtFinestLevel[indexStartThisLevel]];
                    predictionZ[indexStartThisLevel] = new double [nPredictionsAtFinestLevel[indexStartThisLevel]];
                }

                unsigned long tmpIndex=0;
                for(std::set<unsigned long>::iterator jPrediction = indexPredictionsInThisRegion->begin(); jPrediction != indexPredictionsInThisRegion->end(); jPrediction++)
                {
                    predictionLon[indexStartThisLevel][tmpIndex] = data->predictionLon[*jPrediction];
                    predictionLat[indexStartThisLevel][tmpIndex] = data->predictionLat[*jPrediction];
                    
                    if(CHORDAL_DISTANCE_FLAG)
                    {
                        double cosLat = cos(data->predictionLat[*jPrediction] * M_PI/180);
                        double cosLon = cos(data->predictionLon[*jPrediction] * M_PI/180);
                        double sinLat = sin(data->predictionLat[*jPrediction] * M_PI/180);
                        double sinLon = sin(data->predictionLon[*jPrediction] * M_PI/180);

                        predictionX[indexStartThisLevel][tmpIndex] = cosLat * cosLon;
                        predictionY[indexStartThisLevel][tmpIndex] = cosLat * sinLon;
                        predictionZ[indexStartThisLevel][tmpIndex] = sinLat;
                    }

                    tmpIndex++;
                }
            }
        }    
        indexObservationsInThisRegion->clear();
        indexPredictionsInThisRegion->clear();
    }
    

    //Release the memory for the temporary arrays
    delete[] partitionLonMin;
    delete[] partitionLatMin;
    delete[] partitionLonMax;
    delete[] partitionLatMax;
}

/*
Read parent region indices of the regions in the finest level.
*/
void Partition::read_parent_region_index()
{
    if(WORKER == 0)
	{
		gettimeofday(&timeNow, NULL);
		cout<<"===========================> Processor 1: reading parent indices from "<<PARENT_REGION_FILE_NAME<<" starts. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n";
	}

    //Check data file exists
    struct stat fileStatus;
    if(stat(PARENT_REGION_FILE_NAME.c_str(),&fileStatus) == -1 || !S_ISREG(fileStatus.st_mode))
    {
        if(WORKER == 0) cout<<"Program exits with an error: the data file "<<PARENT_REGION_FILE_NAME<<" does not exist.\n";
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    //Open the data file by DATA_FILE_NAME
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD,PARENT_REGION_FILE_NAME.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&file);

    MPI_Status* readStatus = new MPI_Status;
    int count = -1;

    //Get the minimum region index in the finest level
    unsigned long regionStart = 0, regionEnd = 0;
    MPI_File_read(file,(void*)&regionStart,1,MPI_UNSIGNED_LONG,readStatus);
    MPI_Get_count(readStatus,MPI_UNSIGNED_LONG,&count);

    if(count!=1)
    {
        if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<PARENT_REGION_FILE_NAME<<" for the minimum region index in the finest level. Double check the data file format."<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    //Get the maximum region index in the finest level
    MPI_File_read(file,(void*)&regionEnd,1,MPI_UNSIGNED_LONG,readStatus);
    MPI_Get_count(readStatus,MPI_UNSIGNED_LONG,&count);

    if(count!=1)
    {
        if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<PARENT_REGION_FILE_NAME<<" for the maximum region index in the finest level. Double check the data file format."<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    nRegionsAtFinestLevel = regionEnd - regionStart + 1;
    nRegionsAtEachLevel[NUM_LEVELS_M - 1] = nRegionsAtFinestLevel;
    indexStartFinestLevel = regionStart;

    //Get parent region indices
    parent = new unsigned long [nRegionsAtFinestLevel];
    int interval=INT_MAX;
    unsigned long positionStart = 0, positionEnd = positionStart + interval;
    while(positionStart<nRegionsAtFinestLevel)
    {
        if(positionEnd>nRegionsAtFinestLevel) interval = nRegionsAtFinestLevel - positionStart;
        
        MPI_File_read(file,(void*)&parent[positionStart],interval,MPI_UNSIGNED_LONG,readStatus);
        MPI_Get_count(readStatus,MPI_UNSIGNED_LONG,&count);

        positionStart = positionEnd;
        positionEnd += interval;

        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<PARENT_REGION_FILE_NAME<<" for parent region indices of the regions in the finest level. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Close the data file
    MPI_File_close(&file);

    if(WORKER == 0)
	{
		gettimeofday(&timeNow, NULL);
		cout<<"===========================> Processor 1: reading parent indices from "<<PARENT_REGION_FILE_NAME<<" is complete. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n";
	}

    //Calculate childrenStart and childrenEnd
    childrenStart = new unsigned long [indexStartFinestLevel - indexStartSecondFinestLevel]();
    childrenEnd = new unsigned long [indexStartFinestLevel - indexStartSecondFinestLevel]();

    for(unsigned long iRegion = 0 ; iRegion < nRegionsAtFinestLevel; iRegion++)
    {
        unsigned long parentIndexStartLevel = parent[iRegion] - indexStartSecondFinestLevel;
        unsigned long currentRegionIndex = iRegion + indexStartFinestLevel;
        
        if(childrenStart[parentIndexStartLevel] == 0)
            childrenStart[parentIndexStartLevel] = currentRegionIndex;
        else
        {
            if(childrenStart[parentIndexStartLevel] > currentRegionIndex)
                childrenStart[parentIndexStartLevel] = currentRegionIndex; 
        }
        
        if(childrenEnd[parentIndexStartLevel] == 0)
            childrenEnd[parentIndexStartLevel] = currentRegionIndex;
        else
        {
            if(childrenEnd[parentIndexStartLevel] < currentRegionIndex)
                childrenEnd[parentIndexStartLevel] = currentRegionIndex;
        }
    }
}

/////////////////// Above are private member functions /////////////////

/////////////////// Below are public member functions /////////////////

/*Implement the constructor of Partition. Assign values to nRegionsAtEachLevel for the array of the number of regions in each level, nRegionsInTotal for the total number of regions, and MPI quantities.
*/
Partition::Partition()
{

    //If user did not predetermine the number of levels, find the number of levels such that the average number of observations per region similar to the number of knots
    if(NUM_LEVELS_M == -99)//-99 stands for the default method to automatically determine NUM_LEVELS_M 
    {
        //Find the number of regions required at finest level, nRegionsAtFinestLevel 
        unsigned long nRegionsAtFinestLevel = ceil((double)NUM_OBSERVATIONS/(double)NUM_KNOTS_r);

        //Find the number of levels, NUM_LEVELS_M, which satisfies NUM_PARTITIONS_J^(NUM_LEVELS_M-1)>=nRegionsAtFinestLevel
        NUM_LEVELS_M = ceil(log(nRegionsAtFinestLevel)/log(NUM_PARTITIONS_J))+1;

        if(NUM_LEVELS_M < 2)
	    {
            if(WORKER == 0) cout<<"Program exits with an error: the NUM_LEVELS_M calculated by default is "<<NUM_LEVELS_M<<", which is required to be larger than 1. Please decrease NUM_KNOTS_r to get a larger NUM_LEVELS_M.\n";
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
	    }
    }

    //Find the array of the number of regions in each level, nRgnsVec, and the total number of regions, nRegionsInTotal
    nRegionsAtEachLevel = new unsigned long [NUM_LEVELS_M];
    nRegionsAtEachLevel[0] = 1;
    nRegionsInTotal = 1;
    for(int iLevel = 1; iLevel < NUM_LEVELS_M - 1; iLevel++)
    {
        nRegionsAtEachLevel[iLevel] = nRegionsAtEachLevel[iLevel-1]*NUM_PARTITIONS_J;
        nRegionsInTotal += nRegionsAtEachLevel[iLevel];
    }

    indexStartSecondFinestLevel = nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M - 2];

    if(VARIOUS_J_FINEST_LEVEL_FLAG)
    {
        read_parent_region_index();
        nRegionsInTotal += nRegionsAtFinestLevel;
    }
    else
    {
        indexStartFinestLevel = nRegionsInTotal;
        nRegionsAtFinestLevel = nRegionsAtEachLevel[NUM_LEVELS_M -2 ] * NUM_PARTITIONS_J;
        nRegionsAtEachLevel[NUM_LEVELS_M - 1] = nRegionsAtFinestLevel;
        nRegionsInTotal += nRegionsAtFinestLevel;
    }

    //Calculate MPI information
        WORKING_REGION_FLAG = new bool [nRegionsInTotal]();

        unsigned long indexRegionAtFinestLevelStartThisWorker, indexRegionAtFinestLevelEndThisWorker;
        WORKERS_FOR_EACH_REGION = new std::set<unsigned long>[nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M-1]];

        //Assign regions in the finest level to the worker
        unsigned long nRegionsPerWorker = nRegionsAtEachLevel[NUM_LEVELS_M-1] / MPI_SIZE;
        unsigned long nWorkersWithAdditionalRegion = nRegionsAtEachLevel[NUM_LEVELS_M-1] - nRegionsPerWorker * MPI_SIZE;

        if(WORKER < nWorkersWithAdditionalRegion)
        {
            indexRegionAtFinestLevelStartThisWorker =  WORKER * (nRegionsPerWorker+1);
            indexRegionAtFinestLevelEndThisWorker =  indexRegionAtFinestLevelStartThisWorker + nRegionsPerWorker;
        }
        else
        {
            indexRegionAtFinestLevelStartThisWorker =  nWorkersWithAdditionalRegion * (nRegionsPerWorker+1) + (WORKER -  nWorkersWithAdditionalRegion)* nRegionsPerWorker;
            indexRegionAtFinestLevelEndThisWorker =  indexRegionAtFinestLevelStartThisWorker + nRegionsPerWorker - 1;
        }

        indexRegionAtFinestLevelStartThisWorker += nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M-1];
        indexRegionAtFinestLevelEndThisWorker += nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M-1];

        for(unsigned long i = indexRegionAtFinestLevelStartThisWorker; i < indexRegionAtFinestLevelEndThisWorker+1; i++)
            WORKING_REGION_FLAG[i] = true;

        //Find the regions dealt with by this worker at the second finest level
        if(VARIOUS_J_FINEST_LEVEL_FLAG)
        {
            for(unsigned long index = indexRegionAtFinestLevelStartThisWorker; index <= indexRegionAtFinestLevelEndThisWorker; index++)
            {
                unsigned long ancestor = parent[index - indexStartFinestLevel];
                WORKING_REGION_FLAG[ancestor] = true;
                INDICES_REGIONS_AT_CURRENT_LEVEL.insert(ancestor);
                while(ancestor > 0)
                {
                    ancestor = (ancestor-1)/NUM_PARTITIONS_J;
                    WORKING_REGION_FLAG[ancestor] = true;
                }
            }
        }
        else
        {
            for(unsigned long index = indexRegionAtFinestLevelStartThisWorker; index <= indexRegionAtFinestLevelEndThisWorker; index++)
            {
                unsigned long ancestor = index;
                while(ancestor > 0)
                {
                    ancestor = (ancestor-1)/NUM_PARTITIONS_J;
                    WORKING_REGION_FLAG[ancestor] = true;
                }
                INDICES_REGIONS_AT_CURRENT_LEVEL.insert((index-1)/NUM_PARTITIONS_J);
            }
        }

        //Assign WORKERS_FOR_EACH_REGION at the second finest level
        unsigned long tmpIndexStart = indexStartFinestLevel, tmpIndexEnd;
        if(DYNAMIC_SCHEDULE_FLAG) origninalWorker = new int [nRegionsInTotal];

        for(int iWorker = 0; iWorker < nWorkersWithAdditionalRegion; iWorker++)
        {
            tmpIndexEnd =  tmpIndexStart + nRegionsPerWorker + 1;
            for(unsigned long index = tmpIndexStart; index < tmpIndexEnd; index++)
            {
                unsigned long ancestor;
                
                if (VARIOUS_J_FINEST_LEVEL_FLAG) 
                    ancestor = parent[index - indexStartFinestLevel];
                else
                    ancestor = (index-1)/NUM_PARTITIONS_J;
                
                WORKERS_FOR_EACH_REGION[ancestor].insert(iWorker);
                if(DYNAMIC_SCHEDULE_FLAG) 
                {
                    origninalWorker[index] = iWorker;
                    origninalWorker[ancestor] = iWorker;
                    while(ancestor > 0)
                    {
                        ancestor = (ancestor-1)/NUM_PARTITIONS_J;
                        origninalWorker[ancestor] = iWorker;
                    }
                }
            }
            tmpIndexStart += nRegionsPerWorker+1;
            tmpIndexEnd += nRegionsPerWorker+1;
        }

        for(int iWorker = nWorkersWithAdditionalRegion; iWorker < MPI_SIZE; iWorker++)
        {
            tmpIndexEnd =  tmpIndexStart + nRegionsPerWorker;
            for(unsigned long index = tmpIndexStart; index < tmpIndexEnd; index++)
            {
                unsigned long ancestor;

                if (VARIOUS_J_FINEST_LEVEL_FLAG) 
                    ancestor = parent[index - indexStartFinestLevel];
                else
                    ancestor = (index-1)/NUM_PARTITIONS_J;

                WORKERS_FOR_EACH_REGION[ancestor].insert(iWorker);
                if(DYNAMIC_SCHEDULE_FLAG) 
                {
                    origninalWorker[index] = iWorker;
                    origninalWorker[ancestor] = iWorker;
                    while(ancestor > 0)
                    {
                        ancestor = (ancestor-1)/NUM_PARTITIONS_J;
                        origninalWorker[ancestor] = iWorker;
                    }
                }
            }
            tmpIndexStart += nRegionsPerWorker;
            tmpIndexEnd += nRegionsPerWorker;
        }

        //Assign regionStart and regionEnd for each level
        REGION_START = new unsigned long [NUM_LEVELS_M];
        REGION_END = new unsigned long [NUM_LEVELS_M];
        
        REGION_START[NUM_LEVELS_M-1]=indexRegionAtFinestLevelStartThisWorker;
        REGION_END[NUM_LEVELS_M-1]=indexRegionAtFinestLevelEndThisWorker;
        
        if(VARIOUS_J_FINEST_LEVEL_FLAG)
        {
            REGION_START[NUM_LEVELS_M-2] = parent[REGION_START[NUM_LEVELS_M-1]-indexStartFinestLevel];
            REGION_END[NUM_LEVELS_M-2]   = parent[REGION_END[NUM_LEVELS_M-1]-indexStartFinestLevel];
        }
        else
        {
            REGION_START[NUM_LEVELS_M-2] = (REGION_START[NUM_LEVELS_M-1]-1)/NUM_PARTITIONS_J;
            REGION_END[NUM_LEVELS_M-2]   = (REGION_END[NUM_LEVELS_M-1]-1)/NUM_PARTITIONS_J;
        }

        for(int iLevel = NUM_LEVELS_M-3; iLevel > -1; iLevel--)
        {
            REGION_START[iLevel] = (REGION_START[iLevel+1]-1)/NUM_PARTITIONS_J;
            REGION_END[iLevel] = (REGION_END[iLevel+1]-1)/NUM_PARTITIONS_J;
        }
}

/*
Implement the member function of Partition, build_partition, which builds the hierarchical partition
*/

void Partition::build_partition()
{
    //Allocate memory for the coordinates of knots
    knotsLon = new double* [nRegionsInTotal];
    knotsLat = new double* [nRegionsInTotal];
    predictionLon = new double* [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
    predictionLat = new double* [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
    nPredictionsAtFinestLevel = new unsigned long [nRegionsAtEachLevel[NUM_LEVELS_M-1]];

    knotsX = new double* [nRegionsInTotal];
    knotsY = new double* [nRegionsInTotal];
    knotsZ = new double* [nRegionsInTotal];

    predictionX = new double* [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
    predictionY = new double* [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
    predictionZ = new double* [nRegionsAtEachLevel[NUM_LEVELS_M-1]];

    //Allocate memory
    if(!OBS_AS_KNOTS_FLAG)
    {
        nObservationsAtFinestLevel = new unsigned long [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
        residual = new double* [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
        observationLon = new double* [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
        observationLat = new double* [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
        if(CHORDAL_DISTANCE_FLAG)
        {
            observationX = new double* [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
            observationY = new double* [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
            observationZ = new double* [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
        }
    }
    else 
    {
        nKnotsAtFinestLevel = new unsigned long [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
        knotsResidual = new double* [nRegionsAtEachLevel[NUM_LEVELS_M-1]];
    }

    if(PROVIDED_KNOTS_FLAG)
        build_partition_by_user_files();
    else
        build_partition_by_program();

    //Delete variables that are not used anymore
    delete[] data->observationLon;
    delete[] data->observationLat;
    delete[] data->observationResiduals;

    if(PROVIDED_KNOTS_FLAG) delete[] data->observationRegion;

    if(CALCULATION_MODE == "prediction")
    {
        delete[] data->predictionLon;
        delete[] data->predictionLat;
        if(PROVIDED_KNOTS_FLAG) delete[] data->predictionRegion;
    }

    if(DYNAMIC_SCHEDULE_FLAG) 
		dynamic_schedule();
	else
		WORLD = MPI_COMM_WORLD;

    if(WORKER == 0)
	{
		gettimeofday(&timeNow, NULL);
		cout<<"===========================> Processor 1: building hierarchical grid is complete. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n\n";
		if(PRINT_DETAIL_FLAG) print_partition_summary();
	}
}

//Implement the member function of Partition, dynamic_schedule, which assign the work load for each worker by dynamic scheduling
void Partition::dynamic_schedule()
{
    //Synchronize the knots in regions before the finest level
    for(unsigned long iRegion = 0; iRegion < nRegionsInTotal - nRegionsAtFinestLevel; iRegion++)
    {
        if(!WORKING_REGION_FLAG[iRegion])
        {
            knotsLon[iRegion] = new double [nKnots];
            knotsLat[iRegion] = new double [nKnots];

            if(CHORDAL_DISTANCE_FLAG)
            {
                knotsX[iRegion] = new double [nKnots];
                knotsY[iRegion] = new double [nKnots];
                knotsZ[iRegion] = new double [nKnots];
            }
        }

        MPI_Bcast(knotsLon[iRegion],nKnots,MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
        MPI_Bcast(knotsLat[iRegion],nKnots,MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);

        if(CHORDAL_DISTANCE_FLAG)
        {
            MPI_Bcast(knotsX[iRegion],nKnots,MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
            MPI_Bcast(knotsY[iRegion],nKnots,MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
            MPI_Bcast(knotsZ[iRegion],nKnots,MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
        }
    }

    //Synchronize the knots in regions at the finest level
    for(unsigned long iRegion = indexStartFinestLevel; iRegion < nRegionsInTotal; iRegion++)
    {
        if(OBS_AS_KNOTS_FLAG)
        {
            MPI_Bcast(&nKnotsAtFinestLevel[iRegion-indexStartFinestLevel],1,MPI_UNSIGNED_LONG,origninalWorker[iRegion],MPI_COMM_WORLD);

            if(!WORKING_REGION_FLAG[iRegion])
            {
                knotsLon[iRegion] = new double [nKnotsAtFinestLevel[iRegion-indexStartFinestLevel]];
                knotsLat[iRegion] = new double [nKnotsAtFinestLevel[iRegion-indexStartFinestLevel]];
                knotsResidual[iRegion-indexStartFinestLevel] = new double [nKnotsAtFinestLevel[iRegion-indexStartFinestLevel]];

                if(CHORDAL_DISTANCE_FLAG)
                {
                    knotsX[iRegion] = new double [nKnotsAtFinestLevel[iRegion-indexStartFinestLevel]];
                    knotsY[iRegion] = new double [nKnotsAtFinestLevel[iRegion-indexStartFinestLevel]];
                    knotsZ[iRegion] = new double [nKnotsAtFinestLevel[iRegion-indexStartFinestLevel]];
                }
            }

            MPI_Bcast(knotsLon[iRegion],nKnotsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
            MPI_Bcast(knotsLat[iRegion],nKnotsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
            MPI_Bcast(knotsResidual[iRegion-indexStartFinestLevel],nKnotsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);

            if(CHORDAL_DISTANCE_FLAG)
            {
                MPI_Bcast(knotsX[iRegion],nKnotsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
                MPI_Bcast(knotsY[iRegion],nKnotsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
                MPI_Bcast(knotsZ[iRegion],nKnotsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
            }
        }else
        {
            if(!WORKING_REGION_FLAG[iRegion])
            {
                knotsLon[iRegion] = new double [NUM_KNOTS_FINEST];
                knotsLat[iRegion] = new double [NUM_KNOTS_FINEST];
            }
            MPI_Bcast(knotsLon[iRegion],NUM_KNOTS_FINEST,MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
            MPI_Bcast(knotsLat[iRegion],NUM_KNOTS_FINEST,MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
        }
    }

    //Synchronize the observations in regions at the finest level if observations are not used as knots
    if(!OBS_AS_KNOTS_FLAG)
    {
        for(unsigned long iRegion = indexStartFinestLevel; iRegion < nRegionsInTotal; iRegion++)
        {
            MPI_Bcast(&nObservationsAtFinestLevel[iRegion-indexStartFinestLevel],1,MPI_UNSIGNED_LONG,origninalWorker[iRegion],MPI_COMM_WORLD);

            if(!WORKING_REGION_FLAG[iRegion])
            {
                observationLon[iRegion-indexStartFinestLevel] = new double [nObservationsAtFinestLevel[iRegion-indexStartFinestLevel]];
                observationLat[iRegion-indexStartFinestLevel] = new double [nObservationsAtFinestLevel[iRegion-indexStartFinestLevel]];
                residual[iRegion-indexStartFinestLevel] = new double [nObservationsAtFinestLevel[iRegion-indexStartFinestLevel]];

                if(CHORDAL_DISTANCE_FLAG)
                {
                    observationX[iRegion-indexStartFinestLevel] = new double [nObservationsAtFinestLevel[iRegion-indexStartFinestLevel]];
                    observationY[iRegion-indexStartFinestLevel] = new double [nObservationsAtFinestLevel[iRegion-indexStartFinestLevel]];
                    observationZ[iRegion-indexStartFinestLevel] = new double [nObservationsAtFinestLevel[iRegion-indexStartFinestLevel]];
                }
            }

            MPI_Bcast(observationLon[iRegion-indexStartFinestLevel],nObservationsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
            MPI_Bcast(observationLat[iRegion-indexStartFinestLevel],nObservationsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
            MPI_Bcast(residual[iRegion-indexStartFinestLevel],nObservationsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);

            if(CHORDAL_DISTANCE_FLAG)
            {
                MPI_Bcast(observationX[iRegion-indexStartFinestLevel],nObservationsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
                MPI_Bcast(observationY[iRegion-indexStartFinestLevel],nObservationsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
                MPI_Bcast(observationZ[iRegion-indexStartFinestLevel],nObservationsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
            }
        }
    }

    //Synchronize prediction locations in regions at the finest level if predicting
    if(CALCULATION_MODE=="prediction")
    {
        for(unsigned long iRegion = indexStartFinestLevel; iRegion < nRegionsInTotal; iRegion++)
        {
            MPI_Bcast(&nPredictionsAtFinestLevel[iRegion-indexStartFinestLevel],1,MPI_UNSIGNED_LONG,origninalWorker[iRegion],MPI_COMM_WORLD);

            if(!WORKING_REGION_FLAG[iRegion])
            {
                predictionLon[iRegion-indexStartFinestLevel] = new double [nPredictionsAtFinestLevel[iRegion-indexStartFinestLevel]];
                predictionLat[iRegion-indexStartFinestLevel] = new double [nPredictionsAtFinestLevel[iRegion-indexStartFinestLevel]];
            }

            MPI_Bcast(predictionLon[iRegion-indexStartFinestLevel],nPredictionsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
            MPI_Bcast(predictionLat[iRegion-indexStartFinestLevel],nPredictionsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);

            if(CHORDAL_DISTANCE_FLAG)
            {
                if(!WORKING_REGION_FLAG[iRegion])
                {
                    predictionX[iRegion-indexStartFinestLevel] = new double [nPredictionsAtFinestLevel[iRegion-indexStartFinestLevel]];
                    predictionY[iRegion-indexStartFinestLevel] = new double [nPredictionsAtFinestLevel[iRegion-indexStartFinestLevel]];
                    predictionZ[iRegion-indexStartFinestLevel] = new double [nPredictionsAtFinestLevel[iRegion-indexStartFinestLevel]];
                }

                MPI_Bcast(predictionX[iRegion-indexStartFinestLevel],nPredictionsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
                MPI_Bcast(predictionY[iRegion-indexStartFinestLevel],nPredictionsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
                MPI_Bcast(predictionZ[iRegion-indexStartFinestLevel],nPredictionsAtFinestLevel[iRegion-indexStartFinestLevel],MPI_DOUBLE,origninalWorker[iRegion],MPI_COMM_WORLD);
            }
        }
    }
    //Dynamic schedule work load for each worker
    
    ///Clear WORKERS_FOR_EACH_REGION at the second finest level
    for(unsigned long iRegion = nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M-1] - nRegionsAtEachLevel[NUM_LEVELS_M-2]; iRegion < nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M-1]; iRegion++)
        WORKERS_FOR_EACH_REGION[iRegion].clear();

    ///Clear WORKING_REGION_FLAG for all regions
    memset(WORKING_REGION_FLAG,false,nRegionsInTotal);
    
    ///Clear INDICES_REGIONS_AT_CURRENT_LEVEL
    INDICES_REGIONS_AT_CURRENT_LEVEL.clear();

    ///Calculate work load
    double totalComplexity = 0;
    unsigned long *array;

    if(OBS_AS_KNOTS_FLAG)
        array = nKnotsAtFinestLevel;
    else
        array = nObservationsAtFinestLevel;
    
    for(unsigned long index = 0; index < nRegionsAtFinestLevel; index++)
    {
        double tmp = array[index];
        totalComplexity += tmp*tmp;//*tmp;
    }
    double complexityEachWorker = totalComplexity/MPI_SIZE;

    ///Assign working regions for each worker
    REGION_START[NUM_LEVELS_M-1]=nRegionsInTotal;
    REGION_END[NUM_LEVELS_M-1]=0;
    int thisWorker=-1;
    double complexityThisWorker=0;  
    unsigned long indexRegionAtFinestLevelStartThisWorker = nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M-1], indexRegionAtFinestLevelEndThisWorker;
    for(unsigned long index = 0; index < nRegionsAtFinestLevel; index++)
    {
        double tmp = array[index];
        double complexityThisRegion = tmp*tmp;//*tmp;
        complexityThisWorker += complexityThisRegion;

        if(complexityThisWorker >= complexityEachWorker || index == nRegionsAtEachLevel[NUM_LEVELS_M-1]-1)
        {
            thisWorker++;
            if(thisWorker == MPI_SIZE) thisWorker = MPI_SIZE-1;

            indexRegionAtFinestLevelEndThisWorker = indexStartFinestLevel+index;
            
            //Check whether this region should be handled by this worker or next worker
            if(complexityEachWorker - complexityThisWorker + complexityThisRegion < complexityThisWorker - complexityEachWorker 
                && index != nRegionsAtEachLevel[NUM_LEVELS_M-1]-1
                && indexRegionAtFinestLevelEndThisWorker > indexRegionAtFinestLevelStartThisWorker)
            {
                //Next worker handle this region
                indexRegionAtFinestLevelEndThisWorker--;
                complexityThisWorker = complexityThisRegion;
            }
            else
            {
                //This worker handle this region
                complexityThisWorker = 0;
            }

            // if(WORKER==0) cout<<thisWorker<<" "<<indexRegionAtFinestLevelStartThisWorker<<" "<<indexRegionAtFinestLevelEndThisWorker<<endl;

            for(unsigned long jRegion = indexRegionAtFinestLevelStartThisWorker; jRegion < indexRegionAtFinestLevelEndThisWorker+1; jRegion++)
            {
                unsigned long ancestor;

                if(VARIOUS_J_FINEST_LEVEL_FLAG)
                    ancestor = parent[jRegion - indexStartFinestLevel];
                else
                    ancestor = (jRegion-1)/NUM_PARTITIONS_J;
                
                WORKERS_FOR_EACH_REGION[ancestor].insert(thisWorker);
            }

            if(thisWorker == WORKER)
            {
                for(unsigned long jRegion = indexRegionAtFinestLevelStartThisWorker; jRegion < indexRegionAtFinestLevelEndThisWorker+1; jRegion++)
                {
                    unsigned long ancestor = jRegion;
                    WORKING_REGION_FLAG[ancestor] = true;

                    if(VARIOUS_J_FINEST_LEVEL_FLAG)
                        ancestor = parent[jRegion - indexStartFinestLevel];
                    else
                        ancestor = (jRegion-1)/NUM_PARTITIONS_J;
                    
                    INDICES_REGIONS_AT_CURRENT_LEVEL.insert(ancestor);
                    WORKING_REGION_FLAG[ancestor] = true;

                    while(ancestor > 0)
                    {
                        ancestor = (ancestor-1)/NUM_PARTITIONS_J;
                        WORKING_REGION_FLAG[ancestor] = true;
                    }
                    
                }
                if(indexRegionAtFinestLevelStartThisWorker<REGION_START[NUM_LEVELS_M-1]) REGION_START[NUM_LEVELS_M-1]=indexRegionAtFinestLevelStartThisWorker;
                if(indexRegionAtFinestLevelEndThisWorker>REGION_END[NUM_LEVELS_M-1]) REGION_END[NUM_LEVELS_M-1]=indexRegionAtFinestLevelEndThisWorker;
            }

            indexRegionAtFinestLevelStartThisWorker = indexRegionAtFinestLevelEndThisWorker+1;
        }
    }

    //Deactivate MPI processes that are not assigned any regions, which rarely happens
    if(thisWorker<MPI_SIZE-1) 
    {
        MPI_Group worldGroup;
        MPI_Comm_group(MPI_COMM_WORLD,&worldGroup);

        MPI_Group newWorldGroup;
        int ranges[1][3]={{thisWorker+1,MPI_SIZE-1,1}};
        MPI_Group_range_excl(worldGroup,1,ranges,&newWorldGroup);

        MPI_Comm_create(MPI_COMM_WORLD, newWorldGroup, &WORLD);

        MPI_SIZE=thisWorker+1;

        if(WORKER>thisWorker)
        {
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        }
    }
    else
        WORLD = MPI_COMM_WORLD;
    
    if(VARIOUS_J_FINEST_LEVEL_FLAG)
    {
        REGION_START[NUM_LEVELS_M-2] = parent[REGION_START[NUM_LEVELS_M-1]-indexStartFinestLevel];
        REGION_END[NUM_LEVELS_M-2]   = parent[REGION_END[NUM_LEVELS_M-1]-indexStartFinestLevel];
    }
    else
    {
        REGION_START[NUM_LEVELS_M-2] = (REGION_START[NUM_LEVELS_M-1]-1)/NUM_PARTITIONS_J;
        REGION_END[NUM_LEVELS_M-2]   = (REGION_END[NUM_LEVELS_M-1]-1)/NUM_PARTITIONS_J;
    }

    for(int iLevel = NUM_LEVELS_M-3; iLevel > -1; iLevel--)
    {
        REGION_START[iLevel] = (REGION_START[iLevel+1]-1)/NUM_PARTITIONS_J;
        REGION_END[iLevel] = (REGION_END[iLevel+1]-1)/NUM_PARTITIONS_J;
    }

    //Delete regions that are not handled by this WORKER
    for(unsigned long iRegion = 0; iRegion < nRegionsInTotal-nRegionsAtEachLevel[NUM_LEVELS_M-1]; iRegion++)
    {
        if(!WORKING_REGION_FLAG[iRegion])
        {
            delete[] knotsLon[iRegion]; knotsLon[iRegion]=NULL;
            delete[] knotsLat[iRegion]; knotsLat[iRegion]=NULL;

            if(CHORDAL_DISTANCE_FLAG)
            {
                delete[] knotsX[iRegion]; knotsX[iRegion]=NULL;
                delete[] knotsY[iRegion]; knotsY[iRegion]=NULL;
                delete[] knotsZ[iRegion]; knotsZ[iRegion]=NULL;
            }
        }
    }

    for(unsigned long iRegion = nRegionsInTotal - nRegionsAtEachLevel[NUM_LEVELS_M-1]; iRegion < nRegionsInTotal; iRegion++)
    {
        if(!WORKING_REGION_FLAG[iRegion])
        {
            if(OBS_AS_KNOTS_FLAG)
            {
                if(nKnotsAtFinestLevel[iRegion-indexStartFinestLevel] > 0)
                {
                    delete[] knotsResidual[iRegion-indexStartFinestLevel];
                    knotsResidual[iRegion-indexStartFinestLevel]=NULL;

                    delete[] knotsLon[iRegion]; knotsLon[iRegion]=NULL;
                    delete[] knotsLat[iRegion]; knotsLat[iRegion]=NULL;

                    if(CHORDAL_DISTANCE_FLAG)
                    {
                        delete[] knotsX[iRegion]; knotsX[iRegion]=NULL;
                        delete[] knotsY[iRegion]; knotsY[iRegion]=NULL;
                        delete[] knotsZ[iRegion]; knotsZ[iRegion]=NULL;
                    }
                }
            }else
            {
                if(nObservationsAtFinestLevel[iRegion-indexStartFinestLevel] > 0)
                {
                    delete[] residual[iRegion-indexStartFinestLevel];
                    residual[iRegion-indexStartFinestLevel]=NULL;

                    delete[] observationLon[iRegion-indexStartFinestLevel];
                    observationLon[iRegion-indexStartFinestLevel]=NULL;

                    delete[] observationLat[iRegion-indexStartFinestLevel];
                    observationLat[iRegion-indexStartFinestLevel]=NULL;

                    if(CHORDAL_DISTANCE_FLAG)
                    {
                        delete[] observationX[iRegion-indexStartFinestLevel];
                        observationX[iRegion-indexStartFinestLevel]=NULL;

                        delete[] observationY[iRegion-indexStartFinestLevel];
                        observationY[iRegion-indexStartFinestLevel]=NULL;

                        delete[] observationZ[iRegion-indexStartFinestLevel];
                        observationZ[iRegion-indexStartFinestLevel]=NULL;
                    }
                }
            }
            
            if(CALCULATION_MODE=="prediction" && nPredictionsAtFinestLevel[iRegion-indexStartFinestLevel] > 0)
            {
                delete[] predictionLon[iRegion-indexStartFinestLevel];
                predictionLon[iRegion-indexStartFinestLevel]=NULL;

                delete[] predictionLat[iRegion-indexStartFinestLevel];
                predictionLat[iRegion-indexStartFinestLevel]=NULL;

                if(CHORDAL_DISTANCE_FLAG)
                {
                    delete[] predictionX[iRegion-indexStartFinestLevel];
                    predictionX[iRegion-indexStartFinestLevel]=NULL;

                    delete[] predictionY[iRegion-indexStartFinestLevel];
                    predictionY[iRegion-indexStartFinestLevel]=NULL;

                    delete[] predictionZ[iRegion-indexStartFinestLevel];
                    predictionZ[iRegion-indexStartFinestLevel]=NULL;
                }
            }
        }
    }
}

//Implement the member function of Partition, print_partition_summary, which shows a brief summary of the partition
void Partition::print_partition_summary()
{
    cout<<">>The number of levels: "<<NUM_LEVELS_M<<endl<<endl;
    cout<<">>The total number of regions: "<<nRegionsInTotal<<endl<<endl;
    cout<<">>The number of regions in each level: \n    ";
    for(int iLevel = 0; iLevel < NUM_LEVELS_M; iLevel++) cout<<nRegionsAtEachLevel[iLevel]<<" ";
    cout<<endl<<endl;

    cout<<">>Knot coordinates in the coarsest region, up to ten knots are shown: \n    longitude: ";
    for(int iKnot = 0; iKnot < std::min(nKnots,10); iKnot++) 
        printf("%8.2lf ",knotsLon[0][iKnot]);
    cout<<"\n    latitude:  ";
    for(int iKnot = 0; iKnot < std::min(nKnots,10); iKnot++) 
        printf("%8.2lf ",knotsLat[0][iKnot]);
    cout<<endl<<endl;

    if(CALCULATION_MODE=="predict")
        cout<<">>The number of predictions after eliminating duplicates: "<<NUM_PREDICTIONS<<endl<<endl;

}

void Partition::dump_structure_information()
{
    //Open the file "structure_information.txt" to store structure information
    std::ofstream file;
    file.open("structure_information.txt");

    //Dump the number of levels 
    file<<"The number of levels: "<<NUM_LEVELS_M<<endl<<endl;

    //Dump the total number of regions
    file<<"The total number of regions: "<<nRegionsInTotal<<endl<<endl;


    //Dump the number of knots in regions before the finest level
    file<<"The actual number of knots in regions before the finest level: "<<nKnots<<endl<<endl;

    //Dump some statistics of the number of knots in each region at the finest level
    unsigned long nKnotsMin = nKnotsAtFinestLevel[0];
    unsigned long nKnotsMax = nKnotsAtFinestLevel[0];
    unsigned long nRegionsHaveZeroKnots = 0;

    for(unsigned long iRegion = 0; iRegion < nRegionsAtEachLevel[NUM_LEVELS_M-1]; iRegion++)
    {
        if(nKnotsMin>nKnotsAtFinestLevel[iRegion]) nKnotsMin=nKnotsAtFinestLevel[iRegion];
        if(nKnotsMax<nKnotsAtFinestLevel[iRegion]) nKnotsMax=nKnotsAtFinestLevel[iRegion];
        if(nKnotsAtFinestLevel[iRegion] == 0) nRegionsHaveZeroKnots++;
    }

    file<<"The minimal number of knots in regions at the finest level: "<<nKnotsMin<<endl<<endl;
    file<<"The maximal number of knots in regions at the finest level: "<<nKnotsMax<<endl<<endl;
    file<<"The number of in regions at the finest level that have zero knots: "<<nRegionsHaveZeroKnots<<endl<<endl;

    unsigned long thresholdLeft, thresholdRight;
    unsigned long nRegionsBetweenThresholds;
    double interval = (nKnotsMax-nKnotsMin)/10;
    
    thresholdLeft=nKnotsMin;
    file<<"The number of regions at the finest level with number of knots in the following intervals:"<<endl;
    
    for(int i = 0; i < 10; i++)
    {
        nRegionsBetweenThresholds = 0;
        thresholdRight=thresholdLeft+interval;
        if(i == 9) thresholdRight = nKnotsMax;
        for(unsigned long iRegion = 0; iRegion < nRegionsAtEachLevel[NUM_LEVELS_M-1]; iRegion++)
           if(thresholdLeft<=nKnotsAtFinestLevel[iRegion] && nKnotsAtFinestLevel[iRegion]<=thresholdRight) nRegionsBetweenThresholds++;
        file<<"[ "<<thresholdLeft<<" , "<<thresholdRight<<" ]: "<<nRegionsBetweenThresholds<<endl;
        thresholdLeft = thresholdRight+1;
    }    
    file<<endl;

    //Dump the number of regions in each level
    file<<"The number of regions in each level: \n";
    for(int iLevel = 0; iLevel < NUM_LEVELS_M; iLevel++) 
        file<<"Level "<<iLevel+1<<": "<<nRegionsAtEachLevel[iLevel]<<endl;
    file<<endl;

    //Dump the number of knots in each region at the finest level
    file<<"The number of knots in each region at the finest level: \n";
    for(unsigned long iRegion = 0; iRegion < nRegionsAtEachLevel[NUM_LEVELS_M-1]; iRegion++)
        file<<"Region "<<nRegionsInTotal-nRegionsAtEachLevel[NUM_LEVELS_M-1]+iRegion<<": "<<nKnotsAtFinestLevel[iRegion]<<endl;


    //Close the file
    file.close();
}

Partition::~Partition()
{
    delete[] nRegionsAtEachLevel;

    if(OBS_AS_KNOTS_FLAG)
    {
        delete[] nKnotsAtFinestLevel;
        delete[] knotsResidual;
    }else
    {
        delete[] nObservationsAtFinestLevel;
        delete[] residual;
    }
    
    if(CHORDAL_DISTANCE_FLAG)
    {
        delete[] knotsX;
        delete[] knotsY;
        delete[] knotsZ;

        delete[] predictionX, 
        delete[] predictionY;
        delete[] predictionZ;
    }
    else
    {
        delete[] knotsLon;
        delete[] knotsLat;
    
        delete[] predictionLon, 
        delete[] predictionLat;
    }

    delete[] nPredictionsAtFinestLevel;
    
    if(DYNAMIC_SCHEDULE_FLAG) delete[] origninalWorker;
    //cout<<"Partition is deleted\n";
}