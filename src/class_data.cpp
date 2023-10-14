#include <iostream>
#include <fstream>
#include <armadillo>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include "sys/stat.h"

using namespace arma;

#include "class_data.hpp"
#include "constants.hpp"

//Implement the member function of Data, load_data, which loads data from file with name DATA_FILE_NAME. The data file is assumed to begin with the number of locations (type: unsigned 64-bit integer), followed by an array of longitudes, latitudes, and observations (type: 64-bit real numbers)
void Data::load_data()
{
    if(WORKER == 0)
	{
		gettimeofday(&timeNow, NULL);
		cout<<"===========================> Processor 1: loading data starts. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n";
	}

    //Check data file exists
    struct stat fileStatus;
    if(stat(DATA_FILE_NAME.c_str(),&fileStatus) == -1 || !S_ISREG(fileStatus.st_mode))
    {
        if(WORKER == 0) cout<<"Program exits with an error: the data file "<<DATA_FILE_NAME<<" does not exist.\n";
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    //Open the data file by DATA_FILE_NAME
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD,DATA_FILE_NAME.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&file);

    //Read data
    //longitude, latitude, and values of all data points, including observations that have valid values and predicitons that have NaN values
    double *lon, *lat, *values;
    unsigned long * regionIndex;

    //Get the number of locations, nLocations   
    MPI_Status* readStatus = new MPI_Status;
    int count;

    nLocations=0;
    MPI_File_read(file,(void*)&nLocations,1,MPI_UNSIGNED_LONG,readStatus);
    MPI_Get_count(readStatus,MPI_UNSIGNED_LONG,&count);

    if(count!=1)
    {
        if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<DATA_FILE_NAME<<" for the number of locations. Double check the data file format."<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    lon = new double [nLocations];
    lat = new double [nLocations];
    values = new double [nLocations];

    //Get lon
    int interval=INT_MAX;
    unsigned long positionStart = 0, positionEnd = positionStart + interval;
    while(positionStart<nLocations)
    {
        if(positionEnd>nLocations) interval = nLocations - positionStart;
        
        MPI_File_read(file,(void*)&lon[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<DATA_FILE_NAME<<" for longitude. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Get lat
    interval=INT_MAX;
    positionStart = 0, positionEnd = positionStart + interval;
    while(positionStart<nLocations)
    {
        if(positionEnd>nLocations) interval = nLocations - positionStart;
        MPI_File_read(file,(void*)&lat[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<DATA_FILE_NAME<<" for latitude. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Get values
    positionStart = 0, positionEnd = positionStart + interval;
    while(positionStart<nLocations)
    {
        if(positionEnd>nLocations) interval = nLocations - positionStart;
        MPI_File_read(file,(void*)&values[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<DATA_FILE_NAME<<" for observation values. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    if(PROVIDED_KNOTS_FLAG)
    {
        regionIndex = new unsigned long [nLocations];
        //Get region indices
        positionStart = 0, positionEnd = positionStart + interval;
        while(positionStart<nLocations)
        {
            if(positionEnd>nLocations) interval = nLocations - positionStart;
            MPI_File_read(file,(void*)&regionIndex[positionStart],interval,MPI_UNSIGNED_LONG,readStatus);
            MPI_Get_count(readStatus,MPI_UNSIGNED_LONG,&count);

            positionStart = positionEnd;
            positionEnd += interval;
            if(count!=interval)
            {
                if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<DATA_FILE_NAME<<" for observation region indices. Double check the data file format."<<endl;
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }
        }
    }

    //Close the data file
    MPI_File_close(&file);

    //Get min and max
    double minLon = 1e9, minLat = 1e9, maxLon = -1e9, maxLat = -1e9;
    #pragma omp parallel reduction(min:minLon,minLat) reduction(max:maxLon,maxLat)
    for(unsigned long iLocation = 0; iLocation < nLocations; iLocation++)
    {
        minLon=std::min(minLon,lon[iLocation]);
        minLat=std::min(minLat,lat[iLocation]);
        maxLon=std::max(maxLon,lon[iLocation]);   
        maxLat=std::max(maxLat,lat[iLocation]);
    }

    //Assign the boundry, domainBoundaries
    domainBoundaries[0] = minLon;
    domainBoundaries[1] = maxLon + 1e-6 * (maxLon-minLon);
    domainBoundaries[2] = minLat;
    domainBoundaries[3] = maxLat + 1e-6 * (maxLat-minLat);

    //Split the data into observations and missing values
    numObservationsInRawData = 0;
    for(unsigned long iLocation = 0; iLocation < nLocations; iLocation++)
        if(!::isnan(values[iLocation])) numObservationsInRawData++;

    observationLon      = new double [numObservationsInRawData];
    observationLat      = new double [numObservationsInRawData];
    observationRegion   = new unsigned long [numObservationsInRawData];
    double *observationValues = new double [numObservationsInRawData];

    numPredictionsInRawData=nLocations;
    if(CALCULATION_MODE=="prediction" && PREDICTION_LOCATION_MODE=='N')
    {    
        predictionLon       = new double [numPredictionsInRawData];
        predictionLat       = new double [numPredictionsInRawData];
        predictionRegion    = new unsigned long [numPredictionsInRawData];

        unsigned long observationCount=0, predictionCount=0;
        for(unsigned long iLocation = 0; iLocation < nLocations; iLocation++)
        {
            if(!::isnan(values[iLocation]))
            {
                observationLon[observationCount] = lon[iLocation];
                observationLat[observationCount] = lat[iLocation];
                observationValues[observationCount] = values[iLocation];
                observationRegion[observationCount++] = regionIndex[iLocation];
            }else
            {
                predictionLon[predictionCount]      = lon[iLocation];
                predictionLat[predictionCount]      = lat[iLocation];
                predictionRegion[predictionCount++] = regionIndex[iLocation];
            }
        }
        numPredictionsInRawData = predictionCount;
    }else
    {
        unsigned long observationCount=0, predictionCount=0;
        for(unsigned long iLocation = 0; iLocation < nLocations; iLocation++)
        {
            if(!::isnan(values[iLocation]))
            {
                observationLon[observationCount]      = lon[iLocation];
                observationLat[observationCount]      = lat[iLocation];
                observationValues[observationCount] = values[iLocation];
                if(PROVIDED_KNOTS_FLAG) observationRegion[observationCount]   = regionIndex[iLocation];
                observationCount++;
            }
        }
    }

    if(CALCULATION_MODE=="prediction" && PREDICTION_LOCATION_MODE=='D')
    {
        predictionLon       = lon;
        predictionLat       = lat;
        if(PROVIDED_KNOTS_FLAG) predictionRegion    = regionIndex;
    }else
    {
        delete[] lon;
        delete[] lat;
        if(PROVIDED_KNOTS_FLAG) delete[] regionIndex;
    }

    if(CALCULATION_MODE=="prediction" && PREDICTION_LOCATION_MODE=='A')
    {
        if(stat(PREDICTION_LOCATION_FILE.c_str(),&fileStatus) == -1 || !S_ISREG(fileStatus.st_mode))
        {
            if(WORKER == 0) cout<<"Program exits with an error: the prediction location file "<<PREDICTION_LOCATION_FILE<<" does not exist.\n";
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }

        //Open the prediciton location file by PREDICTION_LOCATION_FILE
        MPI_File_open(MPI_COMM_WORLD,PREDICTION_LOCATION_FILE.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&file);

        //Read data
        MPI_File_read(file,(void*)&numPredictionsInRawData,1,MPI_UNSIGNED_LONG,readStatus);
        MPI_Get_count(readStatus,MPI_UNSIGNED_LONG,&count);

        if(count!=1)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<PREDICTION_LOCATION_FILE<<" for the number of locations. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
        
        predictionLon    = new double [numPredictionsInRawData];
        predictionLat    = new double [numPredictionsInRawData];
        predictionRegion = new unsigned long [numPredictionsInRawData];

        //Get prediction lon
        unsigned long positionStart = 0, positionEnd = positionStart + interval;
        while(positionStart<numPredictionsInRawData)
        {
            if(positionEnd>numPredictionsInRawData) interval = numPredictionsInRawData - positionStart;
            
            MPI_File_read(file,(void*)&predictionLon[positionStart],interval,MPI_DOUBLE,readStatus);
            MPI_Get_count(readStatus,MPI_DOUBLE,&count);

            positionStart = positionEnd;
            positionEnd += interval;
            if(count!=interval)
            {
                if(WORKER == 0) cout<<"Program exits with an error: fail reading prediction location file "<<PREDICTION_LOCATION_FILE<<" for longitude. Double check the data file format."<<endl;
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }
        }

        //Get prediction lat
        positionStart = 0, positionEnd = positionStart + interval;
        while(positionStart<numPredictionsInRawData)
        {
            if(positionEnd>numPredictionsInRawData) interval = numPredictionsInRawData - positionStart;
            MPI_File_read(file,(void*)&predictionLat[positionStart],interval,MPI_DOUBLE,readStatus);
            MPI_Get_count(readStatus,MPI_DOUBLE,&count);

            positionStart = positionEnd;
            positionEnd += interval;
            if(count!=interval)
            {
                if(WORKER == 0) cout<<"Program exits with an error: fail reading prediction location file "<<PREDICTION_LOCATION_FILE<<" for latitude. Double check the data file format."<<endl;
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Finalize();
                exit(EXIT_FAILURE);
            }
        }

        if(PROVIDED_KNOTS_FLAG)
        {
            //Get prediction region indices
            positionStart = 0, positionEnd = positionStart + interval;
            while(positionStart<numPredictionsInRawData)
            {
                if(positionEnd>numPredictionsInRawData) interval = numPredictionsInRawData - positionStart;
                MPI_File_read(file,(void*)&predictionRegion[positionStart],interval,MPI_UNSIGNED_LONG,readStatus);
                MPI_Get_count(readStatus,MPI_UNSIGNED_LONG,&count);

                positionStart = positionEnd;
                positionEnd += interval;
                if(count!=interval)
                {
                    if(WORKER == 0) cout<<"Program exits with an error: fail reading prediction location file "<<PREDICTION_LOCATION_FILE<<" for latitude. Double check the data file format."<<endl;
                    MPI_Barrier(MPI_COMM_WORLD);
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
            }
        }

        //Close the prediction location file
        MPI_File_close(&file);
    }

    delete[] values;

    //Eliminate duplicates in the raw data
    if(ELIMINATION_DUPLICATES_FLAG)  
        eliminate_duplicate(observationValues);
    else
    {
        NUM_OBSERVATIONS = numObservationsInRawData;
        NUM_PREDICTIONS = numPredictionsInRawData;
    }

    //Detrend data by a linear regression to lon, lat and a constant
    if(REGRESSION_FLAG)
    {
        mat desigenMatrix(NUM_OBSERVATIONS,3,fill::ones);

        vec lonVector(observationLon,NUM_OBSERVATIONS,false,true);
        vec latVector(observationLat,NUM_OBSERVATIONS,false,true);
        vec valueVector(observationValues,NUM_OBSERVATIONS,false,true);

        desigenMatrix.col(1)=lonVector;
        desigenMatrix.col(2)=latVector;
        
        coefficients = solve(desigenMatrix,valueVector);

        valueVector = valueVector - desigenMatrix * coefficients;
    }
    observationResiduals = observationValues;
    
    delete readStatus;

    if(WORKER == 0)
	{
		gettimeofday(&timeNow, NULL);
		cout<<"===========================> Processor 1: loading data is complete. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n\n";
		if(PRINT_DETAIL_FLAG) print_data_summary();
	}
}

//Implement the member function of Data, print_data_summary, which prints a brief summary of the data
void Data::print_data_summary()
{
    cout<<">>The number of locations: "<<nLocations<<endl;
    cout<<">>The number of valid observations (non-NaN values) in the raw data: "<<numObservationsInRawData<<endl<<endl;
    if(ELIMINATION_DUPLICATES_FLAG) cout<<">>The number of observations after eliminating duplicates: "<<NUM_OBSERVATIONS<<endl<<endl;
     
    cout<<">>The boundry of the spatial domain"<<endl;
    cout<<"  longitude: minimal "<<domainBoundaries[0]<<", maximal "<<domainBoundaries[1]<<endl;
    cout<<"  latitude : minimal "<<domainBoundaries[2]<<", maximal "<<domainBoundaries[3]<<endl<<endl;
    
    cout<<">>Examples of observations"<<endl;
    cout<<"     longitude: ";
    for(unsigned long iObs = 0; iObs < std::min(NUM_OBSERVATIONS - 1, (unsigned long)5); iObs++) 
        printf("%8.2lf ",observationLon[iObs]);
    cout<<"\n     latitude:  ";
    for(unsigned long iObs = 0; iObs < std::min(NUM_OBSERVATIONS - 1, (unsigned long)5); iObs++) 
        printf("%8.2lf ",observationLat[iObs]);
    cout<<"\n     residuals: ";
    for(unsigned long iObs = 0; iObs < std::min(NUM_OBSERVATIONS - 1, (unsigned long)5); iObs++) 
        printf("%8.2lf ",observationResiduals[iObs]);
    cout<<"\n\n";

    if(CALCULATION_MODE=="prediction")
    {
        cout<<">>The number of prediction locations in the raw data: "<<numPredictionsInRawData<<endl<<endl;
        if(ELIMINATION_DUPLICATES_FLAG) cout<<">>The number of prediction locations after eliminating duplicates: "<<NUM_PREDICTIONS<<endl<<endl;

        cout<<">>Examples of prediction locations"<<endl;
        cout<<"     longitude: ";
        for(unsigned long iPred = 0; iPred < std::min(NUM_PREDICTIONS - 1, (unsigned long)5); iPred++) 
            printf("%8.2lf ",predictionLon[iPred]);
        cout<<"\n     latitude:  ";
        for(unsigned long iPred = 0; iPred < std::min(NUM_PREDICTIONS - 1, (unsigned long)5); iPred++) 
            printf("%8.2lf ",predictionLat[iPred]);
        cout<<"\n\n";
    }
}

//Implement the function that eliminate duplicate observation locations and prediction locations
void Data::eliminate_duplicate(double *& observationValues)
{
    struct compare
    {
        Data* data;
        compare(Data *data): data(data) {}
        bool operator() (unsigned long i, unsigned long j) const
        {
            if(data->observationLon[i]<data->observationLon[j]) return true;
            if(data->observationLon[i]==data->observationLon[j]) return data->observationLat[i]<data->observationLat[j];
            return false;
        }
    };

    struct prediction_compare
    {
        Data* data;
        prediction_compare(Data *data): data(data) {}
        bool operator() (unsigned long i, unsigned long j) const
        {
            if(data->predictionLon[i]<data->predictionLon[j]) return true;
            if(data->predictionLon[i]==data->predictionLon[j]) return data->predictionLat[i]<data->predictionLat[j];
            return false;
        }
    };

    unsigned long *tmpIndex = new unsigned long [numObservationsInRawData];
    for(unsigned long index = 0; index < numObservationsInRawData; index++) tmpIndex[index]=index;
    
    std::sort(tmpIndex,tmpIndex+numObservationsInRawData, compare(this));

    bool *duplicateFlag = new bool [numObservationsInRawData]();
    
    NUM_OBSERVATIONS = numObservationsInRawData;

    for(unsigned long index = 1; index < numObservationsInRawData; index++)
        if((observationLon[tmpIndex[index]]==observationLon[tmpIndex[index-1]]) && (observationLat[tmpIndex[index]]==observationLat[tmpIndex[index-1]])) 
        {
            duplicateFlag[index] = true;
            NUM_OBSERVATIONS--;
        }

    double *tmpObservationLon, *tmpObservationLat, *tmpObservationValues;
    tmpObservationLon = new double [NUM_OBSERVATIONS];
    tmpObservationLat = new double [NUM_OBSERVATIONS];
    tmpObservationValues = new double [NUM_OBSERVATIONS]; 

    unsigned long count=0;
    for(unsigned long index = 0; index < numObservationsInRawData; index++)    
    {
        if(duplicateFlag[index]) continue;
        tmpObservationLon[count]=observationLon[tmpIndex[index]];
        tmpObservationLat[count]=observationLat[tmpIndex[index]];
        tmpObservationValues[count++]=observationValues[tmpIndex[index]];
    }

    delete[] observationLon;
    delete[] observationLat;
    delete[] observationValues;

    observationLon=tmpObservationLon;
    observationLat=tmpObservationLat;
    observationValues=tmpObservationValues;

    delete[] tmpIndex;
    delete[] duplicateFlag;

    if(CALCULATION_MODE=="prediction")
    {
        unsigned long *tmpIndex = new unsigned long [numPredictionsInRawData];
        for(unsigned long index = 0; index < numPredictionsInRawData; index++) tmpIndex[index]=index;
        
        std::sort(tmpIndex,tmpIndex+numPredictionsInRawData, prediction_compare(this));

        bool *duplicateFlag = new bool [numPredictionsInRawData]();
        
        NUM_PREDICTIONS = numPredictionsInRawData;

        for(unsigned long index = 1; index < numPredictionsInRawData; index++)
            if((predictionLon[tmpIndex[index]]==predictionLon[tmpIndex[index-1]]) && (predictionLat[tmpIndex[index]]==predictionLat[tmpIndex[index-1]])) 
            {
                duplicateFlag[index] = true;
                NUM_PREDICTIONS--;
            }

        double *tmpPredictionLon, *tmpPredictionLat;
        tmpPredictionLon = new double [NUM_PREDICTIONS];
        tmpPredictionLat = new double [NUM_PREDICTIONS];

        unsigned long count=0;
        for(unsigned long index = 0; index < numPredictionsInRawData; index++)    
        {
            if(duplicateFlag[index]) continue;
            tmpPredictionLon[count]=predictionLon[tmpIndex[index]];
            tmpPredictionLat[count++]=predictionLat[tmpIndex[index]];
        }

        delete[] predictionLon;
        delete[] predictionLat;

        predictionLon=tmpPredictionLon;
        predictionLat=tmpPredictionLat;

        delete[] tmpIndex;
        delete[] duplicateFlag;
    }
}

//Implement the destructor of class Data
Data::~Data()
{
    //delete[] predictionLon;
    //delete[] predictionLat;
    
    //cout<<"Data is deleted.\n";
}
