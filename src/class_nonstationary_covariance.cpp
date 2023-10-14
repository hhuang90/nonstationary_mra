#include <iostream>
#include <fstream>
#include <armadillo>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include "sys/stat.h"

using namespace arma;

#include "class_nonstationary_covariance.hpp"
#include "constants.hpp"

//Calculate the Euclidean distance on a unit sphere between (x1,y1,z1) and (x2,y2,z2)
static inline double euclidean_distance(double x1, double y1, double z1, double x2, double y2, double z2)
{
	//Calculate difference
	double diffx = x2 - x1;
    double diffy = y2 - y1;
    double diffz = z2 - z1;

    double dist = sqrt(diffx*diffx + diffy*diffy + diffz*diffz);
    return(dist);
}

//Calculate the great circle distance on a unit sphere between (lon1,lat1) and (lon2,lat2)
static inline double great_circle_dist(double lon1, double lon2, double lat1, double lat2)
{
	//Convert degrees to radians
	lon1 *= M_PI/180; 
	lon2 *= M_PI/180; 
	lat1 *= M_PI/180; 
	lat2 *= M_PI/180; 

	// Calculate angles
    double angle = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon1-lon2);
    if(angle > 1.0) angle = 1.0;
    if(angle < -1.0) angle = -1.0;

	// Return great circle distances
    double dist = acos(angle);
    return(dist);
}

//Implement the member function of Nonstationary_covariance, load_data, which loads data from files with name SIGMASQ_FILE_NAME, BETA_FILE_NAME, and TAUSQ_FILE_NAME.
void Nonstationary_covariance::load_data()
{
    //Check file exists
    struct stat fileStatus;
    if(stat(SIGMASQ_FILE_NAME.c_str(),&fileStatus) == -1 || !S_ISREG(fileStatus.st_mode))
    {
        if(WORKER == 0) cout<<"Program exits with an error: the file "<<SIGMASQ_FILE_NAME<<" does not exist.\n";
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if(stat(BETA_FILE_NAME.c_str(),&fileStatus) == -1 || !S_ISREG(fileStatus.st_mode))
    {
        if(WORKER == 0) cout<<"Program exits with an error: the file "<<BETA_FILE_NAME<<" does not exist.\n";
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if(stat(TAUSQ_FILE_NAME.c_str(),&fileStatus) == -1 || !S_ISREG(fileStatus.st_mode))
    {
        if(WORKER == 0) cout<<"Program exits with an error: the file "<<TAUSQ_FILE_NAME<<" does not exist.\n";
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    MPI_File file;
    MPI_Status* readStatus = new MPI_Status;
    int count;

    /*Start of reading SIGMASQ_FILE_NAME*/
    //Open the data file by SIGMASQ_FILE_NAME
    MPI_File_open(MPI_COMM_WORLD,SIGMASQ_FILE_NAME.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&file);
    
    //Read data
    //Get the number of basis functions, numSigmasqBasis   
    MPI_File_read(file,(void*)&numSigmasqBasis,1,MPI_UNSIGNED_LONG,readStatus);
    MPI_Get_count(readStatus,MPI_UNSIGNED_LONG,&count);

    if(count!=1)
    {
        if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<SIGMASQ_FILE_NAME<<" for the number of basis functions. Double check the data file format."<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    //Get the intercept
    MPI_File_read(file,(void*)&interceptSigmasq,1,MPI_DOUBLE,readStatus);
    MPI_Get_count(readStatus,MPI_DOUBLE,&count);

    if(count!=1)
    {
        if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<SIGMASQ_FILE_NAME<<" for the intercept. Double check the data file format."<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    sigmasqLon = new double [numSigmasqBasis];
    sigmasqLat = new double [numSigmasqBasis];
    sigmasqWeight = new double [numSigmasqBasis];

    //Get lon
    int interval=INT_MAX;
    unsigned long positionStart = 0, positionEnd = positionStart + interval;
    while(positionStart<numSigmasqBasis)
    {
        if(positionEnd>numSigmasqBasis) interval = numSigmasqBasis - positionStart;
        
        MPI_File_read(file,(void*)&sigmasqLon[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<SIGMASQ_FILE_NAME<<" for longitude. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Get lat
    interval=INT_MAX;
    positionStart = 0, positionEnd = positionStart + interval;
    while(positionStart<numSigmasqBasis)
    {
        if(positionEnd>numSigmasqBasis) interval = numSigmasqBasis - positionStart;
        MPI_File_read(file,(void*)&sigmasqLat[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<SIGMASQ_FILE_NAME<<" for latitude. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Get weights
    positionStart = 0, positionEnd = positionStart + interval;
    while(positionStart<numSigmasqBasis)
    {
        if(positionEnd>numSigmasqBasis) interval = numSigmasqBasis - positionStart;
        MPI_File_read(file,(void*)&sigmasqWeight[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<SIGMASQ_FILE_NAME<<" for weights. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Close the file by SIGMASQ_FILE_NAME
    MPI_File_close(&file);
    /*End of reading SIGMASQ_FILE_NAME*/

    /*Start of reading BETA_FILE_NAME*/
    //Open the data file by BETA_FILE_NAME
    MPI_File_open(MPI_COMM_WORLD,BETA_FILE_NAME.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&file);
    
    //Read data
    //Get the number of basis functions, numSigmasqBasis   
    MPI_File_read(file,(void*)&numBetaBasis,1,MPI_UNSIGNED_LONG,readStatus);
    MPI_Get_count(readStatus,MPI_UNSIGNED_LONG,&count);

    if(count!=1)
    {
        if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<BETA_FILE_NAME<<" for the number of basis functions. Double check the data file format."<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    //Get the intercept
    MPI_File_read(file,(void*)&interceptBeta,1,MPI_DOUBLE,readStatus);
    MPI_Get_count(readStatus,MPI_DOUBLE,&count);

    if(count!=1)
    {
        if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<BETA_FILE_NAME<<" for the intercept. Double check the data file format."<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    betaLon = new double [numBetaBasis];
    betaLat = new double [numBetaBasis];
    betaWeight = new double [numBetaBasis];

    //Get lon
    interval=INT_MAX;
    positionStart = 0; positionEnd = positionStart + interval;
    while(positionStart<numBetaBasis)
    {
        if(positionEnd>numBetaBasis) interval = numBetaBasis - positionStart;
        
        MPI_File_read(file,(void*)&betaLon[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<BETA_FILE_NAME<<" for longitude. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Get lat
    interval=INT_MAX;
    positionStart = 0, positionEnd = positionStart + interval;
    while(positionStart<numBetaBasis)
    {
        if(positionEnd>numBetaBasis) interval = numBetaBasis - positionStart;
        MPI_File_read(file,(void*)&betaLat[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<BETA_FILE_NAME<<" for latitude. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Get weights
    positionStart = 0, positionEnd = positionStart + interval;
    while(positionStart<numBetaBasis)
    {
        if(positionEnd>numBetaBasis) interval = numBetaBasis - positionStart;
        MPI_File_read(file,(void*)&betaWeight[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<BETA_FILE_NAME<<" for weights. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Close the file by BETA_FILE_NAME
    MPI_File_close(&file);
    /*End of reading BETA_FILE_NAME*/

    /*Start of reading TAUSQ_FILE_NAME*/
    //Open the file by TAUSQ_FILE_NAME
    MPI_File_open(MPI_COMM_WORLD,TAUSQ_FILE_NAME.c_str(),MPI_MODE_RDONLY,MPI_INFO_NULL,&file);
    
    //Read data
    //Get the number of basis functions, numSigmasqBasis   
    MPI_File_read(file,(void*)&numTausqBasis,1,MPI_UNSIGNED_LONG,readStatus);
    MPI_Get_count(readStatus,MPI_UNSIGNED_LONG,&count);

    if(count!=1)
    {
        if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<TAUSQ_FILE_NAME<<" for the number of basis functions. Double check the data file format."<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    //Get the intercept
    MPI_File_read(file,(void*)&interceptTausq,1,MPI_DOUBLE,readStatus);
    MPI_Get_count(readStatus,MPI_DOUBLE,&count);

    if(count!=1)
    {
        if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<TAUSQ_FILE_NAME<<" for the intercept. Double check the data file format."<<endl;
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    tausqLon = new double [numTausqBasis];
    tausqLat = new double [numTausqBasis];
    tausqWeight = new double [numTausqBasis];

    //Get lon
    interval=INT_MAX;
    positionStart = 0; positionEnd = positionStart + interval;
    while(positionStart<numTausqBasis)
    {
        if(positionEnd>numTausqBasis) interval = numTausqBasis - positionStart;
        
        MPI_File_read(file,(void*)&tausqLon[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<TAUSQ_FILE_NAME<<" for longitude. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Get lat
    interval=INT_MAX;
    positionStart = 0, positionEnd = positionStart + interval;
    while(positionStart<numTausqBasis)
    {
        if(positionEnd>numTausqBasis) interval = numTausqBasis - positionStart;
        MPI_File_read(file,(void*)&tausqLat[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<TAUSQ_FILE_NAME<<" for latitude. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Get weights
    positionStart = 0, positionEnd = positionStart + interval;
    while(positionStart<numTausqBasis)
    {
        if(positionEnd>numTausqBasis) interval = numTausqBasis - positionStart;
        MPI_File_read(file,(void*)&tausqWeight[positionStart],interval,MPI_DOUBLE,readStatus);
        MPI_Get_count(readStatus,MPI_DOUBLE,&count);

        positionStart = positionEnd;
        positionEnd += interval;
        if(count!=interval)
        {
            if(WORKER == 0) cout<<"Program exits with an error: fail reading data file "<<TAUSQ_FILE_NAME<<" for weights. Double check the data file format."<<endl;
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            exit(EXIT_FAILURE);
        }
    }

    //Close the data file
    MPI_File_close(&file);
    /*End of reading TAUSQ_FILE_NAME*/

    delete readStatus;

    if( (WORKER == 0) && PRINT_DETAIL_FLAG) print_summary();

    if(CHORDAL_DISTANCE_FLAG)
    {
        sigmasqX = new double [numSigmasqBasis];
        sigmasqY = new double [numSigmasqBasis];
        sigmasqZ = new double [numSigmasqBasis];

        for(int i = 0; i < numSigmasqBasis; i++)
        {
            double cosLat = cos(sigmasqLat[i] * M_PI/180);
            double cosLon = cos(sigmasqLon[i] * M_PI/180);
            double sinLat = sin(sigmasqLat[i] * M_PI/180);
            double sinLon = sin(sigmasqLon[i] * M_PI/180);

            sigmasqX[i] = cosLat * cosLon;
            sigmasqY[i] = cosLat * sinLon;
            sigmasqZ[i] = sinLat;
        }

        // delete[] sigmasqLat, sigmasqLon;

        betaX = new double [numBetaBasis];
        betaY = new double [numBetaBasis];
        betaZ = new double [numBetaBasis];

        for(int i = 0; i < numBetaBasis; i++)
        {
            double cosLat = cos(betaLat[i] * M_PI/180);
            double cosLon = cos(betaLon[i] * M_PI/180);
            double sinLat = sin(betaLat[i] * M_PI/180);
            double sinLon = sin(betaLon[i] * M_PI/180);

            betaX[i] = cosLat * cosLon;
            betaY[i] = cosLat * sinLon;
            betaZ[i] = sinLat;
        }

        // delete[] betaLat, betaLon;

        tausqX = new double [numTausqBasis];
        tausqY = new double [numTausqBasis];
        tausqZ = new double [numTausqBasis];

        for(int i = 0; i < numTausqBasis; i++)
        {
            double cosLat = cos(tausqLat[i] * M_PI/180);
            double cosLon = cos(tausqLon[i] * M_PI/180);
            double sinLat = sin(tausqLat[i] * M_PI/180);
            double sinLon = sin(tausqLon[i] * M_PI/180);

            tausqX[i] = cosLat * cosLon;
            tausqY[i] = cosLat * sinLon;
            tausqZ[i] = sinLat;
        }

        // delete[] tausqLat, tausqLon;
    }
}


//Implement the member function of Nonstationary_covariance, print_summary, which prints a brief summary of the nonstationary model
void Nonstationary_covariance::print_summary()
{
    cout<<">>The number of basis functions for sill, range, and nugget are: "<<numSigmasqBasis<<", "
        <<numBetaBasis<<", and "<<numTausqBasis<<" respectively."<< endl<<endl;
    
    cout<<">>The intercepts for sill, range, and nugget are: "<<interceptSigmasq<<", "
        <<interceptBeta<<", and "<<interceptTausq<<" respectively."<< endl<<endl;

    cout<<">>Examples of sill basis functions"<<endl;
    cout<<"     longitude: ";
    for(unsigned long iObs = 0; iObs < std::min(numSigmasqBasis - 1, (unsigned long)5); iObs++) 
        printf("%8.2lf ",sigmasqLon[iObs]);
    cout<<"\n     latitude:  ";
    for(unsigned long iObs = 0; iObs < std::min(numSigmasqBasis - 1, (unsigned long)5); iObs++) 
        printf("%8.2lf ",sigmasqLat[iObs]);
    cout<<"\n     weights: ";
    for(unsigned long iObs = 0; iObs < std::min(numSigmasqBasis - 1, (unsigned long)5); iObs++) 
        printf("%8.2lf ",sigmasqWeight[iObs]);
    cout<<"\n\n";

    cout<<">>Examples of range basis functions"<<endl;
    cout<<"     longitude: ";
    for(unsigned long iObs = 0; iObs < std::min(numBetaBasis - 1, (unsigned long)5); iObs++) 
        printf("%8.2lf ",betaLon[iObs]);
    cout<<"\n     latitude:  ";
    for(unsigned long iObs = 0; iObs < std::min(numBetaBasis - 1, (unsigned long)5); iObs++) 
        printf("%8.2lf ",betaLat[iObs]);
    cout<<"\n     weights: ";
    for(unsigned long iObs = 0; iObs < std::min(numBetaBasis - 1, (unsigned long)5); iObs++) 
        printf("%8.2lf ",betaWeight[iObs]);
    cout<<"\n\n";

    cout<<">>Examples of nugget basis functions"<<endl;
    cout<<"     longitude: ";
    for(unsigned long iObs = 0; iObs < std::min(numTausqBasis - 1, (unsigned long)5); iObs++) 
        printf("%8.2lf ",tausqLon[iObs]);
    cout<<"\n     latitude:  ";
    for(unsigned long iObs = 0; iObs < std::min(numTausqBasis - 1, (unsigned long)5); iObs++) 
        printf("%8.2lf ",tausqLat[iObs]);
    cout<<"\n     weights: ";
    for(unsigned long iObs = 0; iObs < std::min(numTausqBasis - 1, (unsigned long)5); iObs++) 
        printf("%8.2lf ",tausqWeight[iObs]);
    cout<<"\n\n";
}

//Implement the member function of Nonstationary_covariance, get_sigmasq, which returns the sill parameter at (lon,lat)
double Nonstationary_covariance::get_sigmasq(double lon, double lat)
{
    double sigmasq = interceptSigmasq;
    for(unsigned long i = 0; i < numSigmasqBasis; i++)
    {
        double d = great_circle_dist(lon, sigmasqLon[i], lat, sigmasqLat[i]);
        d /= WENDLAND_LEN;
        if(d>=1) continue;
        
        sigmasq += pow(1-d,6) * (35*d*d + 18*d + 3) * sigmasqWeight[i] / 3 ;
    }
    sigmasq = exp(sigmasq);
    return sigmasq;
}

//Implement the member function of Nonstationary_covariance, get_sigmasq, which returns the sill parameter at (x,y,z)
double Nonstationary_covariance::get_sigmasq(double x, double y, double z)
{
    double sigmasq = interceptSigmasq;
    for(unsigned long i = 0; i < numSigmasqBasis; i++)
    {
        double d = euclidean_distance(x, y, z, sigmasqX[i], sigmasqY[i], sigmasqZ[i]);
        d /= WENDLAND_LEN;
        if(d>=1) continue;
        
        sigmasq += pow(1-d,6) * (35*d*d + 18*d + 3) * sigmasqWeight[i] / 3 ;
    }
    sigmasq = exp(sigmasq);
    return sigmasq;
}


//Implement the member function of Nonstationary_covariance, get_beta, which returns the range parameter at (lon,lat)
double Nonstationary_covariance::get_beta(double lon, double lat)
{
    double beta = interceptBeta;
    for(unsigned long i = 0; i < numBetaBasis; i++)
    {
        double d = great_circle_dist(lon, betaLon[i], lat, betaLat[i]);
        d /= WENDLAND_LEN;
        if(d>=1) continue;
        
        beta += pow(1-d,6) * (35*d*d + 18*d + 3) * betaWeight[i] / 3 ;
    }
    beta = exp(beta);
    return beta;
}

//Implement the member function of Nonstationary_covariance, get_beta, which returns the range parameter at (x,y,z)
double Nonstationary_covariance::get_beta(double x, double y, double z)
{
    double beta = interceptBeta;
    for(unsigned long i = 0; i < numBetaBasis; i++)
    {
        double d = euclidean_distance(x, y, z, betaX[i], betaY[i], betaZ[i]);
        d /= WENDLAND_LEN;
        if(d>=1) continue;
        
        beta += pow(1-d,6) * (35*d*d + 18*d + 3) * betaWeight[i] / 3 ;
    }
    beta = exp(beta);
    return beta;
}

//Implement the member function of Nonstationary_covariance, get_tausq, which returns the nugget parameter at (lon,lat)
double Nonstationary_covariance::get_tausq(double lon, double lat)
{
    double tausq = interceptTausq;
    for(unsigned long i = 0; i < numTausqBasis; i++)
    {
        double d = great_circle_dist(lon, tausqLon[i], lat, tausqLat[i]);
        d /= WENDLAND_LEN;
        if(d>=1) continue;
        
        tausq += pow(1-d,6) * (35*d*d + 18*d + 3) * tausqWeight[i] / 3 ;
    }
    tausq = exp(tausq);
    return tausq;
}

//Implement the member function of Nonstationary_covariance, get_tausq, which returns the nugget parameter at (x,y,z)
double Nonstationary_covariance::get_tausq(double x, double y, double z)
{
    double tausq = interceptTausq;
    for(unsigned long i = 0; i < numTausqBasis; i++)
    {
        double d = euclidean_distance(x, y, z, tausqX[i], tausqY[i], tausqZ[i]);
        d /= WENDLAND_LEN;
        if(d>=1) continue;
        
        tausq += pow(1-d,6) * (35*d*d + 18*d + 3) * tausqWeight[i] / 3 ;
    }
    tausq = exp(tausq);
    return tausq;
}