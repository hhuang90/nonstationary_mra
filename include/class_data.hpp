#ifndef _CLASS_DATA
#define _CLASS_DATA

#include <armadillo>

using namespace arma;

class Data
/*
Declare the class of Data, which contains all the information about the data
*/
{
    public:
        //nLocations: the number of locations in the raw data, including observations and missing values
        //numObservationsInRawData: the number of observations in the raw data, may including duplicates 
        unsigned long nLocations, numObservationsInRawData, numPredictionsInRawData;

        //observationResiduals: raw values or residuals after a linear regression to longitude, latitude, and a constant
        double *observationResiduals;
        
        //observationLon:       longitude of observations
        //observationLat:       latitude of observations    
        //observationRegion:    region index of observations
        double *observationLon, *observationLat;
        unsigned long *observationRegion;
        
        //predictionLon:        longitude for predictions
        //predictionLat:        latitude of predictions
        //predictionRegion:     region index of predictions
        double *predictionLon, *predictionLat;
        unsigned long *predictionRegion;

        //coefficients: the coefficients in the regression model
        vec coefficients;

        //domainBoundaries: boundry of the spatial domain, an array of (minLon,maxLon,minLat,maxLat)
        double domainBoundaries[4];

        //Loads data from file with name DATA_FILE_NAME. The data file is assumed to begin with the number of locations (type: unsigned 64-bit integer), followed by an array of longitudes, latitudes, and observations (type: 64-bit real number)
        void load_data();

        //Print a brief summary of the data
        void print_data_summary();

        //Destructor
        ~Data();
        
    private:

        //Eliminate duplicate observation locations and prediction locatins
        void eliminate_duplicate(double *& observationValues);
};
#endif