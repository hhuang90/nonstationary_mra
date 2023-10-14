#ifndef _CLASS_NONSTATIONARY_COVARIANCE
#define _CLASS_NONSTATIONARY_COVARIANCE

#include <armadillo>

using namespace arma;

class Nonstationary_covariance
/*
Declare the class of Data, which contains all the information about the data
*/
{
    public:
        //numSigmasqBasis: the number of basis functions for the sill parameter
        //numBetaBasis: the number of basis functions for the range parameter
        //numTausqBasis: the number of basis functions for the nugget parameter
        unsigned long numSigmasqBasis, numBetaBasis, numTausqBasis;

        //interceptSigmasq: intercept for the sill parameter
        //interceptBeta: intercept for the range parameter
        //interceptTausq: intercept for the nugget parameter
        double interceptSigmasq, interceptBeta, interceptTausq;

        //sigmasqLon: longitude of the center of the basis function for the sill parameter
        //sigmasqLat: latitude of the center of the basis function for the sill parameter
        //sigmasqWeight: weight of the basis function for the sill parameter
        double *sigmasqLon, *sigmasqLat, *sigmasqWeight;

        //When chordal distance is used, the x, y, z coordinates for the sill
        double *sigmasqX, *sigmasqY, *sigmasqZ;

        //betaLon: longitude of the center of the basis function for the range parameter
        //betaLat: latitude of the center of the basis function for the range parameter
        //betaWeight: weight of the basis function for the range parameter
        double *betaLon, *betaLat, *betaWeight;

        //When chordal distance is used, the x, y, z coordinates for the range
        double *betaX, *betaY, *betaZ;

        //tausqLon: longitude of the center of the basis function for the nugget parameter
        //tausqLat: latitude of the center of the basis function for the nugget parameter
        //tausqWeight: weight of the basis function for the nugget parameter
        double *tausqLon, *tausqLat, *tausqWeight;

        //When chordal distance is used, the x, y, z coordinates for the nugget
        double *tausqX, *tausqY, *tausqZ;

        //Loads data from files with name SIGMASQ_FILE_NAME, BETA_FILE_NAME, and TAUSQ_FILE_NAME.
        void load_data();

        //Print a brief summary of the nonstationary model
        void print_summary();

        //Get the sill parameter at (lon,lat)
        double get_sigmasq(double lon, double lat);
        double get_sigmasq(double x, double y, double z);

        //Get the range parameter at (lon,lat)
        double get_beta(double lon, double lat);
        double get_beta(double x, double y, double z);

        //Get the nugget parameter at (lon,lat)
        double get_tausq(double lon, double lat);
        double get_tausq(double x, double y, double z);

        //Destructor
        ~Nonstationary_covariance();
};
#endif