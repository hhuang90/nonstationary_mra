#include <armadillo>

#include "constants.hpp"
#include "class_nonstationary_covariance.hpp"

using namespace arma;

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

/*
Get stationary cross-covariance matrix between two location sets

Input:
    locationsX1: X coordinates of location set 1
    locationsY1: Y coordinates of location set 1
    locationsX2: X coordinates of location set 2
    locationsY2: Y coordinates of location set 2
    nLocations1: the number of locations in lcs1
    nLocations2: the number of locations in lcs2
    sill: sill parameter
    range: range parameter
    nugget: nugget parameter

Output:
    covarianceMatrix: obtained covariances, of dimension nLocations2 * nLocations1
*/
void evaluate_cross_covariance(double *covarianceMatrix,
double *lon1, double *lat1, double *x1, double *y1, double *z1,
double *lon2, double *lat2, double *x2, double *y2, double *z2,
const unsigned long &nLocations1, const unsigned long &nLocations2, const double &sill, const double &range, const double &nugget)
{
    for(unsigned long index1 = 0; index1 < nLocations1; index1++)
        for(unsigned long index2 = 0; index2 < nLocations2; index2++)
        {
            // double xDifference=locationsX1[index1]-locationsX2[index2];
            // double yDifference=locationsY1[index1]-locationsY2[index2];

            // double distance = sqrt( xDifference*xDifference + yDifference*yDifference );

            //Radius of Earth: 6371 km
            double distance;
            
            if(CHORDAL_DISTANCE_FLAG)
                distance = 6371 * euclidean_distance(x1[index1], y1[index1], z1[index1],
                                                        x2[index2], y2[index2], z2[index2]);
            else
                distance = 6371 * great_circle_dist(lon1[index1], lon2[index2],
                                                       lat1[index1], lat2[index2]);

            // if (distance < 1e-6)
                // covarianceMatrix[index1*nLocations2+index2] = sill + nugget;
            // else
                covarianceMatrix[index1*nLocations2+index2] = sill*exp(-distance/range);
        }
}

/*
Get stationary variance-covariance matrix for one location set

Input:
    locationsX: X coordinates
    locationsY: Y coordinates
    nLocations: the number of locations
    sill: sill parameter
    range: range parameter
    nugget: nugget parameter

Output:
    covarianceMatrix: obtained covariances, of dimension nLocations * nLocations
*/
void evaluate_variance_covariance(double *covarianceMatrix, double *lon, double *lat, double *x, double *y, double *z,
const unsigned long &nLocations, const double &sill, const double &range, const double &nugget)
{
    double diagonal = sill + nugget;
    for(unsigned long index1 = 0; index1 < nLocations; index1++)
    {
        for(unsigned long index2 = 0; index2 < index1; index2++)
        {
            // double xDifference=locationsX[index1]-locationsX[index2];
            // double yDifference=locationsY[index1]-locationsY[index2];

            // double distance = sqrt( xDifference*xDifference + yDifference*yDifference );

            //Radius of Earth: 6371 km
            double distance;
            
            if(CHORDAL_DISTANCE_FLAG)
                distance = 6371 * euclidean_distance(x[index1], y[index1], z[index1],
                                                        x[index2], y[index2], z[index2]);
            else
                distance = 6371 * great_circle_dist(lon[index1], lon[index2],
                                                       lat[index1], lat[index2]);
            // if (distance < 1e-6)
                // offDiagonal = sill + nugget;
            // else 
            double offDiagonal = sill*exp(-distance/range);

            covarianceMatrix[index1*nLocations+index2] = offDiagonal;
            covarianceMatrix[index2*nLocations+index1] = offDiagonal;
        }
        covarianceMatrix[index1*nLocations+index1] = diagonal;
    }
}

/*
Get nonstationary cross-covariance matrix between two location sets

Input:
    lon1: lon coordinates of location set 1
    lat1: lat coordinates of location set 1
    x1: x coordinates of location set 1
    y1: y coordinates of location set 1
    z1: z coordinates of location set 1
    lon2: lon coordinates of location set 2
    lat2: lat coordinates of location set 2
    x2: x coordinates of location set 2
    y2: y coordinates of location set 2
    z2: z coordinates of location set 2
    nLocations1: the number of locations in lcs1
    nLocations2: the number of locations in lcs2
    nonstat_cov: the nonstationary covariance model
    
Output:
    covarianceMatrix: obtained covariances, of dimension nLocations2 * nLocations1
*/
void evaluate_nonstationary_cross_covariance(double *covarianceMatrix,
double *lon1, double *lat1, double *x1, double *y1, double *z1,
double *lon2, double *lat2, double *x2, double *y2, double *z2,
const unsigned long &nLocations1, const unsigned long &nLocations2, const double &sill_scale, const double &range_scale, const double &nugget_scale)
{
    double *sigmasq1_all = new double [nLocations1];
    double *beta1_all = new double [nLocations1];

    double *sigmasq2_all = new double [nLocations2];
    double *beta2_all = new double [nLocations2];

    if(CHORDAL_DISTANCE_FLAG)
    {
        for(unsigned long index1 = 0; index1 < nLocations1; index1++)
        {            
            sigmasq1_all[index1] = sill_scale * nonstat_cov->get_sigmasq(x1[index1],y1[index1],z1[index1]);
            beta1_all[index1] = range_scale * nonstat_cov->get_beta(x1[index1],y1[index1],z1[index1]);
        }
        for(unsigned long index2 = 0; index2 < nLocations2; index2++)
        {            
            sigmasq2_all[index2] = sill_scale * nonstat_cov->get_sigmasq(x2[index2],y2[index2],z2[index2]);
            beta2_all[index2] = range_scale * nonstat_cov->get_beta(x2[index2],y2[index2],z2[index2]);
        }   
    }else
    {
        for(unsigned long index1 = 0; index1 < nLocations1; index1++)
        {
            sigmasq1_all[index1] = sill_scale * nonstat_cov->get_sigmasq(lon1[index1],lat1[index1]);
            beta1_all[index1] = range_scale * nonstat_cov->get_beta(lon1[index1],lat1[index1]);
        }
        for(unsigned long index2 = 0; index2 < nLocations2; index2++)
        {            
            sigmasq2_all[index2] = sill_scale * nonstat_cov->get_sigmasq(lon2[index2],lat2[index2]);
            beta2_all[index2] = range_scale * nonstat_cov->get_beta(lon2[index2],lat2[index2]);
        } 
    }
    
    for(unsigned long index1 = 0; index1 < nLocations1; index1++)
    {
        double distance, sigmasq1, sigmasq2, beta1, beta2;

        sigmasq1 = sigmasq1_all[index1];
        beta1 = beta1_all[index1];
        
        // double tausq1 = nonstat_cov->get_tausq(lon1[index1],lat1[index1]);

        for(unsigned long index2 = 0; index2 < nLocations2; index2++)
        {
            sigmasq2 = sigmasq2_all[index2];
            beta2 = beta2_all[index2];
 
            //Radius of Earth: 6371 km
            if(CHORDAL_DISTANCE_FLAG)
            {
                distance = 6371 * euclidean_distance(x1[index1], y1[index1], z1[index1],
                                                        x2[index2], y2[index2], z2[index2]);
            }
            else
            {
                distance = 6371 * great_circle_dist(lon1[index1], lon2[index2],
                                                       lat1[index1], lat2[index2]);
            } 

            double beta_mean_squared = (beta1 * beta1 + beta2 * beta2) / 2;
            double beta_product = beta1 * beta2;

            // double covar = sqrt(sigmasq1 * sigmasq2) * beta_product / beta_mean_squared;
            double covar = sqrt(sigmasq1 * sigmasq2) * pow(beta_product / beta_mean_squared, 1.5);
            covar *= exp(-distance / sqrt(beta_mean_squared));

            covarianceMatrix[index1*nLocations2+index2] = covar;
        }
    }

    delete[] sigmasq1_all;
    delete[] beta1_all;
    
    delete[] sigmasq2_all;
    delete[] beta2_all;
}

/*
Get nonstationary variance-covariance matrix for one location set

Input:
    lon: lon coordinates
    lat: lat coordinates
    nLocations: the number of locations
    nonstat_cov: the nonstationary covariance model

Output:
    covarianceMatrix: obtained covariances, of dimension nLocations * nLocations
*/
void evaluate_nonstationary_variance_covariance(double *covarianceMatrix, double *lon, double *lat, double *x, double *y, double *z, const unsigned long &nLocations, const double &sill_scale, const double &range_scale, const double &nugget_scale)
{
    double *sigmasq = new double [nLocations];
    double *beta = new double [nLocations];
    double *tausq = new double [nLocations];
    
    if(CHORDAL_DISTANCE_FLAG)
    {
        for(unsigned long index = 0; index < nLocations; index++)
        {
            sigmasq[index] = sill_scale * nonstat_cov->get_sigmasq(x[index],y[index],z[index]);
            beta[index] = range_scale * nonstat_cov->get_beta(x[index],y[index],z[index]);
            tausq[index] = nugget_scale * nonstat_cov->get_tausq(x[index],y[index],z[index]);
        }
    }else
    {
        for(unsigned long index = 0; index < nLocations; index++)
        {
            sigmasq[index] = sill_scale * nonstat_cov->get_sigmasq(lon[index],lat[index]);
            beta[index] = range_scale * nonstat_cov->get_beta(lon[index],lat[index]);
            tausq[index] = nugget_scale * nonstat_cov->get_tausq(lon[index],lat[index]);
        }
    }

    for(unsigned long index1 = 0; index1 < nLocations; index1++)
    {
        double distance, sigmasq1, beta1, tausq1, sigmasq2, beta2;

        sigmasq1 = sigmasq[index1];
        beta1 = beta[index1];
        tausq1 = tausq[index1];

        for(unsigned long index2 = 0; index2 < index1; index2++)
        {
            sigmasq2 = sigmasq[index2];
            beta2 = beta[index2];

            //Radius of Earth: 6371 km
            if(CHORDAL_DISTANCE_FLAG)
            {
                distance = 6371 * euclidean_distance(x[index1], y[index1], z[index1],
                                                        x[index2], y[index2], z[index2]);
            }
            else
            {
                distance = 6371 * great_circle_dist(lon[index1], lon[index2],
                                                       lat[index1], lat[index2]);
            }

            double beta_mean_squared = (beta1 * beta1 + beta2 * beta2) / 2;
            double beta_product = beta1 * beta2;

            // double covar = sqrt(sigmasq1 * sigmasq2) * beta_product / beta_mean_squared;
            double covar = sqrt(sigmasq1 * sigmasq2) * pow(beta_product / beta_mean_squared, 1.5);
            covar *= exp(-distance / sqrt(beta_mean_squared));

            covarianceMatrix[index1*nLocations+index2] = covar;
            covarianceMatrix[index2*nLocations+index1] = covar;
        }
        covarianceMatrix[index1*nLocations+index1] = sigmasq1 + tausq1;
    }

    delete[] sigmasq;
    delete[] beta;
    delete[] tausq;
}
