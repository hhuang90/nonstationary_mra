#pragma once

#include <armadillo>
#include "class_nonstationary_covariance.hpp"

using namespace arma;

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
    covarianceMatrix: obtained covariances
*/
void evaluate_cross_covariance(double *covarianceMatrix,
double *lon1, double *lat1, double *x1, double *y1, double *z1,
double *lon2, double *lat2, double *x2, double *y2, double *z2,
const unsigned long &nLocations1, const unsigned long &nLocations2, const double &sill, const double &range, const double &nugget);

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
void evaluate_variance_covariance(double *covarianceMatrix, double *lon, double *lat, double *x, double *y, double *z,
const unsigned long &nLocations, const double &sill, const double &range, const double &nugget);

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
const unsigned long &nLocations1, const unsigned long &nLocations2, const double &sill_scale, const double &range_scale, const double &nugget_scale);

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
void evaluate_nonstationary_variance_covariance(double *covarianceMatrix, double *lon, double *lat, double *x, double *y, double *z, const unsigned long &nLocations, const double &sill_scale, const double &range_scale, const double &nugget_scale);