#pragma once

#include <sys/time.h>
#include <class_data.hpp>
#include <set>

using namespace arma;

class Partition
/*
Declare the class of Partition, which contains all the information about the hierarchical structure of the partition
*/
{
    public:

    //nRegionsInTotal: the total number of regions
    unsigned long nRegionsInTotal;
    
    //nKnotsX: the nubmer of knots in each region before the finest level
    int nKnots;

    //nRegionsAtEachLevel: the array of the number of regions in each level
    unsigned long* nRegionsAtEachLevel;

    //Array of coordinates for knots in each region with dimension [nRegionsInTotal]*[nKnots or TBD]
    double **knotsX, **knotsY, **knotsZ, **knotsLon, **knotsLat;

    //Array of vectors of residual values for knots in each region in the finest level with dimension [nRegionsAtEachLevel[NUM_LEVELS_M-1]]. Used when observations are used as knots in the finest level
    double **knotsResidual;

    //Array of coordinates for predictions in each region in the finest level with dimension [nRegionsAtEachLevel[NUM_LEVELS_M-1]]*[TBD]
    double **predictionX, **predictionY, **predictionZ, **predictionLon, **predictionLat;

    //When observations are NOT used as knots in the finest level
        //Array of vectors of residual values in each region in the finest level with dimension [nRegionsAtEachLevel[NUM_LEVELS_M-1]].
        double **residual;
        //Array of coordinates for observations in each region in the finest level with dimension [nRegionsAtEachLevel[NUM_LEVELS_M-1]]*[TBD]
        double **observationX, **observationY, **observationZ, **observationLon, **observationLat;

        unsigned long *nObservationsAtFinestLevel;

    //nKnotsAtFinestLevel: array of numbers of knots in each region in the finest level with dimension nRegionsAtEachLevel[NUM_LEVELS_M-1]
    //nPredictionsAtFinestLevel: array of numbers of prediction locations in each region in the finest level with dimension nRegionsAtEachLevel[NUM_LEVELS_M-1]
    unsigned long *nKnotsAtFinestLevel, *nPredictionsAtFinestLevel;

    //Array of the original worker rank for all regions
    int* origninalWorker;

    //indexStartFinestLevel: the region index of the first region at the finest level
    //indexStartFinestLevel: the region index of the first region at the second finest level
    unsigned long indexStartFinestLevel, indexStartSecondFinestLevel;
    
    //nRegionsAtFinestLevel: the number of regions in the finest level
    unsigned long nRegionsAtFinestLevel;

    //parent: the parent region index of regions at the finest level
    unsigned long *parent;

    //childrenStart: the region index of the first children of regions at the second finest level
    //childrenEnd: the region index of the first children of regions at the second finest level
    unsigned long *childrenStart, *childrenEnd;

    //Implement the constructor of Partition. Assign values to nRegionsAtEachLevel for the array of the number of regions in each level, nRegionsInTotal for the total number of regions, and MPI quantities.
    Partition();

    //Destructor
    ~Partition();

    //Build the hierarchical partition
    void build_partition();

    //Assign the work load for each worker by dynamic scheduling
    void dynamic_schedule();

    //Show a brief summary of the partition
    void print_partition_summary();

    //Dump structure information to "structure_information.txt"
    void dump_structure_information();

    private:
    /*
    Creats knots in the current region
    Input: 
        xMin, xMax, yMin, yMax: the boudary of the current region
        nKnotsX, nKnotsY: the number of knots in the current region in each direction
    Output: 
        currentKnotsX: the array of the coordinates in longitude for each knot
        currentKnotsY: the array of the coordinates in latitude for each knot
    */
    void create_knots(double *&currentKnotsX, double *&currentKnotsY, const double &xMin, const double &xMax, const int &nKnotsX, const double &yMin, const double &yMax, const int &nKnotsY);

    /*   
    Creats partitions in the current region
    */
    void create_partitions(unsigned long region, double *partitionXMin, double *partitionXMax, double *partitionYMin, double *partitionYMax);

    /*
    Build partitions by this program
    */
    void build_partition_by_program();

    /*
    Build partitions by user-provided knots files
    */
    void build_partition_by_user_files();

    /*
    Read parent region indices of the regions in the finest level.
    */
    void read_parent_region_index();
};