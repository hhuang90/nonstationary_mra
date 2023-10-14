#pragma once

#include <mpi.h>
#include <set>

#include "class_data.hpp"
#include "class_partition.hpp"
#include "class_nonstationary_covariance.hpp"

//User parameters
    //Name for the data file.
    extern std::string DATA_FILE_NAME;

    //The flag for whether to eliminate duplicates in raw data. If it is guaranteed that there are no duplicates in the raw data, set the flag to "false" to save computation time.
    extern bool ELIMINATION_DUPLICATES_FLAG;

    // The flag for whether to use nonstationary covariance model
	extern bool NONSTATIONARY_FLAG;

    //The flag for whether to perform linear regression with respect to longitudes, latitudes, and an intercept. in the raw data.Must be either "true" or "false".
    extern bool REGRESSION_FLAG;

    //The ratio of the smallest distance between any knot and the region boundary to the length of the region in each dimension. Requires a numerical value or "default" to use the default value exp/100.
    extern double OFFSET;

    //The number of partitions in each level. Requires to be either 2 or 4.
    extern int NUM_PARTITIONS_J;

    //The number of knots in each region before the finest resolution level. The number of knots at the finest resolution level is automatically determined by the data and the built structure, and the number can be different in different regions at the finest resolution level.
    extern int NUM_KNOTS_r;

    //The number of levels. Requires a positive integer or "default" to use the number that is automatically determined from NUM_PARTITIONS_J and NUM_KNOTS_r by making the average number of observations per region at the finest resolution level similar to the number of knots in regions before the finest resolution level.
    extern int NUM_LEVELS_M;

    //The type of calculations. Must be one of "prediction", "optimization", "likelihood", "build_structure_only".
    extern std::string CALCULATION_MODE;

    //The flag for whether to print the detailed information when running the code.
    extern bool PRINT_DETAIL_FLAG;

    //The way of specifying the prediction locations. Must be one of 'D' for all the locations in DATA_FILE_NAME no matter whether the associated observation is a valid value or NaN, 'N' for locations in DATA_FILE_NAME that have NaN values, 'A' for locations specified in PREDICTION_LOCATION_FILE.
    extern char PREDICTION_LOCATION_MODE;

    //Name for the prediction location file. Only used when PREDICTION_LOCATION_MODE='A'.
    extern std::string PREDICTION_LOCATION_FILE;

    //The flag for whether to dump prediction results to output file.
    extern bool	DUMP_PREDICTION_RESULTS_FLAG;

    //The output file for storing prediction results. Only used when DUMP_PREDICTION_RESULTS_FLAG="true" and CALCULATION_MODE="predict".
    extern std::string PREDICTION_RESULTS_FILE_NAME;

    //The flag for whether to save temporary files to disk.
	extern bool SAVE_TO_DISK_FLAG;

    //The flag for the dynamic scheduling of workload for each worker
    extern bool DYNAMIC_SCHEDULE_FLAG;

    //The directory for saving the temporary files on disk. Only used when "SAVE_TO_DISK"="true".
	extern std::string TMP_DIRECTORY;

    //Sill parameter in the covariance function. Requires to be a positive numerical value. Will be ignored if CALCULATION_MODE="optimization" where the optimal values that maximizes the likelihood will be found.
	extern double SIGMASQ;

    //Range parameter in the covariance function. Requires to be a positive numerical value. Will be ignored if CALCULATION_MODE="optimization" where the optimal values that maximizes the likelihood will be found.
	extern double BETA;

    //Nugget parameter in the covariance function. Requires to be a positive numerical value. Will be ignored if CALCULATION_MODE="optimization" where the optimal values that maximizes the likelihood will be found.
	extern double TAUSQ;

    //Sill multiplicative factor in the covariance function. Requires to be a positive numerical value. Will be ignored if CALCULATION_MODE="optimization" where the optimal values that maximizes the likelihood will be found.
	extern double SIGMASQ_SCALE;

    //Range multiplicative factor in the covariance function. Requires to be a positive numerical value. Will be ignored if CALCULATION_MODE="optimization" where the optimal values that maximizes the likelihood will be found.
	extern double BETA_SCALE;

    //Nugget multiplicative factor in the covariance function. Requires to be a positive numerical value. Will be ignored if CALCULATION_MODE="optimization" where the optimal values that maximizes the likelihood will be found.
	extern double TAUSQ_SCALE;

    //Maximum number of iterations allowed in the optimization. Only used when CALCULATION_MODE="optimization".
	extern int MAX_ITERATIONS;

    //Initial guess of sill parameter in the covariance function in optimization. Only used when CALCULATION_MODE="optimization".
	extern double SIGMASQ_INITIAL_GUESS;

    //Lower bound for sill parameter in the covariance function in optimization. Only used when CALCULATION_MODE="optimization".
	extern double SIGMASQ_LOWER_BOUND;

    //Upper bound for sill parameter in the covariance function in optimization. Only used when CALCULATION_MODE="optimization".
	extern double SIGMASQ_UPPER_BOUND;

    //Initial guess of range parameter in the covariance function in optimization. Only used when CALCULATION_MODE="optimization".
	extern double BETA_INITIAL_GUESS;

    //Lower bound for range parameter in the covariance function in optimization. Only used when CALCULATION_MODE="optimization".
	extern double BETA_LOWER_BOUND;

    //Upper bound for range parameter in the covariance function in optimization. Only used when CALCULATION_MODE="optimization".
	extern double BETA_UPPER_BOUND;

    //Initial guess of nugget parameter in the covariance function in optimization. Only used when CALCULATION_MODE="optimization".
	extern double TAUSQ_INITIAL_GUESS;

    //Lower bound for nugget parameter in the covariance function in optimization. Only used when CALCULATION_MODE="optimization".
	extern double TAUSQ_LOWER_BOUND;

    //Upper bound for nugget parameter in the covariance function in optimization. Only used when CALCULATION_MODE="optimization".
	extern double TAUSQ_UPPER_BOUND;

    //Nonstarionary sill parameter file. It should be a binary file. The first number is the number of basis functions n_sill that is stored as an unsigned 64-bit integer. The second number is the intercept as a 64-bit real number. Following are n_sill longitudes, n_sill latitudes, and n_sill weights for each basis function, each element of which is stored as a 64-bit real number. The formula is log(sigmasq) = intercept + weight_1 * basis_1 + ... + weight_{n_sill} * basis_{n_sill}
	extern std::string SIGMASQ_FILE_NAME;

    //Nonstarionary range parameter file. It should be a binary file. The first number is the number of basis functions n_range that is stored as an unsigned 64-bit integer. The second number is the intercept as a 64-bit real number. Following are n_range longitudes, n_range latitudes, and n_range weights for each basis function, each element of which is stored as a 64-bit real number. The formula is log(beta) = intercept + weight_1 * basis_1 + ... + weight_{n_range} * basis_{n_range}
	extern std::string BETA_FILE_NAME;

    //Nonstarionary nugget parameter file. It should be a binary file. The first number is the number of basis functions n_nugget that is stored as an unsigned 64-bit integer. The second number is the intercept as a 64-bit real number. Following are n_nugget longitudes, n_nugget latitudes, and n_nugget weights for each basis function, each element of which is stored as a 64-bit real number. The formula is log(tausq) = intercept + weight_1 * basis_1 + ... + weight_{n_nugget} * basis_{n_nugget}
	extern std::string TAUSQ_FILE_NAME;

    /////// Update
    
    //The flag for whether to use 3D Euclidean space
	extern bool CHORDAL_DISTANCE_FLAG;

    //The flag for whether to place knots by provided files
	extern bool PROVIDED_KNOTS_FLAG;

	//The file name for the provided knots. The first number is the total number of knots nr that is stored as an unsigned 64-bit integer. Following are nr x, nr y, nr z coordinates, and nr region indices, each element of which is stored as a 64-bit real number. Only be used if PROVIDED_KNOTS_FLAG is "true".
	extern std::string	KNOTS_FILE_NAME;
	
    //The flag for whether to use the observation locations as knots at the finest level. If "false", the knots at the finest level must be provided in KNOTS_FILE_NAME.
	extern bool OBS_AS_KNOTS_FLAG;

	//The number of knots in each region at the finest resolution level. Only used when OBS_AS_KNOTS_FLAG is "true".
	extern int NUM_KNOTS_FINEST;

    //The flag for whether to have various number of regions in the finest level
	extern bool VARIOUS_J_FINEST_LEVEL_FLAG;

    //The file for the parent region indices of the regions in the finest level. The first (region_min) and second (region_max) number is the minimum and maximum region indices in the finest level, stored as a 64-bit real number. Then (region_max - region_min + 1) parent region indices of the regions in the finest level are followed, stored as unsigned 64-bit integers. Only be used if VARIOUS_J_FINEST_LEVEL_FLAG is "true".
	extern std::string PARENT_REGION_FILE_NAME;

    //Length of Wendland basis function support. Only used when NONSTATIONARY_FLAG=true
	extern double WENDLAND_LEN;
    
//Other constants
    //Data instance
    extern Data *data;

    //Partition instance
    extern Partition *partition;

    //Nonstationary covariance model instance
    extern Nonstationary_covariance *nonstat_cov;
    
    //The total number of unique observation locations that have valid measurements (i.e, not NaN)
    extern unsigned long NUM_OBSERVATIONS;

    //The total number of unique prediction locations
    extern unsigned long NUM_PREDICTIONS;

    //Iteration for optimization
    extern int OPTIMIZATION_ITERATION;

    //Variables for showing execution time
    extern timeval timeBegin, timeNow;

//MPI constants
    //The total size of workers in MPI
    extern int MPI_SIZE;

    //The worker rank
    extern int WORKER;

    //MPI communicator
    extern MPI_Comm WORLD;

    //The starting index of region at each level
    extern unsigned long *REGION_START;

    //The ending index of region at each level
    extern unsigned long *REGION_END;

    //The flag for whether the region is handled by this region
    extern bool * WORKING_REGION_FLAG;

    //The indices of regions at the current level
    extern std::set<unsigned long> INDICES_REGIONS_AT_CURRENT_LEVEL;

    //The indices of regions at the level that is one level below the current level
    extern std::set<unsigned long> INDICES_REGIONS_AT_ONE_LEVEL_BELOW;
    
    //The array of set of worker ranks for each region before the finest level that is dealt with by the underlying worker
    extern std::set<unsigned long>* WORKERS_FOR_EACH_REGION;

    //The tag for likelihood for MPI communication
    extern const int MPI_TAG_LIKELIHOOD;

    //The tag for w for MPI communication
    extern const int MPI_TAG_VEC;

    //The tag for A for MPI communication
    extern const int MPI_TAG_MAT;

    //The tag for dynamic scheduling for MPI communication
    extern const int MPI_TAG_DYNAMIC_SCHEDULE;