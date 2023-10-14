#include <iostream>
#include <mpi.h>

#include "dlib/optimization.h"
#include "dlib/global_optimization.h"

#include "class_data.hpp"
#include "class_partition.hpp"
#include "class_approximation.hpp"
#include "constants.hpp"
#include "optimization.hpp"

double loglikelihood=0;

double objective(const dlib::matrix<double,0,1>& parameter)
{
	double sigmasq=parameter(0);
	double beta=parameter(1);
	double tausq=parameter(2);

	OPTIMIZATION_ITERATION++;
	if(WORKER==0)
	{
		cout<<"##########################################################################\n";
		cout<<"############## Iteration "<<OPTIMIZATION_ITERATION<<". sigmasq: "<<sigmasq<<", beta: "<<beta<<", tausq: "<<tausq<<endl;
		cout<<"##########################################################################\n\n";
	}

	//Reset working region information
	INDICES_REGIONS_AT_CURRENT_LEVEL.clear();
	for(unsigned long iRegion = REGION_START[NUM_LEVELS_M-1]; iRegion < REGION_END[NUM_LEVELS_M-1]+1; iRegion++)
	{
		unsigned long ancestor = iRegion;
		WORKING_REGION_FLAG[ancestor] = true;

		if(VARIOUS_J_FINEST_LEVEL_FLAG)
			ancestor = partition->parent[iRegion - partition->indexStartFinestLevel];
		else
			ancestor = (iRegion-1)/NUM_PARTITIONS_J;

		INDICES_REGIONS_AT_CURRENT_LEVEL.insert(ancestor);
		WORKING_REGION_FLAG[ancestor] = true;

		while(ancestor > 0)
		{
			ancestor = (ancestor-1)/NUM_PARTITIONS_J;
			WORKING_REGION_FLAG[ancestor] = true;
		}
	}

	MPI_Barrier(WORLD);
	Approximation approximation(sigmasq, beta, tausq, 0.0, 0.0, 0.0);
	approximation.likelihood();

	MPI_Bcast(&approximation.loglikelihood,1,MPI_DOUBLE,0,WORLD);
	MPI_Barrier(WORLD);

	//printf("\nIter %d, WORKER %d: %16.16lf %16.16lf %16.16lf %16.16lf\n",OPTIMIZATION_ITERATION+1, WORKER,sigmasq,beta,tausq, approximation.loglikelihood);

	loglikelihood=-0.5*(approximation.loglikelihood+NUM_OBSERVATIONS*log(2*3.14159265359));

	return approximation.loglikelihood;
}


void get_optimal_parameters()
{
	OPTIMIZATION_ITERATION = 0;
	
	dlib::matrix<double,0,1> initialGuess={SIGMASQ_INITIAL_GUESS,BETA_INITIAL_GUESS,TAUSQ_INITIAL_GUESS};
	dlib::matrix<double,0,1> lowerBound = {SIGMASQ_LOWER_BOUND,BETA_LOWER_BOUND,TAUSQ_LOWER_BOUND};
	dlib::matrix<double,0,1> upperBound={SIGMASQ_UPPER_BOUND,BETA_UPPER_BOUND,TAUSQ_UPPER_BOUND};

	// double rhoBegin = 0.49*min(upperBound-lowerBound);


	bool optimizationSuccess=false;
	try
	{
		dlib::find_min_bobyqa(objective,initialGuess,7,lowerBound,upperBound,0.0001,1e-8,MAX_ITERATIONS);
		optimizationSuccess = true;

	}
	catch (dlib::bobyqa_failure error)
	{
		if(WORKER==0)
		{
			cout<<"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n";
			cout<<"\\\\\\\\\\\\\\The optimal parameter values have not been found. Reason: "<<error.what()<<endl;
		}
	}

	if(optimizationSuccess) 
	{
		if(WORKER==0)
		{
			cout<<"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n";
			cout<<"\\\\\\\\\\\\\\The optimal parameter values are, sigmasq: "<<initialGuess(0)<<", beta: "<<initialGuess(1)<<", tausq: "<<initialGuess(2)<<"; The maximum log likelihood is "<<loglikelihood<<".\n";
		}
	}

	if(WORKER==0) cout<<"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n\n";
}



double objective_nonstationary(const dlib::matrix<double,0,1>& parameter)
{
	double sigmasq_scale = parameter(0);
	double beta_scale = parameter(1);
	double tausq_scale = parameter(2);

	OPTIMIZATION_ITERATION++;
	if(WORKER==0)
	{
		cout<<"##########################################################################\n";
		cout<<"############## Iteration "<<OPTIMIZATION_ITERATION<<". sigmasq multiplication: "<<sigmasq_scale<<", beta multiplication: "<<beta_scale<<", tausq multiplication: "<<tausq_scale<<endl;
		cout<<"##########################################################################\n\n";
	}

	//Reset working region information
	INDICES_REGIONS_AT_CURRENT_LEVEL.clear();
	for(unsigned long iRegion = REGION_START[NUM_LEVELS_M-1]; iRegion < REGION_END[NUM_LEVELS_M-1]+1; iRegion++)
	{
		unsigned long ancestor = iRegion;
		WORKING_REGION_FLAG[ancestor] = true;

		if(VARIOUS_J_FINEST_LEVEL_FLAG)
			ancestor = partition->parent[iRegion - partition->indexStartFinestLevel];
		else
			ancestor = (iRegion-1)/NUM_PARTITIONS_J;

		INDICES_REGIONS_AT_CURRENT_LEVEL.insert(ancestor);
		WORKING_REGION_FLAG[ancestor] = true;

		while(ancestor > 0)
		{
			ancestor = (ancestor-1)/NUM_PARTITIONS_J;
			WORKING_REGION_FLAG[ancestor] = true;
		}
	}

	MPI_Barrier(WORLD);
	Approximation approximation(0.0, 0.0, 0.0, sigmasq_scale, beta_scale, tausq_scale);
	approximation.likelihood();

	MPI_Bcast(&approximation.loglikelihood,1,MPI_DOUBLE,0,WORLD);
	MPI_Barrier(WORLD);

	//printf("\nIter %d, WORKER %d: %16.16lf %16.16lf %16.16lf %16.16lf\n",OPTIMIZATION_ITERATION+1, WORKER,sigmasq,beta,tausq, approximation.loglikelihood);

	loglikelihood=-0.5*(approximation.loglikelihood+NUM_OBSERVATIONS*log(2*3.14159265359));

	return approximation.loglikelihood;
}


void get_optimal_nonstationary_parameters()
{
	OPTIMIZATION_ITERATION = 0;
	
	dlib::matrix<double,0,1> initialGuess={1.0,1.0,1.0};
	dlib::matrix<double,0,1> lowerBound = {1e-2, 1e-2, 1e-2};
	dlib::matrix<double,0,1> upperBound={1e2, 1e2, 1e2};

	bool optimizationSuccess=false;
	try
	{
		dlib::find_min_bobyqa(objective_nonstationary, initialGuess, 7, lowerBound, upperBound,0.1,1e-6,MAX_ITERATIONS);
		optimizationSuccess = true;

	}
	catch (dlib::bobyqa_failure error)
	{
		if(WORKER==0)
		{
			cout<<"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n";
			cout<<"\\\\\\\\\\\\\\The optimal parameter values have not been found. Reason: "<<error.what()<<endl;
		}
	}

	if(optimizationSuccess) 
	{
		if(WORKER==0)
		{
			cout<<"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n";
			cout<<"\\\\\\\\\\\\\\The optimal parameter values are, sigmasq multiplication: "<<initialGuess(0)<<", beta multiplication: "<<initialGuess(1)<<", tausq multiplication: "<<initialGuess(2)<<"; The maximum log likelihood is "<<loglikelihood<<".\n";
		}
	}

	if(WORKER==0) cout<<"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n\n";
}
