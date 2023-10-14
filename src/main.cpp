#include <iostream>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>
#include <omp.h>

#include "class_data.hpp"
#include "class_partition.hpp"
#include "class_nonstationary_covariance.hpp"
#include "class_approximation.hpp"
#include "read_user_parameters.hpp"
#include "constants.hpp"
#include "optimization.hpp"

void initialize_MPI(int argc, char* argv[])
{
	//Get current time
	time_t currentTime;
	time(&currentTime);

	gettimeofday(&timeBegin, NULL);
	
	//Initialize MPI with multi-threads
	int tmp;
	MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&tmp);
	
	int isMPIInitialized;
	MPI_Initialized(&isMPIInitialized);

	MPI_Comm_size(MPI_COMM_WORLD, &MPI_SIZE);
	MPI_Comm_rank(MPI_COMM_WORLD, &WORKER);

	if(isMPIInitialized)
	{
		if(WORKER == 0)
		{
			cout<<"\n===========================> Program starts at: "<<ctime(&currentTime);
			cout<<"\n===========================> MPI is initialized with "<<MPI_SIZE<<" MPI process(es), each of which has "<<omp_get_max_threads()<<" OpenMP thread(s).\n\n";
		}
	}
	else
	{
		cout<<"Program exits with an error: something went wrong with initializing MPI.\n";
		exit(EXIT_FAILURE);
	}
}

void finalize_MPI()
{
	delete[] WORKERS_FOR_EACH_REGION;
	delete[] WORKING_REGION_FLAG;

	delete data;
	delete partition;

	if(WORKER == 0)
	{
		gettimeofday(&timeNow, NULL);
		cout<<"===========================> Processor 1: Before Finalizing. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n";
	}

	MPI_Finalize();

	int isMPIFinalized;
	MPI_Finalized(&isMPIFinalized);

	if(isMPIFinalized)
	{
		if(WORKER == 0)
		{
			gettimeofday(&timeNow, NULL);
			cout<<"===========================> Processor 1: MPI is finalized. Elapsed time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n\n";
		}
	}
	else
	{
		cout<<"Program exits with an error: something went wrong with finalizing MPI.\n";
		exit(EXIT_FAILURE);
	}

	if(WORKER == 0)
	{
		time_t currentTime;

		time(&currentTime);
		gettimeofday(&timeNow, NULL);

		cout<<"===========================> Program ends at: "<<ctime(&currentTime);
		cout<<"===========================> Running time: "<<(double)timeNow.tv_sec-(double)timeBegin.tv_sec+((double)timeNow.tv_usec-(double)timeBegin.tv_usec)/1000000.0<<" seconds.\n\n";
	}
}

int main(int argc, char* argv[])
{
	//Initialize MPI
	initialize_MPI(argc, argv);

	//Read user parameters
	read_user_parameters();

	data = new Data;
	data->load_data();

	//Load nonstationary covariance model if desired
	if(NONSTATIONARY_FLAG)
	{
		nonstat_cov = new Nonstationary_covariance;
		nonstat_cov->load_data();
	}

	//Build partition
	partition = new Partition;
	partition->build_partition();

	//Do the calculation mode
	if(CALCULATION_MODE == "build_structure_only") 
	{
		partition->dump_structure_information();
		cout<<"===========================> NOTICE: \"structure_information.txt\" has been generated in the current directory to show the details of the built structure."<<endl<<endl;
	}
	
	if(CALCULATION_MODE == "likelihood")
	{
		Approximation approximation(SIGMASQ, BETA, TAUSQ, SIGMASQ_SCALE, BETA_SCALE, TAUSQ_SCALE);
		approximation.likelihood();
		
	}
	
	if(CALCULATION_MODE=="prediction")
	{
		Approximation approximation(SIGMASQ, BETA, TAUSQ, SIGMASQ_SCALE, BETA_SCALE, TAUSQ_SCALE);
		approximation.likelihood();
		approximation.predict();
		if(DUMP_PREDICTION_RESULTS_FLAG) approximation.dump_prediction_result();
	}

	if(CALCULATION_MODE == "optimization")
	{
		if(NONSTATIONARY_FLAG)
			get_optimal_nonstationary_parameters();
		else
			get_optimal_parameters();
	}

	//Finalize MPI
	finalize_MPI();
	return 0;
}
