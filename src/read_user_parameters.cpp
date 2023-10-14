#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>

#include "read_user_parameters.hpp"
#include "constants.hpp"
#include "sys/stat.h"

void read_from_file()
{
	std::ifstream file;
    file.open("user_parameters");

    if(file.is_open())
	{
		std::string line;
		while(getline(file,line))
		{
			unsigned int current_char_index = 0;

			//Skip leading space and tab
			while(line[current_char_index] == ' ' || line[current_char_index] == '\t') current_char_index++;

			//Skip comment lines
			if(line[current_char_index] == '#' || line[current_char_index] == 0) continue;

			std::string name = "";
			std::string value = "";
			bool nameOrValue = true; //true for name and false for value
			for(unsigned int iCharacter = current_char_index; iCharacter < line.length(); iCharacter++)
			{
				//Skip irrelevant symbols including space, tab, ', and "
				if(line[iCharacter] == ' ' || line[iCharacter] == '\t' || line[iCharacter] == 39 || line[iCharacter] == 34) continue;

				//Check if reading name or value
				if(line[iCharacter] == '=')
				{
					nameOrValue = false;
					continue;
				}

				//Get the string for name or value
				if(nameOrValue)
					name += line[iCharacter];
				else
					value += line[iCharacter];
			}

			if(name=="") continue;

			//Assign value to name
			if(name == "DATA_FILE_NAME")
			{
				DATA_FILE_NAME = value;
				continue;
			}
			if(name == "PREDICTION_RESULTS_FILE_NAME")
			{
				PREDICTION_RESULTS_FILE_NAME = value;
				continue;
			}
			if(name == "SIGMASQ_FILE_NAME")
			{
				SIGMASQ_FILE_NAME = value;
				continue;
			}
			if(name == "BETA_FILE_NAME")
			{
				BETA_FILE_NAME = value;
				continue;
			}
			if(name == "TAUSQ_FILE_NAME")
			{
				TAUSQ_FILE_NAME = value;
				continue;
			}
			if(name == "CALCULATION_MODE")
			{
				CALCULATION_MODE = value;
				continue;
			}
			if(name == "PREDICTION_LOCATION_MODE")
			{
				PREDICTION_LOCATION_MODE = value[0];
				continue;
			}
			if(name == "PREDICTION_LOCATION_FILE")
			{
				PREDICTION_LOCATION_FILE = value;
				continue;
			}
			if(name == "NONSTATIONARY_FLAG")
			{
				std::istringstream tmp(value);
				tmp>>std::boolalpha>>NONSTATIONARY_FLAG;
				continue;
			}
			if(name == "PRINT_DETAIL_FLAG")
			{
				std::istringstream tmp(value);
				tmp>>std::boolalpha>>PRINT_DETAIL_FLAG;
				continue;
			}
			if(name == "DUMP_PREDICTION_RESULTS_FLAG")
			{
				std::istringstream tmp(value);
				tmp>>std::boolalpha>>DUMP_PREDICTION_RESULTS_FLAG;
				continue;
			}
			if(name == "SAVE_TO_DISK_FLAG")
			{
				std::istringstream tmp(value);
				tmp>>std::boolalpha>>SAVE_TO_DISK_FLAG;
				continue;
			}
			if(name == "DYNAMIC_SCHEDULE_FLAG")
			{
				std::istringstream tmp(value);
				tmp>>std::boolalpha>>DYNAMIC_SCHEDULE_FLAG;
				continue;
			}
			if(name == "REGRESSION_FLAG")
			{
				std::istringstream tmp(value);
				tmp>>std::boolalpha>>REGRESSION_FLAG;
				continue;
			}
			if(name == "ELIMINATION_DUPLICATES_FLAG")
			{
				std::istringstream tmp(value);
				tmp>>std::boolalpha>>ELIMINATION_DUPLICATES_FLAG;
				continue;
			}
			if(name == "TMP_DIRECTORY")
			{
				TMP_DIRECTORY = value;
				continue;
			}
			if(name == "OFFSET")
			{
				if(value == "default")
				{
					OFFSET = exp(1)/100;
					continue;
				}
				
				std::istringstream tmp(value);
				tmp>>OFFSET;
				
				if(OFFSET<=0 || OFFSET >=1)
				{
					if(WORKER == 0) cout<<"Program exits with an error: the value of OFFSET can only be in (0,1), while the currently assigned value is "<<OFFSET<<".\n";
					MPI_Barrier(MPI_COMM_WORLD);
					MPI_Finalize();
					exit(EXIT_FAILURE);
				}
				continue;
			}
			if(name == "NUM_PARTITIONS_J")
			{
				std::istringstream tmp(value);
				tmp>>NUM_PARTITIONS_J;
				continue;
			}
			if(name == "NUM_KNOTS_r")
			{
				std::istringstream tmp(value);
				tmp>>NUM_KNOTS_r;
				continue;
			}
			if(name == "NUM_LEVELS_M")
			{
				if(value=="default")
					NUM_LEVELS_M = -99;//-99 stands for the default method to automatically determine NUM_LEVELS_M 
				else
				{
					std::istringstream tmp(value);
					tmp>>NUM_LEVELS_M;
				}
				continue;
			}
			if(name == "SIGMASQ")
			{
				std::istringstream tmp(value);
				tmp>>SIGMASQ;
				continue;
			}
			if(name == "BETA")
			{
				std::istringstream tmp(value);
				tmp>>BETA;
				continue;
			}
			if(name == "TAUSQ")
			{
				std::istringstream tmp(value);
				tmp>>TAUSQ;
				continue;
			}
			if(name == "SIGMASQ_SCALE")
			{
				std::istringstream tmp(value);
				tmp>>SIGMASQ_SCALE;
				continue;
			}
			if(name == "BETA_SCALE")
			{
				std::istringstream tmp(value);
				tmp>>BETA_SCALE;
				continue;
			}
			if(name == "TAUSQ_SCALE")
			{
				std::istringstream tmp(value);
				tmp>>TAUSQ_SCALE;
				continue;
			}
			if(name == "MAX_ITERATIONS")
			{
				std::istringstream tmp(value);
				tmp>>MAX_ITERATIONS;
				continue;
			}
			if(name == "SIGMASQ_LOWER_BOUND")
			{
				std::istringstream tmp(value);
				tmp>>SIGMASQ_LOWER_BOUND;
				continue;
			}
			if(name == "BETA_LOWER_BOUND")
			{
				std::istringstream tmp(value);
				tmp>>BETA_LOWER_BOUND;
				continue;
			}
			if(name == "TAUSQ_LOWER_BOUND")
			{
				std::istringstream tmp(value);
				tmp>>TAUSQ_LOWER_BOUND;
				continue;
			}
			if(name == "SIGMASQ_UPPER_BOUND")
			{
				std::istringstream tmp(value);
				tmp>>SIGMASQ_UPPER_BOUND;
				continue;
			}
			if(name == "BETA_UPPER_BOUND")
			{
				std::istringstream tmp(value);
				tmp>>BETA_UPPER_BOUND;
				continue;
			}
			if(name == "TAUSQ_UPPER_BOUND")
			{
				std::istringstream tmp(value);
				tmp>>TAUSQ_UPPER_BOUND;
				continue;
			}			
			if(name == "SIGMASQ_INITIAL_GUESS")
			{
				std::istringstream tmp(value);
				tmp>>SIGMASQ_INITIAL_GUESS;
				continue;
			}
			if(name == "BETA_INITIAL_GUESS")
			{
				std::istringstream tmp(value);
				tmp>>BETA_INITIAL_GUESS;
				continue;
			}
			if(name == "TAUSQ_INITIAL_GUESS")
			{
				std::istringstream tmp(value);
				tmp>>TAUSQ_INITIAL_GUESS;
				continue;
			}

			if(name == "CHORDAL_DISTANCE_FLAG")
			{
				std::istringstream tmp(value);
				tmp>>std::boolalpha>>CHORDAL_DISTANCE_FLAG;
				continue;
			}
			if(name == "PROVIDED_KNOTS_FLAG")
			{
				std::istringstream tmp(value);
				tmp>>std::boolalpha>>PROVIDED_KNOTS_FLAG;
				continue;
			}
			if(name == "KNOTS_FILE_NAME")
			{
				KNOTS_FILE_NAME = value;
				continue;
			}
			if(name == "OBS_AS_KNOTS_FLAG")
			{
				std::istringstream tmp(value);
				tmp>>std::boolalpha>>OBS_AS_KNOTS_FLAG;
				continue;
			}
			if(name == "NUM_KNOTS_FINEST")
			{
				std::istringstream tmp(value);
				tmp>>NUM_KNOTS_FINEST;
				continue;
			}
			if(name == "VARIOUS_J_FINEST_LEVEL_FLAG")
			{
				std::istringstream tmp(value);
				tmp>>std::boolalpha>>VARIOUS_J_FINEST_LEVEL_FLAG;
				continue;
			}
			if(name == "PARENT_REGION_FILE_NAME")
			{
				PARENT_REGION_FILE_NAME = value;
				continue;
			}
			if(name == "WENDLAND_LEN")
			{
				std::istringstream tmp(value);
				tmp>>WENDLAND_LEN;
				continue;
			}
			if(WORKER == 0) cout<<"Program exits with an error: The variable with name \""+name+"\" in the configuration file \"user_parameters\" is not defined. Check again with the variable name by looking at the comments.\n";
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}
	}
	else
	{
		if(WORKER == 0) cout<<"Program exits with an error: fail to open the configuration file \"user_parameters\"\n";
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

}

void print_parameters_summary()
{
	cout<<">>The data file: "<<DATA_FILE_NAME<<endl;
	cout<<"  The calculation mode: "<<CALCULATION_MODE<<endl;

	if(NONSTATIONARY_FLAG)
		cout<<"  The program uses the nonstationary covariance function\n";
	else
		cout<<"  The program uses the stationary covariance function\n";

	if(CHORDAL_DISTANCE_FLAG)
		cout<<"  The chordal distance is used with coordinates: (cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)).\n\n";
	else
		cout<<"  Coordinates are longitude and latitude.\n  The great circle distance is used\n\n";

	if(PROVIDED_KNOTS_FLAG)
		cout<<">>The program uses user-provided knots from file "<<KNOTS_FILE_NAME<<endl;
	else
		cout<<">>The program automatically places knots in each region\n";
	
	cout<<"  The number of levels: "<<NUM_LEVELS_M<<endl;
	cout<<"  The number of subregions in each partitioning: "<<NUM_PARTITIONS_J<<endl;
	cout<<"  The ratio of the smallest distance between any knot and the region boundary to the length of the region in each dimension: "<<OFFSET<<endl;
	cout<<"  The number of knots in each region before the finest resolution level: "<<NUM_KNOTS_r<<endl<<endl;

	if(CALCULATION_MODE != "optimization")
	{
		if(NONSTATIONARY_FLAG)
		{
			cout<<">>The nonstarionary partial sill parameter file: "<<SIGMASQ_FILE_NAME<<endl;
			cout<<"  The nonstarionary range parameter file: "<<BETA_FILE_NAME<<endl;
			cout<<"  The nonstarionary nugget parameter file: "<<TAUSQ_FILE_NAME<<endl<<endl;

			cout<<">>The used partial sill multiplicative factor: "<<SIGMASQ_SCALE<<endl;
			cout<<"  The used range multiplicative factor: "<<BETA_SCALE<<endl;
			cout<<"  The used nugget multiplicative factor: "<<TAUSQ_SCALE<<endl<<endl;
		}else
		{
			cout<<">>The used partial sill parameter: "<<SIGMASQ<<endl;
			cout<<"  The used range parameter: "<<BETA<<endl;
			cout<<"  The used nugget parameter: "<<TAUSQ<<endl<<endl;
		}
	}
	else
	{
		cout<<">>The maximum iterations allowed in the optimization: "<<MAX_ITERATIONS<<endl<<endl;
		if(!NONSTATIONARY_FLAG)
		{
			cout<<"  The initial guess for the partial sill"<<SIGMASQ_INITIAL_GUESS<<" and the optimization interval: ["<<SIGMASQ_LOWER_BOUND<<", "<<SIGMASQ_UPPER_BOUND<<"]\n";
			cout<<"  The initial guess for the range"<<BETA_INITIAL_GUESS<<" and the optimization interval: ["<<BETA_LOWER_BOUND<<", "<<BETA_UPPER_BOUND<<"]\n";
			cout<<"  The initial guess for the nugget"<<TAUSQ_INITIAL_GUESS<<" and the optimization interval: ["<<TAUSQ_LOWER_BOUND<<", "<<TAUSQ_UPPER_BOUND<<"]\n\n";
		}
	}		

	if(CALCULATION_MODE=="prediction")
	{
		if(PREDICTION_LOCATION_MODE == 'D')
			cout<<">>The program predicts at all location in the data file: "<<DATA_FILE_NAME<<endl;
		if(PREDICTION_LOCATION_MODE == 'N')
			cout<<">>The program predicts at all location with NaN values in the data file: "<<DATA_FILE_NAME<<endl;
		if(PREDICTION_LOCATION_MODE == 'A')
			cout<<">>The program predicts at all location in the data file: "<<PREDICTION_LOCATION_FILE<<endl;

		if(DUMP_PREDICTION_RESULTS_FLAG)
			cout<<"  The program saves prediction results to: "<<PREDICTION_RESULTS_FILE_NAME<<endl<<endl;
		else
			cout<<"  The program does NOT save prediction results"<<endl<<endl;
	}

	if(ELIMINATION_DUPLICATES_FLAG)
		cout<<">>The program will automatically eliminate duplicate observations\n\n";
	else
		cout<<">>The program will NOT automatically eliminate duplicate observations\n\n";

	if(REGRESSION_FLAG)
		cout<<">>The program will perform linear regression with respect to longitudes, latitudes, and an intercept\n\n";
	else
		cout<<">>The program will NOT perform linear regression with respect to longitudes, latitudes, and an intercept\n\n";

	if(DYNAMIC_SCHEDULE_FLAG)
		cout<<">>The program uses dynamic scheduling of workload for each worker\n\n";
	else
		cout<<">>The program does NOT use dynamic scheduling of workload for each worker\n\n";

	if(SAVE_TO_DISK_FLAG)
		cout<<">>The program saves temporary files to disk at directory: "<<TMP_DIRECTORY<<endl<<endl;
}

void check_errors()
{
	// if(PROVIDED_KNOTS_FLAG && OBS_AS_KNOTS_FLAG)
	// {
	// 	if(WORKER == 0) cout<<"When PROVIDED_KNOTS_FLAG is true, OBS_AS_KNOTS_FLAG must be false, i.e., the program currently only supports choosing knots in the finest level by the the knots file "<<KNOTS_FILE_NAME<<" (name specified by KNOTS_FILE_NAME).\n";
	// 	MPI_Barrier(MPI_COMM_WORLD);
	// 	MPI_Finalize();
	// 	exit(EXIT_FAILURE);			
	// }
	if(NONSTATIONARY_FLAG && WENDLAND_LEN < 0)
	{
		if(WORKER == 0) cout<<"When NONSTATIONARY_FLAG = true, WENDLAND_LEN is used, whose value must be a positive float number.\n";
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		exit(EXIT_FAILURE);	
	}
	if(!OBS_AS_KNOTS_FLAG)
	{
		if(WORKER == 0) cout<<"Not implemented yet when OBS_AS_KNOTS_FLAG = false.\n";
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		exit(EXIT_FAILURE);		
	}
	if(VARIOUS_J_FINEST_LEVEL_FLAG && !PROVIDED_KNOTS_FLAG)
	{
		if(WORKER == 0) cout<<"When VARIOUS_J_FINEST_LEVEL_FLAG is true, PROVIDED_KNOTS_FLAG must be true, and the file for the parent region indices of the regions in the finest level, "<<KNOTS_FILE_NAME<<" (name specified by PARENT_REGION_FILE_NAME) must be available.\n";
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		exit(EXIT_FAILURE);		
	}
	// if(!PROVIDED_KNOTS_FLAG && !OBS_AS_KNOTS_FLAG)
	// {
	// 	if(WORKER == 0) cout<<"When OBS_AS_KNOTS_FLAG is false, PROVIDED_KNOTS_FLAG must be true, and the knots file "<<KNOTS_FILE_NAME<<" (name specified by KNOTS_FILE_NAME) must be available.\n";
	// 	MPI_Barrier(MPI_COMM_WORLD);
	// 	MPI_Finalize();
	// 	exit(EXIT_FAILURE);			
	// }
	if(NUM_PARTITIONS_J!=2&&NUM_PARTITIONS_J!=4)
	{
		if(WORKER == 0) cout<<"Program exits with an error: the specified NUM_PARTITIONS_J is "<<NUM_PARTITIONS_J<<". Only implemented for NUM_PARTITIONS_J equal to 2 or 4.\n";
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	if(NUM_LEVELS_M < 2 && NUM_LEVELS_M != -99) //-99 stands for the default method to automatically determine NUM_LEVELS_M
	{
		if(WORKER == 0) cout<<"Program exits with an error: the specified NUM_LEVELS_M is "<<NUM_LEVELS_M<<", which is required to be larger than 1.\n";
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	if( CALCULATION_MODE!="prediction" &&
		CALCULATION_MODE!="optimization" &&
		CALCULATION_MODE!="likelihood" &&
		CALCULATION_MODE!="build_structure_only")
	{
		if(WORKER == 0) cout<<"Program exits with an error: the specified CALCULATION_MODE is \""<<CALCULATION_MODE<<"\", which must be one of \"predict\", \"optimization\", \"likelihood\", and \"build_structure_only\".\n";
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	if(CALCULATION_MODE == "build_structure_only" && MPI_SIZE > 1) 
	{
		if(WORKER == 0) cout<<"Program exits with an error: do not use more than one MPI process to show the details of the built structure when setting CALCULATION_MODE to \"build_structure_only\""<<endl;
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	if(PREDICTION_LOCATION_MODE!='D'&&PREDICTION_LOCATION_MODE!='N'&&PREDICTION_LOCATION_MODE!='A')
	{
		if(WORKER == 0) cout<<"Program exits with an error: the specified PREDICTION_LOCATION_MODE is '"<<PREDICTION_LOCATION_MODE<<"', which must be one of 'D', 'N', and 'A'\n";
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Finalize();
		exit(EXIT_FAILURE);
	}

	if(SAVE_TO_DISK_FLAG)
	{
		//Check whether The temporary directory that stores variables in runtime exists
		struct stat directoryStatus;
		if(stat(TMP_DIRECTORY.c_str(),&directoryStatus) == -1)
		{
			if(WORKER == 0) cout<<"Program exits with an error: the directory "<<TMP_DIRECTORY<<" does not exist. Please create it before running the program.\n";
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}
		
		if(!S_ISDIR(directoryStatus.st_mode))
		{
			if(WORKER == 0) cout<<"Program exits with an error: "<<TMP_DIRECTORY<<" is not a directory. Please create a directory with name "<<TMP_DIRECTORY<<" before running the program.\n";
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}
	}
	
	if(CALCULATION_MODE == "optimization")
	{
		if(SIGMASQ_UPPER_BOUND<SIGMASQ_LOWER_BOUND)
		{
			if(WORKER == 0) cout<<"Program exits with an error: the specified SIGMASQ_UPPER_BOUND "<<SIGMASQ_UPPER_BOUND<<" is smaller than the specified SIGMASQ_LOWER_BOUND "<<SIGMASQ_LOWER_BOUND<<".\n";
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}
		if(BETA_UPPER_BOUND<BETA_LOWER_BOUND)
		{
			if(WORKER == 0) cout<<"Program exits with an error: the specified BETA_UPPER_BOUND "<<BETA_UPPER_BOUND<<" is smaller than the specified BETA_LOWER_BOUND "<<BETA_LOWER_BOUND<<".\n";
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}
		if(TAUSQ_UPPER_BOUND<TAUSQ_LOWER_BOUND)
		{
			if(WORKER == 0) cout<<"Program exits with an error: the specified TAUSQ_UPPER_BOUND "<<TAUSQ_UPPER_BOUND<<" is smaller than the specified TAUSQ_LOWER_BOUND "<<TAUSQ_LOWER_BOUND<<".\n";
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}

		if(SIGMASQ_INITIAL_GUESS<SIGMASQ_LOWER_BOUND)
		{
			if(WORKER == 0) cout<<"Program exits with an error: the specified SIGMASQ_INITIAL_GUESS "<<SIGMASQ_INITIAL_GUESS<<" is smaller than the specified SIGMASQ_LOWER_BOUND "<<SIGMASQ_LOWER_BOUND<<".\n";
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}
		if(BETA_INITIAL_GUESS<BETA_LOWER_BOUND)
		{
			if(WORKER == 0) cout<<"Program exits with an error: the specified BETA_UPPER_BOUND "<<BETA_INITIAL_GUESS<<" is smaller than the specified BETA_LOWER_BOUND "<<BETA_LOWER_BOUND<<".\n";
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}
		if(TAUSQ_INITIAL_GUESS<TAUSQ_LOWER_BOUND)
		{
			if(WORKER == 0) cout<<"Program exits with an error: the specified TAUSQ_INITIAL_GUESS "<<TAUSQ_INITIAL_GUESS<<" is smaller than the specified TAUSQ_LOWER_BOUND "<<TAUSQ_LOWER_BOUND<<".\n";
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}

		if(SIGMASQ_INITIAL_GUESS>SIGMASQ_UPPER_BOUND)
		{
			if(WORKER == 0) cout<<"Program exits with an error: the specified SIGMASQ_INITIAL_GUESS "<<SIGMASQ_INITIAL_GUESS<<" is smaller than the specified SIGMASQ_UPPER_BOUND "<<SIGMASQ_UPPER_BOUND<<".\n";
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}
		if(BETA_INITIAL_GUESS>BETA_UPPER_BOUND)
		{
			if(WORKER == 0) cout<<"Program exits with an error: the specified BETA_UPPER_BOUND "<<BETA_INITIAL_GUESS<<" is smaller than the specified BETA_UPPER_BOUND "<<BETA_UPPER_BOUND<<".\n";
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}
		if(TAUSQ_INITIAL_GUESS>TAUSQ_UPPER_BOUND)
		{
			if(WORKER == 0) cout<<"Program exits with an error: the specified TAUSQ_INITIAL_GUESS "<<TAUSQ_INITIAL_GUESS<<" is smaller than the specified TAUSQ_UPPER_BOUND "<<TAUSQ_UPPER_BOUND<<".\n";
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Finalize();
			exit(EXIT_FAILURE);
		}
	}else
	{
		if(NONSTATIONARY_FLAG)
		{
			if(SIGMASQ_SCALE <= 0)
			{
				if(WORKER == 0) cout<<"Program exits with an error: the specified SIGMASQ_SCALE is "<<SIGMASQ_SCALE<<", not positive.\n\n";
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Finalize();
				exit(EXIT_FAILURE);
			}
			if(BETA_SCALE <= 0)
			{
				if(WORKER == 0) cout<<"Program exits with an error: the specified BETA_SCALE is "<<BETA_SCALE<<", not positive.\n\n";
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Finalize();
				exit(EXIT_FAILURE);
			}
			if(TAUSQ_SCALE <= 0)
			{
				if(WORKER == 0) cout<<"Program exits with an error: the specified TAUSQ_SCALE is "<<TAUSQ_SCALE<<", not positive.\n\n";
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Finalize();
				exit(EXIT_FAILURE);
			}
		}else
		{
			if(SIGMASQ <= 0)
			{
				if(WORKER == 0) cout<<"Program exits with an error: the specified SIGMASQ is "<<SIGMASQ<<", not positive.\n\n";
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Finalize();
				exit(EXIT_FAILURE);
			}
			if(BETA <= 0)
			{
				if(WORKER == 0) cout<<"Program exits with an error: the specified BETA is "<<BETA<<", not positive.\n\n";
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Finalize();
				exit(EXIT_FAILURE);
			}
			if(TAUSQ <= 0)
			{
				if(WORKER == 0) cout<<"Program exits with an error: the specified TAUSQ is "<<TAUSQ<<", not positive.\n\n";
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Finalize();
				exit(EXIT_FAILURE);
			}
		}
	}
}
//Function for reading user_parameters
void read_user_parameters()
{
	//Read the configuration file "user_parameters"
	read_from_file();

	//Check errors
	check_errors();

	if(PRINT_DETAIL_FLAG && WORKER == 0) print_parameters_summary();
}