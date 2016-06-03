// SM 3.0 or greater GPUS only!
// compile with:  nvcc bab_stream.cu -o stream -arch=sm_30 -std=c++11 --expt-relaxed-constexpr
#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <curand_kernel.h>
#include <vector>
#include <time.h>
#include <unistd.h> // sleep
#include <thread>
#include "../bab_gui.cpp"
#define BITS_PER_INT 32
#define NaN std::numeric_limits<double>::quiet_NaN()
unsigned long NUM_DIMS = 0; // Global variable holding the number of dimensions per interval

//#define DEBUGG 1

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
    bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
        exit(code);
    }
}

inline uint interval_purger(float in)
{
  static int remove_counter = 0;
  if(remove_counter > 0) {
    remove_counter--;
    return 1; //remove me
  }
  else if(std::isnan(in)){
    remove_counter = 2 * NUM_DIMS - 1;
    return 1; //remove me
  }
  else
    return 0; // keep me
}

void split(std::vector<float> &v, unsigned long interval_offset, unsigned long NUM_DIMS) 
{
	float interval[NUM_DIMS*2];
	for(unsigned long i = 0; i < NUM_DIMS * 2; ++i) {
		interval[i] = v[interval_offset + i];
	}
	// Get the widest dim
	float max_width = (v[1] - v[0]);
	uint index_of_max_width = 0;
	for(int j = 1; j < NUM_DIMS; ++j) {
		float this_width = (v[1 + j * 2] - v[0 + j * 2]);
		if(this_width > max_width) {
			max_width = this_width;
			index_of_max_width = j;
		}
	}
	//printf("Max width: %f\n", max_width);
	//assert(max_width > 0);

	// Split into 2 new intervals
	for(int i = 0; i < NUM_DIMS; i++) {
		if(i == index_of_max_width) {
			v.push_back(interval[2 * i]);
			v.push_back(interval[2 * i + 1] - max_width / 2.0);
		}
		else {
			v.push_back(interval[2 * i]);
			v.push_back(interval[2 * i + 1]);
		}
	}

	for(int i = 0; i < NUM_DIMS; i++) {
		if(i == index_of_max_width) {
			v.push_back(interval[2 * i] + max_width / 2.0);
			v.push_back(interval[2 * i + 1] );
		}
		else {
			v.push_back(interval[2 * i]);
			v.push_back(interval[2 * i + 1]);
		}
	}

}
// void update_candidates(std::vector<float> &v, unsigned long begin_offset, unsigned long size_to_span) 
// {
// 	long unsigned original_size = v.size();
// 	printf("begin offset: %lu, size_to_span: %lu\n", begin_offset, size_to_span);
// 	std::vector<float>::iterator end_vaild = 
// 	std::remove_if(v.begin(), v.end(), interval_purger);
// 	printf("elems marked for removal!\n");
// 	v.erase(end_vaild, v.end());
// 	long unsigned elems_removed = original_size - v.size();
// 	printf("%lu elems purged!\n", elems_removed);
//   	unsigned long pre_expand_size = v.size(); // OPTIMIZE TO UPDATE ONE CHUNK
//   	printf("Cabdudates is now of size: %lu\n", pre_expand_size);
//   	unsigned long stride = NUM_DIMS * 2;
// 	for(unsigned long i = 0; i < pre_expand_size; i *= stride) {
//   		split(v, i, NUM_DIMS);
//   		v[i] = NaN;
//   	}
//   	end_vaild = std::remove_if(v.begin(), v.end(), interval_purger);
//   	v.erase(end_vaild, v.end());

// }
void update_candidates(std::vector<float> &v, std::vector<float> &s, unsigned long NUM_DIMS, double EPSILON) 
{
	// Grab solutions by checking intervals that weren't deleted.
	// GPU PARALLELIZABLE
	long unsigned original_size = v.size();
	for(long unsigned i = 0; i < original_size; i += NUM_DIMS * 2) {
		if(std::isnan(v[i])) {
			continue;
		}
		double volume = v[i + 1] - v[i];
		for(int j = 1; j < NUM_DIMS; ++j) {
			float low = v[i + 2 * j];
			float high = v[i + 2 * j + 1];
			float width = high - low;
			volume *= width;
		}

		if(volume <= EPSILON ) {
			for(unsigned long j = 0; j < 2 * NUM_DIMS; ++j) {
				s.push_back(v[i + j]);
			}
			v[i] = NaN;
		}
	}
	// Clean candidates
	std::vector<float>::iterator end_vaild = std::remove_if(v.begin(), v.end(), interval_purger);
	//printf("elems marked for removal!\n");
	v.erase(end_vaild, v.end());
	original_size = v.size();
	end_vaild = std::remove_if(v.begin(), v.end(), interval_purger);
	//printf("elems marked for removal!\n");
	v.erase(end_vaild, v.end());
	long unsigned elems_removed = original_size - v.size();
	//printf("%lu elems purged!\n", elems_removed);
  	unsigned long pre_expand_size = v.size(); // OPTIMIZE TO UPDATE ONE CHUNK
  	//printf("Candidates is now of size: %lu\n", pre_expand_size);
  	unsigned long stride = NUM_DIMS * 2;
  	//printf("Expanding...\n");
	for(unsigned long i = 0; i < pre_expand_size; i += stride) {
  		split(v, i, NUM_DIMS);
  	}
  	//printf("Puring parents intervals\n");
 	for(unsigned long i = 0; i < pre_expand_size; i += stride) {
  		v[i] = NaN;
  	}
  	end_vaild = std::remove_if(v.begin(), v.end(), interval_purger);
  	v.erase(end_vaild, v.end());

}

__device__ inline float squared(float v) { return v * v; }

__device__ inline uint determine_valid_interval_line(float * start, unsigned long NUM_DIMS)
{
	float xmin = start[0];
	float xmax = start[1];
	float ymin = start[2];
	float ymax = start[3];

	//if(xmin < ymax && xmax > ymin)
	int within_line = (ymin <= xmax && ymax >= xmin);
	// for(unsigned long i = 0; i < NUM_DIMS * 2; ++i) {
	// 	garbage *= start[i];
	// }
    // return 1;
    return within_line;

}

__device__ inline uint determine_valid_interval_sphere(float * start, unsigned long NUM_DIMS)
{
	float R = 1.0;
	float C1X = start[0];
	float C1Y = start[2];
	float C1Z = start[4];
	float C2X = start[1];
	float C2Y = start[3];
	float C2Z = start[5];
	float SX = 0.0;
	float SY = 0.0;
	float SZ = 0.0;
	float xmin = C1X;
	float xmax = C2X;
	float ymin = C1Y;
	float ymax = C2Y;

    float dist_squared = R * R;
    /* assume C1 and C2 are element-wise sorted, if not, do that now */
    if (SX < C1X) dist_squared -= squared(SX - C1X);
    else if (SX > C2X) dist_squared -= squared(SX - C2X);
    if (SY < C1Y) dist_squared -= squared(SY - C1Y);
    else if (SY > C2Y) dist_squared -= squared(SY - C2Y);
    if (SZ < C1Z) dist_squared -= squared(SZ - C1Z);
    else if (SZ > C2Z) dist_squared -= squared(SZ - C2Z);
    return dist_squared > 0;

}

__device__ inline uint determine_valid_interval_spheres(float * start, unsigned long NUM_DIMS)
{
	float R = 1.0;
	float C1X = start[0];
	float C1Y = start[2];
	float C1Z = start[4];
	float C2X = start[1];
	float C2Y = start[3];
	float C2Z = start[5];
	float SX = 0.0;
	float SY = 0.0;
	float SZ = 0.0;
	float xmin = C1X;
	float xmax = C2X;
	float ymin = C1Y;
	float ymax = C2Y;

    float dist_squared = R * R;
    /* assume C1 and C2 are element-wise sorted, if not, do that now */
    if (SX < C1X) dist_squared -= squared(SX - C1X);
    else if (SX > C2X) dist_squared -= squared(SX - C2X);
    if (SY < C1Y) dist_squared -= squared(SY - C1Y);
    else if (SY > C2Y) dist_squared -= squared(SY - C2Y);
    if (SZ < C1Z) dist_squared -= squared(SZ - C1Z);
    else if (SZ > C2Z) dist_squared -= squared(SZ - C2Z);
    bool ans1 = dist_squared > 0;

//////////////////sphere 2
	R = 1.5;
	SX = 2.0;
	SY = 7.0;
	SZ = 2.0;

    dist_squared = R * R;
    /* assume C1 and C2 are element-wise sorted, if not, do that now */
    if (SX < C1X) dist_squared -= squared(SX - C1X);
    else if (SX > C2X) dist_squared -= squared(SX - C2X);
    if (SY < C1Y) dist_squared -= squared(SY - C1Y);
    else if (SY > C2Y) dist_squared -= squared(SY - C2Y);
    if (SZ < C1Z) dist_squared -= squared(SZ - C1Z);
    else if (SZ > C2Z) dist_squared -= squared(SZ - C2Z);
    bool ans2 = dist_squared > 0;

//////////////////sphere 3
	R = 3.0;
	SX = -4.0;
	SY = -4.0;
	SZ = -4.0;

    dist_squared = R * R;
    /* assume C1 and C2 are element-wise sorted, if not, do that now */
    if (SX < C1X) dist_squared -= squared(SX - C1X);
    else if (SX > C2X) dist_squared -= squared(SX - C2X);
    if (SY < C1Y) dist_squared -= squared(SY - C1Y);
    else if (SY > C2Y) dist_squared -= squared(SY - C2Y);
    if (SZ < C1Z) dist_squared -= squared(SZ - C1Z);
    else if (SZ > C2Z) dist_squared -= squared(SZ - C2Z);
    bool ans3 = dist_squared > 0;

//////////////////sphere 4
	R = 0.7;
	SX = 5.0;
	SY = -1.0;
	SZ = 7.0;


    dist_squared = R * R;
    /* assume C1 and C2 are element-wise sorted, if not, do that now */
    if (SX < C1X) dist_squared -= squared(SX - C1X);
    else if (SX > C2X) dist_squared -= squared(SX - C2X);
    if (SY < C1Y) dist_squared -= squared(SY - C1Y);
    else if (SY > C2Y) dist_squared -= squared(SY - C2Y);
    if (SZ < C1Z) dist_squared -= squared(SZ - C1Z);
    else if (SZ > C2Z) dist_squared -= squared(SZ - C2Z);
    bool ans4 = dist_squared > 0;

    return ans1 || ans2 || ans3 || ans4 || determine_valid_interval_line(start, NUM_DIMS);

}


// SM 3.0 > devices only
__global__
void branch_and_bound(float * intervals, unsigned long num_floats, unsigned long NUM_DIMS)
{
	unsigned long thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	thread_index *= NUM_DIMS * 2;
	unsigned long jump_length = blockDim.x * gridDim.x * NUM_DIMS * 2;
	while(thread_index < num_floats) {
		float new_val = intervals[thread_index];
		float * start_addr = intervals + thread_index;
		uint result = determine_valid_interval_spheres(start_addr, NUM_DIMS);
		new_val = (result == 1 ? new_val : NaN);
		// printf("New val: %f\n", new_val);
		intervals[thread_index] = new_val;
		thread_index += jump_length;
	}
	// if(thread_index % jump_length == 0) {
	// 	for(int z = 0; z < num_floats; ++z)
	// 	 	printf("GPU: intervals[%d]: %f\n", z, intervals[z]);
	// }

}

// unsigned long get_num_intervals_left(std::vector<float> &a, unsigned long FLOATS_PER_INTERVAL, int _num_devices) {
// 	unsigned long num_devices = (unsigned long) _num_devices;
// 	return (unsigned long) a.size() / (FLOATS_PER_INTERVAL * num_devices);
// }

void divy_up_work(std::vector<float> &candidate_intervals, unsigned long FLOATS_PER_INTERVAL, 
	int num_devices, std::vector<unsigned long> & interval_sizes, 
	std::vector<unsigned long> & float_offsets)
{
	unsigned long total_floats = candidate_intervals.size();
	unsigned long total_intervals = total_floats / FLOATS_PER_INTERVAL;
	unsigned long intervals_distributed = 0;
	unsigned long stride = total_intervals / num_devices;
	if(stride * num_devices != total_intervals) // fraction
		stride += 1;
	// Split as evenly as possible
	for(int i = 0; i < num_devices; ++i) {
			unsigned long this_length = std::min(total_intervals - intervals_distributed, stride);
			unsigned long this_offset = intervals_distributed;
			intervals_distributed += this_length;
			// convert to raw size in bytes
			float_offsets.push_back(this_offset * FLOATS_PER_INTERVAL * sizeof(float));
			// Keep size in units of INTERVAL_SIZE
			interval_sizes.push_back(this_length);
	}
	// for(int i = 0; i < num_devices; ++i) {
	// 	printf("offset and size and stride and total_intervals %d: %lu, %lu, %lu, %lu\n", i, float_offsets[i], interval_sizes[i], stride, total_intervals);
	// }
}

void visualize_realtime(Level_Set_GUI LS_GUI) {
	// RUN GUI LOOP
	LS_GUI.mainLoop();
	return;
}

int main(int argc, char **argv) {
	if(argc != 3) {
		printf("./bab <number_of_dims> <min epsilon>\n");
		return -1;
	}

	assert( sizeof(size_t) == 8 ); 
	assert( sizeof(unsigned long) == 8 );
	NUM_DIMS = (unsigned long) atoi(argv[1]);
	float EPSILON = (float) atof(argv[2]);

	assert(NUM_DIMS > 0 && NUM_DIMS < 7);
	printf("Num dims: %lu\n", NUM_DIMS);
	time_t initial, final;
	int num_devices;
    cudaDeviceProp prop;
  	gpuErrchk( cudaGetDeviceProperties(&prop, 0) ); // Assume all GPUs on this system are the same
	gpuErrchk( cudaGetDeviceCount(&num_devices) );
	int NUM_SMS = prop.multiProcessorCount;
	printf("Num devices: %d. SMs per device: %d\n", num_devices, NUM_SMS);
	// Compute optimal launch bounds and other constants
	uint MAX_BLOCK_SIZE = 1024;
	uint BLOCKS_PER_SM = 2;
	uint NUM_BLOCKS = BLOCKS_PER_SM * NUM_SMS;
	unsigned long INTERVAL_SIZE = (2 * sizeof(float) * NUM_DIMS);
	unsigned long FLOATS_PER_INTERVAL = 2 * NUM_DIMS;
	
	// GUI INIT

	Level_Set_GUI &LS_GUI = Level_Set_GUI::getInstance();
	LS_GUI.dimensions = NUM_DIMS;
	LS_GUI.setup(argc, argv);
	//std::thread first(visualize_realtime, LS_GUI); // create thread called "first"

	// END GUI


	// Host-Side allocations
	// Make up a search space
	float ** search_space = new float * [NUM_DIMS];
	for(int i = 0; i < NUM_DIMS; ++i) {
		search_space[i] = new float[2]; //min, max
	}
	// Populate search space
	for(int i = 0; i < NUM_DIMS; i++) {
		search_space[i][0] = -10.0; // min
		search_space[i][1] = 10.0; // max
	}
	// Create vector of handles to device bool_arrays
	std::vector<float *> dev_candidate_intervals(num_devices);
	// Divy up the workspace. level 1: num_devices. Level 2: num_dims. Level 3: lower, upper.
	float *** initial_search_spaces = new float ** [num_devices];
	for(int i = 0; i < num_devices; ++i) {
		initial_search_spaces[i] = new float * [NUM_DIMS];
	}
	for(int i = 0; i < num_devices; ++i) {
		for(int j = 0; j < NUM_DIMS; ++j) {
			initial_search_spaces[i][j] = new float [2]; //lower bound, upper bound
		}
	}

	// First N - 1 dimensions are the same as the initial searchspace dims
	for(int i = 0; i < num_devices; ++i) {
		for(int j = 0; j < NUM_DIMS - 1; ++j) {
			initial_search_spaces[i][j][0] = search_space[j][0];
			initial_search_spaces[i][j][1] = search_space[j][1];
		}
		// Split the last dimension evenly
		float last_dim_min = search_space[NUM_DIMS - 1][0];
		float last_dim_max = search_space[NUM_DIMS - 1][1];
		float stride = (last_dim_max - last_dim_min) / num_devices;
		float this_lower_bound = i * stride + last_dim_min;
		initial_search_spaces[i][NUM_DIMS - 1][0] = this_lower_bound;
		initial_search_spaces[i][NUM_DIMS - 1][1] = std::min(this_lower_bound + stride, last_dim_max); 
	}


	// Store the number of intervals being tested on the GPUs
	std::vector<unsigned long> num_intervals(num_devices);
	// Store the sizesof the arrays that hold candidate intervals
	std::vector<unsigned long> array_capacities(num_devices);
	std::vector<unsigned long> array_sizes(num_devices);
	// Store the candidate intervals and solution intervals 
	std::vector<float> candidate_intervals;
	std::vector<float> satisfactory_intervals;
	long unsigned iterations = 0;
	// Populate the stack of candidates.
	for(int i = 0; i < num_devices; ++i) {
		for(int j = 0; j < NUM_DIMS; ++j) {
			candidate_intervals.push_back(initial_search_spaces[i][j][0]); // lower bound
			candidate_intervals.push_back(initial_search_spaces[i][j][1]); // upper bound
		}
	}

	while(candidate_intervals.size() != 0) 
	{
		initial = clock();
		std::vector<unsigned long> interval_sizes;
		std::vector<unsigned long> interval_offsets_bytes;
		divy_up_work(candidate_intervals, FLOATS_PER_INTERVAL, num_devices, interval_sizes, interval_offsets_bytes);
		// Launch kernels on each GPU
		iterations++;
		for(int i = 0; i < num_devices; ++i) {
			// Select GPU
			cudaSetDevice(i);
			// Read in available memory on this GPU
			size_t free_memory, total_memory;
			gpuErrchk( cudaMemGetInfo(&free_memory, &total_memory) );
			// Determine how big we can make the array of intervals
			unsigned long intervals_available = interval_sizes[i];
			unsigned long max_array_capacity = .99*(free_memory) / INTERVAL_SIZE;
			array_capacities[i] = std::min(max_array_capacity, intervals_available);
			//printf("cap: %lu, avail: %lu\n", max_array_capacity, intervals_available);
			array_sizes[i] = array_capacities[i] * INTERVAL_SIZE;
			// Malloc space for this array
			gpuErrchk( cudaMalloc((void **) &dev_candidate_intervals[i], array_sizes[i]) );
			// Copy over intervals
			float * intervals_start_addr = &candidate_intervals[0] + interval_offsets_bytes[i] / sizeof(float);
			//printf("array size: %lu, interval_offsets_bytes: %lu\n",array_sizes[i], interval_offsets_bytes[i]);
			//for(int z = 0; z < array_sizes[i] / sizeof(float); ++z)
		 		//printf("MEMCPY: candidate_intervals[%d]: %f\n", z, *(intervals_start_addr + z));
			gpuErrchk( cudaMemcpyAsync(dev_candidate_intervals[i], intervals_start_addr, array_sizes[i], cudaMemcpyHostToDevice) );

			branch_and_bound<<<NUM_BLOCKS, MAX_BLOCK_SIZE>>> (dev_candidate_intervals[i], array_sizes[i] / sizeof(float), NUM_DIMS);
	        // Check for errors on kernel call
	        cudaError err = cudaGetLastError();
	        if(cudaSuccess != err)
	            printf("Error %s\n",cudaGetErrorString(err));
	     	// Read back the procesed intervals ontop of their old data
		 	gpuErrchk( cudaMemcpyAsync(&candidate_intervals[0] + interval_offsets_bytes[i] / sizeof(float),
		 	 	dev_candidate_intervals[i], 
		 		array_sizes[i],
		 		cudaMemcpyDeviceToHost) );
		 	gpuErrchk( cudaFree(dev_candidate_intervals[i]) );
		 	// for(int z = 0; z < candidate_intervals.size(); ++z)
		 	// 	printf("CPU: dev_candidate_intervals[%d]: %f\n", z, candidate_intervals[z]);
		}
		update_candidates(candidate_intervals, satisfactory_intervals, NUM_DIMS, EPSILON);
		// UPDATE INTERVALS ON GUI

		LS_GUI.update_candidates(candidate_intervals);
		LS_GUI.update_solutions(satisfactory_intervals);
        LS_GUI.display();
    	//sleep(1);
		final = clock();
// if (iterations < 25)
// {
 
// 		for (unsigned long tempnumer = 0; tempnumer < candidate_intervals.size(); tempnumer++)
// 		{
// 				printf(" %f ", candidate_intervals[tempnumer]);
// 		}
//  printf("\n");
// }		// END GUI

		printf("Iteration %lu time: %f (s). Num candidates: %lu, Num solutions: %lu\n",
			iterations,
			double(final - initial) / CLOCKS_PER_SEC,
			candidate_intervals.size() / (2 * NUM_DIMS),
			satisfactory_intervals.size() / (2 * NUM_DIMS)
		);
		// for(int i = 0; i < candidate_intervals.size(); ++i)
		// 	printf("candidate_intervals[%d]: %f\n", i, candidate_intervals[i]);

	}
	// cleanup host	
	// Clean up initial search spaces
	for(int i = 0; i < num_devices; ++i) {
		for(int j = 0; j < NUM_DIMS; ++j) {
			delete [] initial_search_spaces[i][j];
		}
	}
	// // Clean up streams
	// for(int i = 0; i < num_devices; ++i) {
	//     gpuErrchk( cudaStreamDestroy(streams[i]) );
	// }

	for(int i = 0; i < num_devices; ++i) {
		delete [] initial_search_spaces[i];
	}
	delete initial_search_spaces;

	// Clean up search space
	for(int i = 0; i < NUM_DIMS; ++i) {
		delete [] search_space[i];
	}
	delete search_space;
	// Close visualization
//	first.join();
        

        // GUI MAIN LOOP
        LS_GUI.mainLoop(); 
}


void widest_dims(float *** initial_search_spaces, int num_devices, int NUM_DIMS) {
	// Vector to hold the index of the widest dim for each GPU's search space
	std::vector<uint> indices_of_widest_dim(num_devices);
	for(int i = 0; i < num_devices; ++i) {
		float max_width = (initial_search_spaces[i][0][1] - initial_search_spaces[i][0][0]);
		uint index_of_max_width = 0;
		for(int j = 1; j < NUM_DIMS; ++j) {
			float this_width = (initial_search_spaces[i][j][1] - initial_search_spaces[i][j][0]);
			if(this_width > max_width) {
				max_width = this_width;
				index_of_max_width = j;
			}
		}
		assert(max_width > 0);
		indices_of_widest_dim[i] = index_of_max_width;
	}
}
