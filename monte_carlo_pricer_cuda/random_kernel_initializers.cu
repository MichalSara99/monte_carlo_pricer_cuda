#include"random_kernel_initializers.cuh"


namespace random_kernel_initializers {


	__global__
		void initialiseRandomKernel1D(unsigned int seed, curandState_t* states,
			unsigned int nPaths) {
		// Thread Id (corresponds to path number)
		const unsigned int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
		// Initialise a RNG for every Path
		if (t_idx < nPaths)
			curand_init(seed, t_idx, 0, &states[t_idx]);
	}

	__global__
		void initialiseRandomKernel2D(unsigned int seed, curandState_t* states,
			unsigned int nPathsWidth, unsigned int nPathsHeight) {
		// Thread t_idx (corresponds to path number)
		const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
		const unsigned int t_idx = c_idx + nPathsWidth * r_idx;
		// Initialise a RNG for every Path
		const unsigned int nPaths = nPathsWidth * nPathsHeight;
		if (t_idx < nPaths)
			curand_init(seed, t_idx, 0, &states[t_idx]);
	}

	__global__
		void initialiseRandomKernel3D(unsigned int seed, curandState_t* states,
			unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth) {
		// Thread t_idx (corresponds to path number)
		const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
		const unsigned int l_idx = blockIdx.z * blockDim.z + threadIdx.z;
		const unsigned int t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight*l_idx;
		// Initialise a RNG for every Path
		const unsigned int nPaths = nPathsWidth * nPathsHeight * nPathsDepth;
		if (t_idx < nPaths)
			curand_init(seed, t_idx, 0, &states[t_idx]);
	}


}
