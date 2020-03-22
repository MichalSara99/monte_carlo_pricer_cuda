#pragma once
#if !defined(_RANDON_KERNEL_INITIALIZERS)
#define _RANDON_KERNEL_INITIALIZERS


#include<cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>
#include<device_launch_parameters.h>


namespace random_kernel_initializers {

	__global__
		void initialiseRandomKernel1D(unsigned int seed, curandState_t* states,
			unsigned int nPaths);

	__global__
		void initialiseRandomKernel2D(unsigned int seed, curandState_t* states,
			unsigned int nPathsWidth, unsigned int nPathsHeight);

	__global__
		void initialiseRandomKernel3D(unsigned int seed, curandState_t* states,
			unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth);

}







#endif ///_RANDON_KERNEL_INITIALIZERS