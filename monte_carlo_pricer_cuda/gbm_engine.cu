#include<cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>
#include<device_launch_parameters.h>
#include"sde_builder_cuda.h"
#include<cassert>
#include"random_kernel_initializers.cuh"
#include"mc_types.h"
#include"gbm_kernels.h"



namespace fdm_engine_cuda {

	using mc_types::FDMScheme;
	using mc_types::GPUConfiguration;
	using mc_types::PathValuesType;

	void fdm_engine_cuda::GBMPathEngineDouble::_generate1D(double *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, unsigned int nSteps, double dt) const {
		// initialise RNG states
		const unsigned int threadsPerBlock = THREADS_PER_BLOCK;
		unsigned int blocksPerGrid = (nPaths + threadsPerBlock - 1) / threadsPerBlock;
		random_kernel_initializers::initialiseRandomKernel1D << <threadsPerBlock, blocksPerGrid >> > (time(0), states, nPaths);
		switch (scheme) {
		case FDMScheme::EulerScheme:
		{
			gbm_kernels::euler_scheme::generatePathsKernel1D<double> << <threadsPerBlock, blocksPerGrid >> > (this->gbm_, d_paths, states,
				nPaths, nSteps, dt);

		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			gbm_kernels::milstein_scheme::generatePathsKernel1D<double><< <threadsPerBlock, blocksPerGrid >> > (this->gbm_, d_paths, states,
				nPaths, nSteps, dt);
		}
		break;
		}
	}

	void fdm_engine_cuda::GBMPathEngineDouble::_generate2D(double *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, unsigned int nSteps, double dt)const {
		const unsigned int widthSize{ 1000 };
		assert((nPaths%widthSize) == 0);
		unsigned int heightSize{ nPaths / widthSize };
		const unsigned int threadsPerBlockX = THREADS_2D_PER_BLOCK_X;
		const unsigned int threadsPerBlockY = THREADS_2D_PER_BLOCK_Y;
		unsigned int blocksPerGridX = (widthSize + threadsPerBlockX - 1) / threadsPerBlockX;
		unsigned int blocksPerGridY = (heightSize + threadsPerBlockY - 1) / threadsPerBlockY;
		const dim3 blockSize = dim3(threadsPerBlockX, threadsPerBlockY);
		const dim3 gridSize = dim3(blocksPerGridX, blocksPerGridY);
		random_kernel_initializers::initialiseRandomKernel2D << <gridSize, blockSize >> > (time(0), states, widthSize, heightSize);
		switch (scheme) {
		case FDMScheme::EulerScheme:
		{
			gbm_kernels::euler_scheme::generatePathsKernel2D<double><<<gridSize, blockSize >> > (this->gbm_, d_paths, states,
				widthSize, heightSize, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			gbm_kernels::milstein_scheme::generatePathsKernel2D<double><< <gridSize, blockSize >> > (this->gbm_, d_paths, states,
				widthSize, heightSize, nSteps, dt);
		}
		break;
		}
	}


	void fdm_engine_cuda::GBMPathEngineDouble::_generate3D(double *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, unsigned int nSteps, double dt)const {
		const unsigned int widthSize{ 100 };
		const unsigned int heightSize{ 100 };
		assert((nPaths % (widthSize*heightSize)) == 0);
		unsigned int depthSize{ nPaths / (widthSize*heightSize) };
		const unsigned int threadsPerBlockX = THREADS_3D_PER_BLOCK_X;
		const unsigned int threadsPerBlockY = THREADS_3D_PER_BLOCK_Y;
		const unsigned int threadsPerBlockZ = THREADS_3D_PER_BLOCK_Z;
		unsigned int blocksPerGridX = (widthSize + threadsPerBlockX - 1) / threadsPerBlockX;
		unsigned int blocksPerGridY = (heightSize + threadsPerBlockY - 1) / threadsPerBlockY;
		unsigned int blocksPerGridZ = (depthSize + threadsPerBlockZ - 1) / threadsPerBlockZ;
		const dim3 blockSize = dim3(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);
		const dim3 gridSize = dim3(blocksPerGridX, blocksPerGridY, blocksPerGridZ);
		random_kernel_initializers::initialiseRandomKernel3D << <gridSize, blockSize >> > (time(0), states, widthSize, heightSize, depthSize);

		switch (scheme) {
		case FDMScheme::EulerScheme:
		{
			gbm_kernels::euler_scheme::generatePathsKernel3D<double> << <gridSize, blockSize >> > (this->gbm_, d_paths, states,
				widthSize, heightSize, depthSize, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			gbm_kernels::milstein_scheme::generatePathsKernel3D<double> << <gridSize, blockSize >> > (this->gbm_, d_paths, states,
				widthSize, heightSize, depthSize, nSteps, dt);
		}
		break;
		}
	}


	fdm_engine_cuda::PathValuesType<fdm_engine_cuda::PathValuesType<double>>
		fdm_engine_cuda::GBMPathEngineDouble::simulate(unsigned int nPaths, unsigned int nSteps, double dt,
			FDMScheme scheme, GPUConfiguration config)const {

		double *d_paths = NULL;
		curandState_t *states; // RNG state for each thread
							   // Allocate memory for the paths
		cudaMalloc(&d_paths, nPaths * nSteps * sizeof(double));
		// Allocate memory for RNG states
		cudaMalloc(&states, nPaths * sizeof(curandState_t));

		switch (config) {
		case GPUConfiguration::Grid1D:
		{
			_generate1D(d_paths, states, scheme, nPaths, nSteps, dt);
		}
		break;
		case GPUConfiguration::Grid2D:
		{
			_generate2D(d_paths, states, scheme, nPaths, nSteps, dt);
		}
		break;
		case GPUConfiguration::Grid3D:
		{
			_generate3D(d_paths, states, scheme, nPaths, nSteps, dt);
		}
		break;
		default:
		{
			_generate1D(d_paths, states, scheme, nPaths, nSteps, dt);
		}
		break;
		}

		// Allocate memory on the host:
		double *h_paths = (double *)malloc(nPaths*nSteps * sizeof(double));
		// Copy from device to host:
		cudaMemcpy(h_paths, d_paths, nPaths*nSteps * sizeof(double),
			cudaMemcpyKind::cudaMemcpyDeviceToHost);


		std::vector<std::vector<double>> paths(nPaths);
		for (std::size_t s = 0; s < paths.size(); ++s) {
			std::vector<double> path(nSteps);
			for (std::size_t p = 0; p < path.size(); ++p) {
				path[p] = std::move(h_paths[s + paths.size()*p]);
			}
			paths[s] = std::move(path);
		}
		free(h_paths);
		cudaFree(d_paths);
		cudaFree(states);
		return paths;
	}



	void fdm_engine_cuda::GBMPathEngineFloat::_generate1D(float *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, unsigned int nSteps, float dt) const {
		// initialise RNG states
		const unsigned int threadsPerBlock = THREADS_PER_BLOCK;
		unsigned int blocksPerGrid = (nPaths + threadsPerBlock - 1) / threadsPerBlock;
		random_kernel_initializers::initialiseRandomKernel1D << <threadsPerBlock, blocksPerGrid >> > (time(0), states, nPaths);
		switch (scheme) {
		case FDMScheme::EulerScheme:
		{
			gbm_kernels::euler_scheme::generatePathsKernel1D<float> << <threadsPerBlock, blocksPerGrid >> > (this->gbm_, d_paths, states,
				nPaths, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			gbm_kernels::milstein_scheme::generatePathsKernel1D<float><< <threadsPerBlock, blocksPerGrid >> > (this->gbm_, d_paths, states,
				nPaths, nSteps, dt);
		}
		break;
		}
	}

	void fdm_engine_cuda::GBMPathEngineFloat::_generate2D(float *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, unsigned int nSteps, float dt)const {
		const unsigned int widthSize{ 1000 };
		assert((nPaths%widthSize) == 0);
		unsigned int heightSize{ nPaths / widthSize };
		const unsigned int threadsPerBlockX = THREADS_2D_PER_BLOCK_X;
		const unsigned int threadsPerBlockY = THREADS_2D_PER_BLOCK_Y;
		unsigned int blocksPerGridX = (widthSize + threadsPerBlockX - 1) / threadsPerBlockX;
		unsigned int blocksPerGridY = (heightSize + threadsPerBlockY - 1) / threadsPerBlockY;
		const dim3 blockSize = dim3(threadsPerBlockX, threadsPerBlockY);
		const dim3 gridSize = dim3(blocksPerGridX, blocksPerGridY);
		random_kernel_initializers::initialiseRandomKernel2D << <gridSize, blockSize >> > (time(0), states, widthSize, heightSize);

		switch (scheme) {
		case FDMScheme::EulerScheme:
		{
			gbm_kernels::euler_scheme::generatePathsKernel2D<float> << <gridSize, blockSize >> > (this->gbm_, d_paths, states,
				widthSize, heightSize, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			gbm_kernels::milstein_scheme::generatePathsKernel2D<float> << <gridSize, blockSize >> > (this->gbm_, d_paths, states,
				widthSize, heightSize, nSteps, dt);
		}
		break;
		}
	}

	void fdm_engine_cuda::GBMPathEngineFloat::_generate3D(float *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, unsigned int nSteps, float dt)const {
		const unsigned int widthSize{ 100 };
		const unsigned int heightSize{ 100 };
		assert((nPaths % (widthSize*heightSize)) == 0);
		unsigned int depthSize{ nPaths / (widthSize*heightSize) };
		const unsigned int threadsPerBlockX = THREADS_3D_PER_BLOCK_X;
		const unsigned int threadsPerBlockY = THREADS_3D_PER_BLOCK_Y;
		const unsigned int threadsPerBlockZ = THREADS_3D_PER_BLOCK_Z;
		unsigned int blocksPerGridX = (widthSize + threadsPerBlockX - 1) / threadsPerBlockX;
		unsigned int blocksPerGridY = (heightSize + threadsPerBlockY - 1) / threadsPerBlockY;
		unsigned int blocksPerGridZ = (depthSize + threadsPerBlockZ - 1) / threadsPerBlockZ;
		const dim3 blockSize = dim3(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);
		const dim3 gridSize = dim3(blocksPerGridX, blocksPerGridY, blocksPerGridZ);
		random_kernel_initializers::initialiseRandomKernel3D << <gridSize, blockSize >> > (time(0), states, widthSize, heightSize, depthSize);
		switch (scheme) {
		case FDMScheme::EulerScheme:
		{
			gbm_kernels::euler_scheme::generatePathsKernel3D<float> << <gridSize, blockSize >> > (this->gbm_, d_paths, states,
				widthSize, heightSize, depthSize, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			gbm_kernels::milstein_scheme::generatePathsKernel3D<float> << <gridSize, blockSize >> > (this->gbm_, d_paths, states,
				widthSize, heightSize, depthSize, nSteps, dt);
		}
		break;
		}
	}


	fdm_engine_cuda::PathValuesType<fdm_engine_cuda::PathValuesType<float>>
		fdm_engine_cuda::GBMPathEngineFloat::simulate(unsigned int nPaths, unsigned int nSteps, float dt,
			FDMScheme scheme, GPUConfiguration config)const {

		float *d_paths = NULL;
		curandState_t *states; // RNG state for each thread
							   // Allocate memory for the paths
		cudaMalloc(&d_paths, nPaths * nSteps * sizeof(float));
		// Allocate memory for RNG states
		cudaMalloc(&states, nPaths * sizeof(curandState_t));


		switch (config) {
		case GPUConfiguration::Grid1D:
		{
			_generate1D(d_paths, states, scheme, nPaths, nSteps, dt);
		}
		break;
		case GPUConfiguration::Grid2D:
		{
			_generate2D(d_paths, states, scheme, nPaths, nSteps, dt);
		}
		break;
		case GPUConfiguration::Grid3D:
		{
			_generate3D(d_paths, states, scheme, nPaths, nSteps, dt);
		}
		break;
		default:
		{
			_generate1D(d_paths, states, scheme, nPaths, nSteps, dt);
		}
		break;
		}

		// Allocate memory on the host:
		float *h_paths = (float *)malloc(nPaths*nSteps * sizeof(float));
		// Copy from device to host:
		cudaMemcpy(h_paths, d_paths, nPaths*nSteps * sizeof(float),
			cudaMemcpyKind::cudaMemcpyDeviceToHost);

		std::vector<std::vector<float>> paths(nPaths);
		for (std::size_t s = 0; s < paths.size(); ++s) {
			std::vector<float> path(nSteps);
			for (std::size_t p = 0; p < path.size(); ++p) {
				path[p] = std::move(h_paths[s + paths.size()*p]);
			}
			paths[s] = std::move(path);
		}


		free(h_paths);
		cudaFree(d_paths);
		cudaFree(states);

		return paths;
	}

}