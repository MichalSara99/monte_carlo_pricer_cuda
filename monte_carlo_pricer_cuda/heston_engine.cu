#include<cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>
#include<device_launch_parameters.h>
#include"sde_builder_cuda.h"
#include<cassert>
#include"random_kernel_initializers.cuh"
#include"mc_types.h"
#include"heston_kernels.h"

namespace fdm_engine_cuda {

	using mc_types::FDMScheme;
	using mc_types::GPUConfiguration;
	using mc_types::PathValuesType;

	//=====================================================================
	//====== equidistant overloads
	//=====================================================================

	void fdm_engine_cuda::HestonPathEngineDouble::_generate1D(double *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, unsigned int nSteps, double dt) const {
		// initialise RNG states
		const unsigned int threadsPerBlock = THREADS_PER_BLOCK;
		unsigned int blocksPerGrid = (nPaths + threadsPerBlock - 1) / threadsPerBlock;
		random_kernel_initializers::initialiseRandomKernel1D << <threadsPerBlock, blocksPerGrid >> >(time(0), states, nPaths);
		switch (scheme) {
		case FDMScheme::EulerScheme:
		{
			heston_kernels::euler_scheme::generatePathsKernel1D<double> << <threadsPerBlock, blocksPerGrid >> >(this->heston_, d_paths, states,
				nPaths, nSteps, dt);

		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			heston_kernels::milstein_scheme::generatePathsKernel1D<double> << <threadsPerBlock, blocksPerGrid >> >(this->heston_, d_paths, states,
				nPaths, nSteps, dt);
		}
		break;
		}
	}

	void fdm_engine_cuda::HestonPathEngineDouble::_generate2D(double *d_paths, curandState_t *states, FDMScheme scheme,
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
			heston_kernels::euler_scheme::generatePathsKernel2D<double> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			heston_kernels::milstein_scheme::generatePathsKernel2D<double> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, nSteps, dt);
		}
		break;
		}
	}


	void fdm_engine_cuda::HestonPathEngineDouble::_generate3D(double *d_paths, curandState_t *states, FDMScheme scheme,
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
			heston_kernels::euler_scheme::generatePathsKernel3D<double> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, depthSize, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			heston_kernels::milstein_scheme::generatePathsKernel3D<double> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, depthSize, nSteps, dt);
		}
		break;
		}
	}

	//=====================================================================
	//====== non-equidistant overloads
	//=====================================================================

	void fdm_engine_cuda::HestonPathEngineDouble::_generate1D(double *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, double const *d_times, unsigned int size) const {
		// initialise RNG states
		const unsigned int threadsPerBlock = THREADS_PER_BLOCK;
		unsigned int blocksPerGrid = (nPaths + threadsPerBlock - 1) / threadsPerBlock;
		random_kernel_initializers::initialiseRandomKernel1D << <threadsPerBlock, blocksPerGrid >> >(time(0), states, nPaths);
		switch (scheme) {
		case FDMScheme::EulerScheme:
		{
			heston_kernels::euler_scheme::generatePathsKernel1D<double> << <threadsPerBlock, blocksPerGrid >> >(this->heston_, d_paths, states,
				nPaths, d_times, size);

		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			heston_kernels::milstein_scheme::generatePathsKernel1D<double> << <threadsPerBlock, blocksPerGrid >> >(this->heston_, d_paths, states,
				nPaths, d_times, size);
		}
		break;
		}
	}

	void fdm_engine_cuda::HestonPathEngineDouble::_generate2D(double *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, double const *d_times, unsigned int size)const {
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
			heston_kernels::euler_scheme::generatePathsKernel2D<double> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, d_times, size);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			heston_kernels::milstein_scheme::generatePathsKernel2D<double> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, d_times, size);
		}
		break;
		}
	}


	void fdm_engine_cuda::HestonPathEngineDouble::_generate3D(double *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, double const *d_times, unsigned int size)const {
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
			heston_kernels::euler_scheme::generatePathsKernel3D<double> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, depthSize, d_times, size);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			heston_kernels::milstein_scheme::generatePathsKernel3D<double> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, depthSize, d_times, size);
		}
		break;
		}
	}

	//=====================================================================
	//====== equidistant overloads
	//=====================================================================

	fdm_engine_cuda::PathValuesType<fdm_engine_cuda::PathValuesType<double>>
		fdm_engine_cuda::HestonPathEngineDouble::simulate(unsigned int nPaths, unsigned int nSteps, double dt,
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


	//=====================================================================
	//====== non-equidistant overloads
	//=====================================================================

	fdm_engine_cuda::PathValuesType<fdm_engine_cuda::PathValuesType<double>>
		fdm_engine_cuda::HestonPathEngineDouble::simulate(unsigned int nPaths, TimePointsType<double> const &timePoints,
			FDMScheme scheme, GPUConfiguration config)const {
		unsigned int size = timePoints.size();
		// Allocate memory on the host:
		double *h_times = (double*)malloc(size * sizeof(double));
		// Copy:
		for (std::size_t t = 0; t < size; ++t) {
			h_times[t] = timePoints.at(t);
		}

		double *d_paths = NULL;
		double *d_times = NULL;
		// RNG state for each thread
		curandState_t *states;
		// Allocate memory for the paths
		cudaMalloc(&d_paths, nPaths * size * sizeof(double));
		// Allocate memory for the times
		cudaMalloc(&d_times, size * sizeof(double));
		// Allocate memory for RNG states
		cudaMalloc(&states, nPaths * sizeof(curandState_t));
		// Copy h_times to d_times,i.e. from hosdt to device:
		cudaMemcpy(d_times, h_times, size * sizeof(double), cudaMemcpyKind::cudaMemcpyHostToDevice);

		switch (config) {
		case GPUConfiguration::Grid1D:
		{
			_generate1D(d_paths, states, scheme, nPaths, d_times, size);
		}
		break;
		case GPUConfiguration::Grid2D:
		{
			_generate2D(d_paths, states, scheme, nPaths, d_times, size);
		}
		break;
		case GPUConfiguration::Grid3D:
		{
			_generate3D(d_paths, states, scheme, nPaths, d_times, size);
		}
		break;
		default:
		{
			_generate1D(d_paths, states, scheme, nPaths, d_times, size);
		}
		break;
		}

		// Allocate memory on the host:
		double *h_paths = (double *)malloc(nPaths*size * sizeof(double));
		// Copy from device to host:
		cudaMemcpy(h_paths, d_paths, nPaths*size * sizeof(double),
			cudaMemcpyKind::cudaMemcpyDeviceToHost);


		std::vector<std::vector<double>> paths(nPaths);
		for (std::size_t s = 0; s < paths.size(); ++s) {
			std::vector<double> path(size);
			for (std::size_t p = 0; p < path.size(); ++p) {
				path[p] = std::move(h_paths[s + paths.size()*p]);
			}
			paths[s] = std::move(path);
		}
		free(h_times);
		free(h_paths);
		cudaFree(d_paths);
		cudaFree(d_times);
		cudaFree(states);
		return paths;
	}


	//=====================================================================
	//====== equidistant overloads
	//=====================================================================

	void fdm_engine_cuda::HestonPathEngineFloat::_generate1D(float *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, unsigned int nSteps, float dt) const {
		// initialise RNG states
		const unsigned int threadsPerBlock = THREADS_PER_BLOCK;
		unsigned int blocksPerGrid = (nPaths + threadsPerBlock - 1) / threadsPerBlock;
		random_kernel_initializers::initialiseRandomKernel1D << <threadsPerBlock, blocksPerGrid >> > (time(0), states, nPaths);
		switch (scheme) {
		case FDMScheme::EulerScheme:
		{
			heston_kernels::euler_scheme::generatePathsKernel1D<float> << <threadsPerBlock, blocksPerGrid >> > (this->heston_, d_paths, states,
				nPaths, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			heston_kernels::milstein_scheme::generatePathsKernel1D<float> << <threadsPerBlock, blocksPerGrid >> > (this->heston_, d_paths, states,
				nPaths, nSteps, dt);
		}
		break;
		}
	}

	void fdm_engine_cuda::HestonPathEngineFloat::_generate2D(float *d_paths, curandState_t *states, FDMScheme scheme,
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
			heston_kernels::euler_scheme::generatePathsKernel2D<float> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			heston_kernels::milstein_scheme::generatePathsKernel2D<float> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, nSteps, dt);
		}
		break;
		}
	}

	void fdm_engine_cuda::HestonPathEngineFloat::_generate3D(float *d_paths, curandState_t *states, FDMScheme scheme,
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
			heston_kernels::euler_scheme::generatePathsKernel3D<float> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, depthSize, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			heston_kernels::milstein_scheme::generatePathsKernel3D<float> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, depthSize, nSteps, dt);
		}
		break;
		}
	}

	//=====================================================================
	//====== non-equidistant overloads
	//=====================================================================

	void fdm_engine_cuda::HestonPathEngineFloat::_generate1D(float *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, float const *d_times, unsigned int size) const {
		// initialise RNG states
		const unsigned int threadsPerBlock = THREADS_PER_BLOCK;
		unsigned int blocksPerGrid = (nPaths + threadsPerBlock - 1) / threadsPerBlock;
		random_kernel_initializers::initialiseRandomKernel1D << <threadsPerBlock, blocksPerGrid >> > (time(0), states, nPaths);
		switch (scheme) {
		case FDMScheme::EulerScheme:
		{
			heston_kernels::euler_scheme::generatePathsKernel1D<float> << <threadsPerBlock, blocksPerGrid >> > (this->heston_, d_paths, states,
				nPaths, d_times, size);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			heston_kernels::milstein_scheme::generatePathsKernel1D<float> << <threadsPerBlock, blocksPerGrid >> > (this->heston_, d_paths, states,
				nPaths, d_times, size);
		}
		break;
		}
	}

	void fdm_engine_cuda::HestonPathEngineFloat::_generate2D(float *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, float const *d_times, unsigned int size)const {
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
			heston_kernels::euler_scheme::generatePathsKernel2D<float> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, d_times, size);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			heston_kernels::milstein_scheme::generatePathsKernel2D<float> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, d_times, size);
		}
		break;
		}
	}

	void fdm_engine_cuda::HestonPathEngineFloat::_generate3D(float *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, float const *d_times, unsigned int size)const {
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
			heston_kernels::euler_scheme::generatePathsKernel3D<float> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, depthSize, d_times, size);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			heston_kernels::milstein_scheme::generatePathsKernel3D<float> << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, depthSize, d_times, size);
		}
		break;
		}
	}

	//=====================================================================
	//====== equidistant overloads
	//=====================================================================

	fdm_engine_cuda::PathValuesType<fdm_engine_cuda::PathValuesType<float>>
		fdm_engine_cuda::HestonPathEngineFloat::simulate(unsigned int nPaths, unsigned int nSteps, float dt,
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

	//=====================================================================
	//====== non-equidistant overloads
	//=====================================================================

	fdm_engine_cuda::PathValuesType<fdm_engine_cuda::PathValuesType<float>>
		fdm_engine_cuda::HestonPathEngineFloat::simulate(unsigned int nPaths,TimePointsType<float> const &timePoints,
			FDMScheme scheme, GPUConfiguration config)const {

		unsigned int size = timePoints.size();
		// Allocate memory on the host:
		float *h_times = (float*)malloc(size * sizeof(float));
		// Copy:
		for (std::size_t t = 0; t < size; ++t) {
			h_times[t] = timePoints.at(t);
		}

		float *d_paths = NULL;
		float *d_times = NULL;
		// RNG state for each thread
		curandState_t *states;
		// Allocate memory for the paths
		cudaMalloc(&d_paths, nPaths * size * sizeof(float));
		// Allocate memory for the times
		cudaMalloc(&d_times, size * sizeof(float));
		// Allocate memory for RNG states
		cudaMalloc(&states, nPaths * sizeof(curandState_t));
		// Copy h_times to d_times,i.e. from hosdt to device:
		cudaMemcpy(d_times, h_times, size * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice);

		switch (config) {
		case GPUConfiguration::Grid1D:
		{
			_generate1D(d_paths, states, scheme, nPaths, d_times, size);
		}
		break;
		case GPUConfiguration::Grid2D:
		{
			_generate2D(d_paths, states, scheme, nPaths, d_times, size);
		}
		break;
		case GPUConfiguration::Grid3D:
		{
			_generate3D(d_paths, states, scheme, nPaths, d_times, size);
		}
		break;
		default:
		{
			_generate1D(d_paths, states, scheme, nPaths, d_times, size);
		}
		break;
		}

		// Allocate memory on the host:
		float *h_paths = (float *)malloc(nPaths*size * sizeof(float));
		// Copy from device to host:
		cudaMemcpy(h_paths, d_paths, nPaths*size * sizeof(float),
			cudaMemcpyKind::cudaMemcpyDeviceToHost);


		std::vector<std::vector<float>> paths(nPaths);
		for (std::size_t s = 0; s < paths.size(); ++s) {
			std::vector<float> path(size);
			for (std::size_t p = 0; p < path.size(); ++p) {
				path[p] = std::move(h_paths[s + paths.size()*p]);
			}
			paths[s] = std::move(path);
		}
		free(h_times);
		free(h_paths);
		cudaFree(d_paths);
		cudaFree(d_times);
		cudaFree(states);
		return paths;
	}



}


