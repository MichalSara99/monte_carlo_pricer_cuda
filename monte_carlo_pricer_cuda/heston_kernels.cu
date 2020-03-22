#include<cuda_runtime.h>
#include<curand.h>
#include<curand_kernel.h>
#include<device_launch_parameters.h>
#include"sde_builder_cuda.h"
#include<cassert>
#include"random_kernel_initializers.cuh"
#include"mc_types.h"


namespace heston_kernels {


	namespace euler_scheme {

		__global__
			void generatePathsKernelDouble1D(sde_builder_cuda::HestonModel<double> heston, double *d_paths, curandState_t* states,
				unsigned int nPaths, unsigned int nSteps, double dt) {
			// Path index
			const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < nPaths) {
				double last = heston.init1();
				double var = heston.init2();
				double last_new{};
				double var_new{};
				double z1{};
				double z2{};
				d_paths[idx] = last;

				unsigned int i = 0;
				for (int k = idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					z1 = curand_normal(&states[idx]);
					z2 = curand_normal(&states[idx]);
					last_new = last + heston.drift1(i*dt, last, var)*dt +
						heston.diffusion1(i*dt, last, var)*sqrtf(dt)*z1;
					var_new = var + heston.drift2(i*dt, last, var)*dt +
						heston.diffusion2(i*dt, last, var)*sqrtf(dt)*
						(heston.rho() * z1 + sqrt(1.0 - heston.rho()*heston.rho()) * z2);
					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}

		__global__
			void generatePathsKernelFloat1D(sde_builder_cuda::HestonModel<float> heston, float *d_paths, curandState_t* states,
				unsigned int nPaths, unsigned int nSteps, float dt) {
			// Path index
			const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < nPaths) {
				float last = heston.init1();
				float var = heston.init2();
				float last_new{};
				float var_new{};
				float z1{};
				float z2{};
				d_paths[idx] = last;

				unsigned int i = 0;
				for (int k = idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					z1 = curand_normal(&states[idx]);
					z2 = curand_normal(&states[idx]);
					last_new = last + heston.drift1(i*dt, last, var)*dt +
						heston.diffusion1(i*dt, last, var)*sqrtf(dt)*z1;
					var_new = var + heston.drift2(i*dt, last, var)*dt +
						heston.diffusion2(i*dt, last, var)*sqrtf(dt)*
						(heston.rho() * z1 + sqrt(1.0 - heston.rho()*heston.rho()) * z2);
					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}

		__global__
			void generatePathsKernelDouble2D(sde_builder_cuda::HestonModel<double> heston, double *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nSteps, double dt) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight;

			if (t_idx < nPaths) {
				double last = heston.init1();
				double var = heston.init2();
				double last_new{};
				double var_new{};
				double z1{};
				double z2{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					last_new = last + heston.drift1(i*dt, last, var)*dt +
						heston.diffusion1(i*dt, last, var)*sqrtf(dt)*z1;
					var_new = var + heston.drift2(i*dt, last, var)*dt +
						heston.diffusion2(i*dt, last, var)*sqrtf(dt)*
						(heston.rho() * z1 + sqrt(1.0 - heston.rho()*heston.rho()) * z2);
					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}

		__global__
			void generatePathsKernelFloat2D(sde_builder_cuda::HestonModel<float> heston, float *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nSteps, float dt) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight;

			if (t_idx < nPaths) {
				float last = heston.init1();
				float var = heston.init2();
				float last_new{};
				float var_new{};
				float z1{};
				float z2{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					last_new = last + heston.drift1(i*dt, last, var)*dt +
						heston.diffusion1(i*dt, last, var)*sqrtf(dt)*z1;
					var_new = var + heston.drift2(i*dt, last, var)*dt +
						heston.diffusion2(i*dt, last, var)*sqrtf(dt)*
						(heston.rho() * z1 + sqrt(1.0 - heston.rho()*heston.rho()) * z2);
					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}


		__global__
			void generatePathsKernelDouble3D(sde_builder_cuda::HestonModel<double> heston, double *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
				unsigned int nSteps, double dt) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int l_idx = blockIdx.z * blockDim.z + threadIdx.z;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight*l_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight * nPathsDepth;

			if (t_idx < nPaths) {
				double last = heston.init1();
				double var = heston.init2();
				double last_new{};
				double var_new{};
				double z1{};
				double z2{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					last_new = last + heston.drift1(i*dt, last, var)*dt +
						heston.diffusion1(i*dt, last, var)*sqrtf(dt)*z1;
					var_new = var + heston.drift2(i*dt, last, var)*dt +
						heston.diffusion2(i*dt, last, var)*sqrtf(dt)*
						(heston.rho() * z1 + sqrt(1.0 - heston.rho()*heston.rho()) * z2);
					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}

		__global__
			void generatePathsKernelFloat3D(sde_builder_cuda::HestonModel<float> heston, float *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
				unsigned int nSteps, float dt) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int l_idx = blockIdx.z * blockDim.z + threadIdx.z;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight*l_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight * nPathsDepth;

			if (t_idx < nPaths) {
				float last = heston.init1();
				float var = heston.init2();
				float last_new{};
				float var_new{};
				float z1{};
				float z2{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					last_new = last + heston.drift1(i*dt, last, var)*dt +
						heston.diffusion1(i*dt, last, var)*sqrtf(dt)*z1;
					var_new = var + heston.drift2(i*dt, last, var)*dt +
						heston.diffusion2(i*dt, last, var)*sqrtf(dt)*
						(heston.rho() * z1 + sqrt(1.0 - heston.rho()*heston.rho()) * z2);
					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}


	};


	namespace milstein_scheme {






	};


};


namespace fdm_engine_cuda {

	using mc_types::FDMScheme;
	using mc_types::GPUConfiguration;
	using mc_types::PathValuesType;

	void fdm_engine_cuda::HestonPathEngineDouble::_generate1D(double *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, unsigned int nSteps, double dt) const {
		// initialise RNG states
		const unsigned int threadsPerBlock = THREADS_PER_BLOCK;
		unsigned int blocksPerGrid = (nPaths + threadsPerBlock - 1) / threadsPerBlock;
		random_kernel_initializers::initialiseRandomKernel1D << <threadsPerBlock, blocksPerGrid >> >(time(0), states, nPaths);
		switch (scheme) {
		case FDMScheme::EulerScheme:
		{
			heston_kernels::euler_scheme::generatePathsKernelDouble1D << <threadsPerBlock, blocksPerGrid >> >(this->heston_, d_paths, states,
				nPaths, nSteps, dt);

		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			throw std::exception("Not yet impolemented!");
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
			heston_kernels::euler_scheme::generatePathsKernelDouble2D << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			throw std::exception("Not yet impolemented!");
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
			heston_kernels::euler_scheme::generatePathsKernelDouble3D << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, depthSize, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			throw std::exception("Not yet impolemented!");
		}
		break;
		}
	}


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



	void fdm_engine_cuda::HestonPathEngineFloat::_generate1D(float *d_paths, curandState_t *states, FDMScheme scheme,
		unsigned int nPaths, unsigned int nSteps, float dt) const {
		// initialise RNG states
		const unsigned int threadsPerBlock = THREADS_PER_BLOCK;
		unsigned int blocksPerGrid = (nPaths + threadsPerBlock - 1) / threadsPerBlock;
		random_kernel_initializers::initialiseRandomKernel1D << <threadsPerBlock, blocksPerGrid >> > (time(0), states, nPaths);
		switch (scheme) {
		case FDMScheme::EulerScheme:
		{
			heston_kernels::euler_scheme::generatePathsKernelFloat1D << <threadsPerBlock, blocksPerGrid >> > (this->heston_, d_paths, states,
				nPaths, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			throw std::exception("Not yet implementd.");
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
			heston_kernels::euler_scheme::generatePathsKernelFloat2D << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			throw std::exception("Not yet implementd.");
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
			heston_kernels::euler_scheme::generatePathsKernelFloat3D << <gridSize, blockSize >> > (this->heston_, d_paths, states,
				widthSize, heightSize, depthSize, nSteps, dt);
		}
		break;
		case FDMScheme::MilsteinScheme:
		{
			throw std::exception("Not yet implementd.");
		}
		break;
		}
	}


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



}


