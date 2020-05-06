#pragma once
#if !defined(_KERNELS)
#define _KERNELS

#include<device_launch_parameters.h>
#include<curand_kernel.h>
#include"mc_types.h"

namespace kernels {

	namespace one_factor_kernels {


		namespace euler_scheme {

			// ==========================================================
			// ===== Kernels for equidistant steps 
			// ==========================================================

			template<typename SDEBuilder,
					typename T = SDEBuilder::value_type>
			__global__
			void generatePathsKernel1D(SDEBuilder sdeBuilder,T* d_paths, curandState_t *states,
					unsigned int nPaths, unsigned int nSteps, T dt) {
				// path idx:
				unsigned int const t_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const total = nSteps * nPaths;
				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init();
				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					last = last + sdeBuilder.drift(i*dt, last)*dt +
						sdeBuilder.diffusion(i*dt, last)*sqrtf(dt)*curand_normal(&states[t_idx]);
					d_paths[t] = last;
				}

			}


			template<typename SDEBuilder,
					typename T = SDEBuilder::value_type>
			__global__
			void generatePathsKernel2D(SDEBuilder sdeBuilder,T* d_paths, curandState_t *states,
					unsigned int nPathsWidth,unsigned int nPathsHeight, unsigned int nSteps, T dt) {
				// path idx:
				unsigned int const c_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y*blockIdx.y + threadIdx.y;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth;
				unsigned int const nPaths = nPathsWidth * nPathsHeight;
				unsigned int const total = nPaths * nSteps;
				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init();
				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					last = last + sdeBuilder.drift(i*dt, last)*dt +
						sdeBuilder.diffusion(i*dt, last)*sqrtf(dt)*curand_normal(&states[t_idx]);
					d_paths[t] = last;
				}

			}


			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel3D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPathsWidth, unsigned int nPathsHeight,unsigned int nPathDepth,
					unsigned int nSteps, T dt) {
				// path idx:
				unsigned int const c_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y*blockIdx.y + threadIdx.y;
				unsigned int const l_idx = blockDim.z*blockIdx.z + threadIdx.z;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth + l_idx * nPathsWidth*nPathsHeight;
				unsigned int const nPaths = nPathsWidth * nPathsHeight * nPathDepth;
				unsigned int const total = nPaths * nSteps;
				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init();
				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					last = last + sdeBuilder.drift(i*dt, last)*dt +
						sdeBuilder.diffusion(i*dt, last)*sqrtf(dt)*curand_normal(&states[t_idx]);
					d_paths[t] = last;
				}
			}

			// ==========================================================
			// ===== Kernels for non-equidistant steps 
			// ==========================================================

			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel1D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPaths, T const *d_times, unsigned int size) {
				// path idx:
				unsigned int const t_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const total = size * nPaths;
				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init();
				T dt{};
				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					dt = d_times[i + 1] - d_times[i];
					last = last + sdeBuilder.drift(d_times[i], last)*dt +
						sdeBuilder.diffusion(d_times[i], last)*sqrtf(dt)*curand_normal(&states[t_idx]);
					d_paths[t] = last;
				}


			}


			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel2D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPathsWidth, unsigned int nPathsHeight,
					T const *d_times, unsigned int size) {
				// path idx:
				unsigned int const c_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y*blockIdx.y + threadIdx.y;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth;
				unsigned int const nPaths = nPathsWidth * nPathsHeight;
				unsigned int const total = nPaths * size;
				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init();
				T dt{};
				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					dt = d_times[i + 1] - d_times[i];
					last = last + sdeBuilder.drift(d_times[i], last)*dt +
						sdeBuilder.diffusion(d_times[i], last)*sqrtf(dt)*curand_normal(&states[t_idx]);
					d_paths[t] = last;
				}
			}


			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel3D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathDepth,
					T const *d_times, unsigned int size) {
				// path idx:
				unsigned int const c_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y*blockIdx.y + threadIdx.y;
				unsigned int const l_idx = blockDim.z*blockIdx.z + threadIdx.z;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth + l_idx * nPathsWidth*nPathsHeight;
				unsigned int const nPaths = nPathsWidth * nPathsHeight * nPathDepth;
				unsigned int const total = nPaths * size;
				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init();
				T dt{};
				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					dt = d_times[i + 1] - d_times[i];
					last = last + sdeBuilder.drift(d_times[i], last)*dt +
						sdeBuilder.diffusion(d_times[i], last)*sqrtf(dt)*curand_normal(&states[t_idx]);
					d_paths[t] = last;
				}
			}


		}

		namespace milstein_scheme {

			// ==========================================================
			// ===== Kernels for equidistant steps 
			// ==========================================================

			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel1D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPaths, unsigned int nSteps, T dt) {
				// path idx:
				unsigned int const t_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const total = nSteps * nPaths;
				if (t_idx >= nPaths)
					return;

				T z{};
				unsigned int i{ 0 };
				T last = sdeBuilder.init();
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z = curand_normal(&states[t_idx]);
					last = last + sdeBuilder.drift(i*dt, last)*dt +
						sdeBuilder.diffusion(i*dt, last)*sqrtf(dt)*z +
						0.5*sdeBuilder.diffusion(i*dt, last) *
						((sdeBuilder.diffusion(i*dt, last + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion(i*dt, last - 0.5*DIFF_STEP)) / DIFF_STEP)*
							((sqrtf(dt)*z)*(sqrtf(dt)*z) - dt);
					d_paths[t] = last;
				}
			}


			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel2D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nSteps, T dt) {
				// path idx:
				unsigned int const c_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y*blockIdx.y + threadIdx.y;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth;
				unsigned int const nPaths = nPathsWidth * nPathsHeight;
				unsigned int const total = nPaths * nSteps;
				if (t_idx >= nPaths)
					return;

				T z{};
				unsigned int i{ 0 };
				T last = sdeBuilder.init();
				d_paths[t_idx] = last;
				
				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths,++i) {
					z = curand_normal(&states[t_idx]);
					last = last + sdeBuilder.drift(i*dt, last)*dt +
						sdeBuilder.diffusion(i*dt, last)*sqrtf(dt)*z +
						0.5*sdeBuilder.diffusion(i*dt, last) *
						((sdeBuilder.diffusion(i*dt, last + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion(i*dt, last - 0.5*DIFF_STEP)) / DIFF_STEP)*
							((sqrtf(dt)*z)*(sqrtf(dt)*z) - dt);
					d_paths[t] = last;
				}
			}


			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
			__global__
				void generatePathsKernel3D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathDepth,
					unsigned int nSteps, T dt) {
				// path idx:
				unsigned int const c_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y*blockIdx.y + threadIdx.y;
				unsigned int const l_idx = blockDim.z*blockIdx.z + threadIdx.z;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth + l_idx * nPathsWidth*nPathsHeight;
				unsigned int const nPaths = nPathsWidth * nPathsHeight * nPathDepth;
				unsigned int const total = nPaths * nSteps;
				if (t_idx >= nPaths)
					return;

				T z{};
				unsigned int i{ 0 };
				T last = sdeBuilder.init();
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z = curand_normal(&states[t_idx]);
					last = last + sdeBuilder.drift(i*dt, last)*dt +
						sdeBuilder.diffusion(i*dt, last)*sqrtf(dt)*z +
						0.5*sdeBuilder.diffusion(i*dt, last) *
						((sdeBuilder.diffusion(i*dt, last + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion(i*dt, last - 0.5*DIFF_STEP)) / DIFF_STEP)*
							((sqrtf(dt)*z)*(sqrtf(dt)*z) - dt);
					d_paths[t] = last;
				}
			}

			// ==========================================================
			// ===== Kernels for non-equidistant steps 
			// ==========================================================

			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel1D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPaths, T const *d_times, unsigned int size) {
				// path idx:
				unsigned int const t_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const total = size * nPaths;
				if (t_idx >= nPaths)
					return;

				T dt{};
				T z{};
				unsigned int i{ 0 };
				T last = sdeBuilder.init();
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths,++i) {
					z = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];
					last = last + sdeBuilder.drift(d_times[i], last)*dt +
						sdeBuilder.diffusion(d_times[i], last)*sqrtf(dt)*z +
						0.5*sdeBuilder.diffusion(d_times[i], last) *
						((sdeBuilder.diffusion(d_times[i], last + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion(d_times[i], last - 0.5*DIFF_STEP)) / DIFF_STEP)*
							((sqrtf(dt)*z)*(sqrtf(dt)*z) - dt);
					d_paths[t] = last;
				}
			}


			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
			__global__
				void generatePathsKernel2D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPathsWidth, unsigned int nPathsHeight,
					T const *d_times, unsigned int size) {
				// path idx:
				unsigned int const c_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y*blockIdx.y + threadIdx.y;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth;
				unsigned int const nPaths = nPathsWidth * nPathsHeight;
				unsigned int const total = nPaths * size;
				if (t_idx >= nPaths)
					return;

				T dt{};
				T z{};
				unsigned int i{ 0 };
				T last = sdeBuilder.init();
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];
					last = last + sdeBuilder.drift(d_times[i], last)*dt +
						sdeBuilder.diffusion(d_times[i], last)*sqrtf(dt)*z +
						0.5*sdeBuilder.diffusion(d_times[i], last) *
						((sdeBuilder.diffusion(d_times[i], last + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion(d_times[i], last - 0.5*DIFF_STEP)) / DIFF_STEP)*
							((sqrtf(dt)*z)*(sqrtf(dt)*z) - dt);
					d_paths[t] = last;
				}
			}


			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
			__global__
				void generatePathsKernel3D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathDepth,
					T const *d_times, unsigned int size) {
				// path idx:
				unsigned int const c_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y*blockIdx.y + threadIdx.y;
				unsigned int const l_idx = blockDim.z*blockIdx.z + threadIdx.z;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth + l_idx * nPathsWidth*nPathsHeight;
				unsigned int const nPaths = nPathsWidth * nPathsHeight * nPathDepth;
				unsigned int const total = nPaths * size;
				if (t_idx >= nPaths)
					return;
				
				T z{};
				T dt{};
				unsigned int i{ 0 };
				T last = sdeBuilder.init();
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];
					last = last + sdeBuilder.drift(d_times[i], last)*dt +
						sdeBuilder.diffusion(d_times[i], last)*sqrtf(dt)*z +
						0.5*sdeBuilder.diffusion(d_times[i], last) *
						((sdeBuilder.diffusion(d_times[i], last + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion(d_times[i], last - 0.5*DIFF_STEP)) / DIFF_STEP)*
							((sqrtf(dt)*z)*(sqrtf(dt)*z) - dt);
					d_paths[t] = last;
				}
			}



		}



	}



}






#endif ///_KERNELS