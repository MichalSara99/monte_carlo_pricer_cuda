#pragma once
#if !defined(_GBM_KERNELS)
#define _GBM_KERNELS

#include"sde_builder_cuda.h"
#include"device_launch_parameters.h"

namespace gbm_kernels {



	namespace euler_scheme {

		// ==========================================================
		// ===== Kernels for equidistant steps 
		// ==========================================================

		template<typename T>
		__global__
			void generatePathsKernel1D(sde_builder_cuda::GeometricBrownianMotion<T> gbm,
				T *d_paths, curandState_t* states,
				unsigned int nPaths, unsigned int nSteps, T dt) {
			// Path index
			const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < nPaths) {
				T last = gbm.init();
				d_paths[idx] = last;

				unsigned int i = 0;
				for (int k = idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					last = last + gbm.drift(i*dt, last)*dt +
						gbm.diffusion(i*dt, last)*sqrtf(dt)*curand_normal(&states[idx]);
					d_paths[k] = last;
					i++;
				}
			}
		}


		template<typename T>
		__global__
			void generatePathsKernel2D(sde_builder_cuda::GeometricBrownianMotion<T> gbm,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nSteps, T dt) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight;

			if (t_idx < nPaths) {
				T last = gbm.init();
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					last = last + gbm.drift(i*dt, last)*dt +
						gbm.diffusion(i*dt, last)*sqrtf(dt)*curand_normal(&states[t_idx]);
					d_paths[k] = last;
					i++;
				}
			}
		}

		template<typename T>
		__global__
			void generatePathsKernel3D(sde_builder_cuda::GeometricBrownianMotion<T> gbm,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
				unsigned int nSteps, T dt) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int l_idx = blockIdx.z * blockDim.z + threadIdx.z;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight* l_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight * nPathsDepth;

			if (t_idx < nPaths) {
				T last = gbm.init();
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					last = last + gbm.drift(i*dt, last)*dt +
						gbm.diffusion(i*dt, last)*sqrtf(dt)*curand_normal(&states[t_idx]);
					d_paths[k] = last;
					i++;
				}
			}
		}


		// ==========================================================
		// ===== Kernels for non-equidistant steps 
		// ==========================================================

		template<typename T>
		__global__
			void generatePathsKernel1D(sde_builder_cuda::GeometricBrownianMotion<T> gbm,
				T *d_paths, curandState_t* states,
				unsigned int nPaths, T const *d_times, unsigned int size) {
			// Path index
			const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < nPaths) {
				T last = gbm.init();
				T dt{};
				d_paths[idx] = last;

				unsigned int i = 0;
				for (int k = idx + nPaths; k < size*nPaths; k += nPaths) {
					dt = d_times[i + 1] - d_times[i];
					last = last + gbm.drift(d_times[i], last)*(dt) +
						gbm.diffusion(d_times[i], last)*sqrtf(dt)*curand_normal(&states[idx]);
					d_paths[k] = last;
					i++;
				}
			}
		}


		template<typename T>
		__global__
			void generatePathsKernel2D(sde_builder_cuda::GeometricBrownianMotion<T> gbm,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, T const *d_times, unsigned int size) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight;

			if (t_idx < nPaths) {
				T last = gbm.init();
				T dt{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < size*nPaths; k += nPaths) {
					dt = d_times[i + 1] - d_times[i];
					last = last + gbm.drift(d_times[i], last)*(dt) +
						gbm.diffusion(d_times[i], last)*sqrtf(dt)*curand_normal(&states[t_idx]);
					d_paths[k] = last;
					i++;
				}
			}
		}



		template<typename T>
		__global__
			void generatePathsKernel3D(sde_builder_cuda::GeometricBrownianMotion<T> gbm,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
				T const *d_times, unsigned int size) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int l_idx = blockIdx.z * blockDim.z + threadIdx.z;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight* l_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight * nPathsDepth;

			if (t_idx < nPaths) {
				T last = gbm.init();
				T dt{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < size*nPaths; k += nPaths) {
					dt = d_times[i + 1] - d_times[i];
					last = last + gbm.drift(d_times[i], last)*(dt) +
						gbm.diffusion(d_times[i], last)*sqrtf(dt)*curand_normal(&states[t_idx]);
					d_paths[k] = last;
					i++;
				}
			}
		}
	};



	namespace milstein_scheme {

		// ==========================================================
		// ===== Kernels for equidistant steps 
		// ==========================================================

		template<typename T>
		__global__
			void generatePathsKernel1D(sde_builder_cuda::GeometricBrownianMotion<T> gbm,
				T *d_paths, curandState_t* states,
				unsigned int nPaths, unsigned int nSteps, T dt) {
			// Path index
			const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < nPaths) {
				T last = gbm.init();
				T z{};
				d_paths[idx] = last;

				unsigned int i = 0;
				for (unsigned int k = idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					z = curand_normal(&states[idx]);
					last = last + gbm.drift(i*dt, last)*dt +
						gbm.diffusion(i*dt, last)*sqrtf(dt)*z +
						0.5*gbm.diffusion(i*dt, last) *
						((gbm.diffusion(i*dt, last + 0.5*DIFF_STEP) -
							gbm.diffusion(i*dt, last - 0.5*DIFF_STEP)) / DIFF_STEP)*
							((sqrtf(dt)*z)*(sqrtf(dt)*z) - dt);
					d_paths[k] = last;
					i++;
				}
			}
		}

		template<typename T>
		__global__
			void generatePathsKernel2D(sde_builder_cuda::GeometricBrownianMotion<T> gbm,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nSteps, T dt) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight;

			if (t_idx < nPaths) {
				T last = gbm.init();
				T z{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					z = curand_normal(&states[t_idx]);
					last = last + gbm.drift(i*dt, last)*dt +
						gbm.diffusion(i*dt, last)*sqrtf(dt)*z +
						0.5*gbm.diffusion(i*dt, last) *
						((gbm.diffusion(i*dt, last + 0.5*DIFF_STEP) -
							gbm.diffusion(i*dt, last - 0.5*DIFF_STEP)) / DIFF_STEP)*
							((sqrtf(dt)*z)*(sqrtf(dt)*z) - dt);
					d_paths[k] = last;
					i++;
				}
			}
		}

		template<typename T>
		__global__
			void generatePathsKernel3D(sde_builder_cuda::GeometricBrownianMotion<T> gbm,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
				unsigned int nSteps, T dt) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int l_idx = blockIdx.z * blockDim.z + threadIdx.z;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight* l_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight * nPathsDepth;

			if (t_idx < nPaths) {
				T last = gbm.init();
				T z{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					z = curand_normal(&states[t_idx]);
					last = last + gbm.drift(i*dt, last)*dt +
						gbm.diffusion(i*dt, last)*sqrtf(dt)*z +
						0.5*gbm.diffusion(i*dt, last) *
						((gbm.diffusion(i*dt, last + 0.5*DIFF_STEP) -
							gbm.diffusion(i*dt, last - 0.5*DIFF_STEP)) / DIFF_STEP)*
							((sqrtf(dt)*z)*(sqrtf(dt)*z) - dt);
					d_paths[k] = last;
					i++;
				}
			}
		}

		// ==========================================================
		// ===== Kernels for non-equidistant steps 
		// ==========================================================

		template<typename T>
		__global__
			void generatePathsKernel1D(sde_builder_cuda::GeometricBrownianMotion<T> gbm,
				T *d_paths, curandState_t* states,
				unsigned int nPaths, T const *d_times, unsigned int size) {
			// Path index
			const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < nPaths) {
				T last = gbm.init();
				T z{};
				T dt{};
				d_paths[idx] = last;

				unsigned int i = 0;
				for (unsigned int k = idx + nPaths; k < size*nPaths; k += nPaths) {
					z = curand_normal(&states[idx]);
					dt = d_times[i + 1] - d_times[i];
					last = last + gbm.drift(d_times[i], last)*(dt) +
						gbm.diffusion(d_times[i], last)*sqrtf(dt)*z +
						0.5*gbm.diffusion(d_times[i], last) *
						((gbm.diffusion(d_times[i], last + 0.5*DIFF_STEP) -
							gbm.diffusion(d_times[i], last - 0.5*DIFF_STEP)) / DIFF_STEP)*
							((sqrtf(dt)*z)*(sqrtf(dt)*z) - (dt));
					d_paths[k] = last;
					i++;
				}
			}
		}

		template<typename T>
		__global__
			void generatePathsKernel2D(sde_builder_cuda::GeometricBrownianMotion<T> gbm,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, T const *d_times, unsigned int size) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight;

			if (t_idx < nPaths) {
				T last = gbm.init();
				T z{};
				T dt{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < size*nPaths; k += nPaths) {
					z = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];
					last = last + gbm.drift(d_times[i], last)*(dt) +
						gbm.diffusion(d_times[i], last)*sqrtf(dt)*z +
						0.5*gbm.diffusion(d_times[i], last) *
						((gbm.diffusion(d_times[i], last + 0.5*DIFF_STEP) -
							gbm.diffusion(d_times[i], last - 0.5*DIFF_STEP)) / DIFF_STEP)*
							((sqrtf(dt)*z)*(sqrtf(dt)*z) - dt);
					d_paths[k] = last;
					i++;
				}
			}
		}

		template<typename T>
		__global__
			void generatePathsKernel3D(sde_builder_cuda::GeometricBrownianMotion<T> gbm,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
				T const *d_times, unsigned int size) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int l_idx = blockIdx.z * blockDim.z + threadIdx.z;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight* l_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight * nPathsDepth;

			if (t_idx < nPaths) {
				T last = gbm.init();
				T z{};
				T dt{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < size*nPaths; k += nPaths) {
					z = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];
					last = last + gbm.drift(i*dt, last)*dt +
						gbm.diffusion(d_times[i], last)*sqrtf(dt)*z +
						0.5*gbm.diffusion(d_times[i], last) *
						((gbm.diffusion(d_times[i], last + 0.5*DIFF_STEP) -
							gbm.diffusion(d_times[i], last - 0.5*DIFF_STEP)) / DIFF_STEP)*
							((sqrtf(dt)*z)*(sqrtf(dt)*z) - dt);
					d_paths[k] = last;
					i++;
				}
			}
		}

	};

}




#endif ///GBM_KERNELS