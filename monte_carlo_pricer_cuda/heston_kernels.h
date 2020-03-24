#pragma once
#if !defined(_HESTON_KERNELS)
#define _HESTON_KERNELS

#include<device_launch_parameters.h>
#include"sde_builder_cuda.h"

namespace heston_kernels {


	namespace euler_scheme {

		// ==========================================================
		// ===== Kernels for equidistant steps 
		// ==========================================================

		template<typename T>
		__global__
			void generatePathsKernel1D(sde_builder_cuda::HestonModel<T> heston,
				T *d_paths, curandState_t* states,
				unsigned int nPaths, unsigned int nSteps, T dt) {
			// Path index
			const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < nPaths) {
				T last = heston.init1();
				T var = heston.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
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


		template<typename T>
		__global__
			void generatePathsKernel2D(sde_builder_cuda::HestonModel<T> heston,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nSteps, T dt) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight;

			if (t_idx < nPaths) {
				T last = heston.init1();
				T var = heston.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
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


		template<typename T>
		__global__
			void generatePathsKernel3D(sde_builder_cuda::HestonModel<T> heston,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
				unsigned int nSteps, T dt) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int l_idx = blockIdx.z * blockDim.z + threadIdx.z;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight*l_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight * nPathsDepth;

			if (t_idx < nPaths) {
				T last = heston.init1();
				T var = heston.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
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

		// ==========================================================
		// ===== Kernels for non-equidistant steps 
		// ==========================================================

		template<typename T>
		__global__
			void generatePathsKernel1D(sde_builder_cuda::HestonModel<T> heston,
				T *d_paths, curandState_t* states,
				unsigned int nPaths, T const *d_times, unsigned int size) {
			// Path index
			const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < nPaths) {
				T last = heston.init1();
				T var = heston.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				T dt{};
				d_paths[idx] = last;

				unsigned int i = 0;
				for (int k = idx + nPaths; k < size*nPaths; k += nPaths) {
					z1 = curand_normal(&states[idx]);
					z2 = curand_normal(&states[idx]);
					dt = d_times[i + 1] - d_times[i];
					last_new = last + heston.drift1(d_times[i], last, var)*dt +
						heston.diffusion1(d_times[i], last, var)*sqrtf(dt)*z1;
					var_new = var + heston.drift2(d_times[i], last, var)*dt +
						heston.diffusion2(d_times[i], last, var)*sqrtf(dt)*
						(heston.rho() * z1 + sqrt(1.0 - heston.rho()*heston.rho()) * z2);
					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}


		template<typename T>
		__global__
			void generatePathsKernel2D(sde_builder_cuda::HestonModel<T> heston,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, T const *d_times, unsigned int size) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight;

			if (t_idx < nPaths) {
				T last = heston.init1();
				T var = heston.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				T dt{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < size*nPaths; k += nPaths) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];
					last_new = last + heston.drift1(d_times[i], last, var)*dt +
						heston.diffusion1(d_times[i], last, var)*sqrtf(dt)*z1;
					var_new = var + heston.drift2(d_times[i], last, var)*dt +
						heston.diffusion2(d_times[i], last, var)*sqrtf(dt)*
						(heston.rho() * z1 + sqrt(1.0 - heston.rho()*heston.rho()) * z2);
					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}


		template<typename T>
		__global__
			void generatePathsKernel3D(sde_builder_cuda::HestonModel<T> heston,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
				T const *d_times, unsigned int size) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int l_idx = blockIdx.z * blockDim.z + threadIdx.z;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight*l_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight * nPathsDepth;

			if (t_idx < nPaths) {
				T last = heston.init1();
				T var = heston.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				T dt{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < size*nPaths; k += nPaths) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];
					last_new = last + heston.drift1(d_times[i], last, var)*dt +
						heston.diffusion1(d_times[i], last, var)*sqrtf(dt)*z1;
					var_new = var + heston.drift2(d_times[i], last, var)*dt +
						heston.diffusion2(d_times[i], last, var)*sqrtf(dt)*
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

		// ==========================================================
		// ===== Kernels for equidistant steps 
		// ==========================================================

		template<typename T>
		__global__
			void generatePathsKernel1D(sde_builder_cuda::HestonModel<T> heston, 
				T *d_paths, curandState_t* states,
				unsigned int nPaths, unsigned int nSteps, T dt) {
			// Path index
			const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < nPaths) {
				T last = heston.init1();
				T var = heston.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				d_paths[idx] = last;

				unsigned int i = 0;
				for (unsigned int k = idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					z1 = curand_normal(&states[idx]);
					z2 = curand_normal(&states[idx]);

					last_new = last +
						heston.drift1(i*dt, last, var)*dt +
						heston.diffusion1(i*dt, last, var) *
						sqrtf(dt) * z1 +
						0.5*heston.diffusion1(i*dt, last, var) *
						((heston.diffusion1(i*dt, last + 0.5*DIFF_STEP, var) -
							heston.diffusion1(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt)*((z1)*(z1)-1.0) +
						0.5*(heston.rho())*heston.diffusion2(i*dt, last, var) *
						((heston.diffusion1(i*dt, last, var + 0.5*DIFF_STEP) -
							heston.diffusion1(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*((z1)*(z1)-1.0) +
						sqrtf(1.0 - (heston.rho())*(heston.rho()))*
						heston.diffusion2(i*dt, last, var)*
						((heston.diffusion1(i *dt, last, var + 0.5*DIFF_STEP) -
							heston.diffusion1(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					var_new = var +
						heston.drift2(i*dt, last, var)*dt +
						heston.diffusion2(i*dt, last, var) *
						sqrtf(dt) *
						(heston.rho() * z1 + sqrtf(1.0 - (heston.rho() * heston.rho())) * z2) +
						0.5*(heston.rho())*heston.diffusion1(i*dt, last, var)*
						((heston.diffusion2(i*dt, last + 0.5*DIFF_STEP, var) -
							heston.diffusion2(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt) * ((z1)*(z1)-1.0) +
						0.5 * heston.diffusion2(i*dt, last, var)*
						((heston.diffusion2(i*dt, last, var + 0.5*DIFF_STEP) -
							heston.diffusion2(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP))*
							(dt) * (((heston.rho())*z1 + std::sqrt(1.0 - (heston.rho())*(heston.rho()))*z2)*
						((heston.rho())*z1 + sqrtf(1.0 - (heston.rho())*(heston.rho()))*z2) - 1.0) +
						sqrtf(1.0 - (heston.rho())*(heston.rho()))*
						heston.diffusion1(i*dt, last, var)*
						((heston.diffusion2(i*dt, last + 0.5*DIFF_STEP, var) -
							heston.diffusion2(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}

		template<typename T>
		__global__
			void generatePathsKernel2D(sde_builder_cuda::HestonModel<T> heston,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nSteps, T dt) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight;

			if (t_idx < nPaths) {
				T last = heston.init1();
				T var = heston.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);

					last_new = last +
						heston.drift1(i*dt, last, var)*dt +
						heston.diffusion1(i*dt, last, var) *
						sqrtf(dt) * z1 +
						0.5*heston.diffusion1(i*dt, last, var) *
						((heston.diffusion1(i*dt, last + 0.5*DIFF_STEP, var) -
							heston.diffusion1(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt)*((z1)*(z1)-1.0) +
						0.5*(heston.rho())*heston.diffusion2(i*dt, last, var) *
						((heston.diffusion1(i*dt, last, var + 0.5*DIFF_STEP) -
							heston.diffusion1(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*((z1)*(z1)-1.0) +
						sqrtf(1.0 - (heston.rho())*(heston.rho()))*
						heston.diffusion2(i*dt, last, var)*
						((heston.diffusion1(i *dt, last, var + 0.5*DIFF_STEP) -
							heston.diffusion1(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					var_new = var +
						heston.drift2(i*dt, last, var)*dt +
						heston.diffusion2(i*dt, last, var) *
						sqrtf(dt) *
						(heston.rho() * z1 + sqrtf(1.0 - (heston.rho() * heston.rho())) * z2) +
						0.5*(heston.rho())*heston.diffusion1(i*dt, last, var)*
						((heston.diffusion2(i*dt, last + 0.5*DIFF_STEP, var) -
							heston.diffusion2(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt) * ((z1)*(z1)-1.0) +
						0.5 * heston.diffusion2(i*dt, last, var)*
						((heston.diffusion2(i*dt, last, var + 0.5*DIFF_STEP) -
							heston.diffusion2(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP))*
							(dt) * (((heston.rho())*z1 + std::sqrt(1.0 - (heston.rho())*(heston.rho()))*z2)*
						((heston.rho())*z1 + sqrtf(1.0 - (heston.rho())*(heston.rho()))*z2) - 1.0) +
						sqrtf(1.0 - (heston.rho())*(heston.rho()))*
						heston.diffusion1(i*dt, last, var)*
						((heston.diffusion2(i*dt, last + 0.5*DIFF_STEP, var) -
							heston.diffusion2(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}

		template<typename T>
		__global__
			void generatePathsKernel3D(sde_builder_cuda::HestonModel<T> heston,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
				unsigned int nSteps, T dt) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int l_idx = blockIdx.z * blockDim.z + threadIdx.z;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight*l_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight * nPathsDepth;

			if (t_idx < nPaths) {
				T last = heston.init1();
				T var = heston.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < nSteps*nPaths; k += nPaths) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);

					last_new = last +
						heston.drift1(i*dt, last, var)*dt +
						heston.diffusion1(i*dt, last, var) *
						sqrtf(dt) * z1 +
						0.5*heston.diffusion1(i*dt, last, var) *
						((heston.diffusion1(i*dt, last + 0.5*DIFF_STEP, var) -
							heston.diffusion1(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt)*((z1)*(z1)-1.0) +
						0.5*(heston.rho())*heston.diffusion2(i*dt, last, var) *
						((heston.diffusion1(i*dt, last, var + 0.5*DIFF_STEP) -
							heston.diffusion1(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*((z1)*(z1)-1.0) +
						sqrtf(1.0 - (heston.rho())*(heston.rho()))*
						heston.diffusion2(i*dt, last, var)*
						((heston.diffusion1(i *dt, last, var + 0.5*DIFF_STEP) -
							heston.diffusion1(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					var_new = var +
						heston.drift2(i*dt, last, var)*dt +
						heston.diffusion2(i*dt, last, var) *
						sqrtf(dt) *
						(heston.rho() * z1 + sqrtf(1.0 - (heston.rho() * heston.rho())) * z2) +
						0.5*(heston.rho())*heston.diffusion1(i*dt, last, var)*
						((heston.diffusion2(i*dt, last + 0.5*DIFF_STEP, var) -
							heston.diffusion2(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt) * ((z1)*(z1)-1.0) +
						0.5 * heston.diffusion2(i*dt, last, var)*
						((heston.diffusion2(i*dt, last, var + 0.5*DIFF_STEP) -
							heston.diffusion2(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP))*
							(dt) * (((heston.rho())*z1 + std::sqrt(1.0 - (heston.rho())*(heston.rho()))*z2)*
						((heston.rho())*z1 + sqrtf(1.0 - (heston.rho())*(heston.rho()))*z2) - 1.0) +
						sqrtf(1.0 - (heston.rho())*(heston.rho()))*
						heston.diffusion1(i*dt, last, var)*
						((heston.diffusion2(i*dt, last + 0.5*DIFF_STEP, var) -
							heston.diffusion2(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}

		// ==========================================================
		// ===== Kernels for non-equidistant steps 
		// ==========================================================

		template<typename T>
		__global__
			void generatePathsKernel1D(sde_builder_cuda::HestonModel<T> heston,
				T *d_paths, curandState_t* states,
				unsigned int nPaths, T const *d_times, unsigned int size) {
			// Path index
			const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

			if (idx < nPaths) {
				T last = heston.init1();
				T var = heston.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				T dt{};
				d_paths[idx] = last;

				unsigned int i = 0;
				for (unsigned int k = idx + nPaths; k < size*nPaths; k += nPaths) {
					z1 = curand_normal(&states[idx]);
					z2 = curand_normal(&states[idx]);
					dt = d_times[i + 1] - d_times[i];
					last_new = last +
						heston.drift1(d_times[i], last, var)*dt +
						heston.diffusion1(d_times[i], last, var) *
						sqrtf(dt) * z1 +
						0.5*heston.diffusion1(d_times[i], last, var) *
						((heston.diffusion1(d_times[i], last + 0.5*DIFF_STEP, var) -
							heston.diffusion1(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt)*((z1)*(z1)-1.0) +
						0.5*(heston.rho())*heston.diffusion2(d_times[i], last, var) *
						((heston.diffusion1(d_times[i], last, var + 0.5*DIFF_STEP) -
							heston.diffusion1(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*((z1)*(z1)-1.0) +
						sqrtf(1.0 - (heston.rho())*(heston.rho()))*
						heston.diffusion2(d_times[i], last, var)*
						((heston.diffusion1(d_times[i], last, var + 0.5*DIFF_STEP) -
							heston.diffusion1(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					var_new = var +
						heston.drift2(d_times[i], last, var)*dt +
						heston.diffusion2(d_times[i], last, var) *
						sqrtf(dt) *
						(heston.rho() * z1 + sqrtf(1.0 - (heston.rho() * heston.rho())) * z2) +
						0.5*(heston.rho())*heston.diffusion1(d_times[i], last, var)*
						((heston.diffusion2(d_times[i], last + 0.5*DIFF_STEP, var) -
							heston.diffusion2(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt) * ((z1)*(z1)-1.0) +
						0.5 * heston.diffusion2(d_times[i], last, var)*
						((heston.diffusion2(d_times[i], last, var + 0.5*DIFF_STEP) -
							heston.diffusion2(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP))*
							(dt) * (((heston.rho())*z1 + std::sqrt(1.0 - (heston.rho())*(heston.rho()))*z2)*
						((heston.rho())*z1 + sqrtf(1.0 - (heston.rho())*(heston.rho()))*z2) - 1.0) +
						sqrtf(1.0 - (heston.rho())*(heston.rho()))*
						heston.diffusion1(d_times[i], last, var)*
						((heston.diffusion2(d_times[i], last + 0.5*DIFF_STEP, var) -
							heston.diffusion2(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}

		template<typename T>
		__global__
			void generatePathsKernel2D(sde_builder_cuda::HestonModel<T> heston,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, T const *d_times, unsigned int size) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight;

			if (t_idx < nPaths) {
				T last = heston.init1();
				T var = heston.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				T dt{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < size*nPaths; k += nPaths) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];
					last_new = last +
						heston.drift1(d_times[i], last, var)*dt +
						heston.diffusion1(d_times[i], last, var) *
						sqrtf(dt) * z1 +
						0.5*heston.diffusion1(d_times[i], last, var) *
						((heston.diffusion1(d_times[i], last + 0.5*DIFF_STEP, var) -
							heston.diffusion1(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt)*((z1)*(z1)-1.0) +
						0.5*(heston.rho())*heston.diffusion2(d_times[i], last, var) *
						((heston.diffusion1(d_times[i], last, var + 0.5*DIFF_STEP) -
							heston.diffusion1(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*((z1)*(z1)-1.0) +
						sqrtf(1.0 - (heston.rho())*(heston.rho()))*
						heston.diffusion2(d_times[i], last, var)*
						((heston.diffusion1(d_times[i], last, var + 0.5*DIFF_STEP) -
							heston.diffusion1(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					var_new = var +
						heston.drift2(d_times[i], last, var)*dt +
						heston.diffusion2(d_times[i], last, var) *
						sqrtf(dt) *
						(heston.rho() * z1 + sqrtf(1.0 - (heston.rho() * heston.rho())) * z2) +
						0.5*(heston.rho())*heston.diffusion1(d_times[i], last, var)*
						((heston.diffusion2(d_times[i], last + 0.5*DIFF_STEP, var) -
							heston.diffusion2(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt) * ((z1)*(z1)-1.0) +
						0.5 * heston.diffusion2(d_times[i], last, var)*
						((heston.diffusion2(d_times[i], last, var + 0.5*DIFF_STEP) -
							heston.diffusion2(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP))*
							(dt) * (((heston.rho())*z1 + std::sqrt(1.0 - (heston.rho())*(heston.rho()))*z2)*
						((heston.rho())*z1 + sqrtf(1.0 - (heston.rho())*(heston.rho()))*z2) - 1.0) +
						sqrtf(1.0 - (heston.rho())*(heston.rho()))*
						heston.diffusion1(d_times[i], last, var)*
						((heston.diffusion2(d_times[i], last + 0.5*DIFF_STEP, var) -
							heston.diffusion2(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}

		template<typename T>
		__global__
			void generatePathsKernel3D(sde_builder_cuda::HestonModel<T> heston,
				T *d_paths, curandState_t* states,
				unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
				T const *d_times, unsigned int size) {
			// Path index
			const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
			const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
			const unsigned int l_idx = blockIdx.z * blockDim.z + threadIdx.z;
			const unsigned int t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight*l_idx;
			const unsigned int nPaths = nPathsWidth * nPathsHeight * nPathsDepth;

			if (t_idx < nPaths) {
				T last = heston.init1();
				T var = heston.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				T dt{};
				d_paths[t_idx] = last;

				unsigned int i = 0;
				for (unsigned int k = t_idx + nPaths; k < size*nPaths; k += nPaths) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];
					last_new = last +
						heston.drift1(d_times[i], last, var)*dt +
						heston.diffusion1(d_times[i], last, var) *
						sqrtf(dt) * z1 +
						0.5*heston.diffusion1(d_times[i], last, var) *
						((heston.diffusion1(d_times[i], last + 0.5*DIFF_STEP, var) -
							heston.diffusion1(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt)*((z1)*(z1)-1.0) +
						0.5*(heston.rho())*heston.diffusion2(d_times[i], last, var) *
						((heston.diffusion1(d_times[i], last, var + 0.5*DIFF_STEP) -
							heston.diffusion1(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*((z1)*(z1)-1.0) +
						sqrtf(1.0 - (heston.rho())*(heston.rho()))*
						heston.diffusion2(d_times[i], last, var)*
						((heston.diffusion1(d_times[i], last, var + 0.5*DIFF_STEP) -
							heston.diffusion1(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					var_new = var +
						heston.drift2(d_times[i], last, var)*dt +
						heston.diffusion2(d_times[i], last, var) *
						sqrtf(dt) *
						(heston.rho() * z1 + sqrtf(1.0 - (heston.rho() * heston.rho())) * z2) +
						0.5*(heston.rho())*heston.diffusion1(d_times[i], last, var)*
						((heston.diffusion2(d_times[i], last + 0.5*DIFF_STEP, var) -
							heston.diffusion2(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt) * ((z1)*(z1)-1.0) +
						0.5 * heston.diffusion2(d_times[i], last, var)*
						((heston.diffusion2(d_times[i], last, var + 0.5*DIFF_STEP) -
							heston.diffusion2(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP))*
							(dt) * (((heston.rho())*z1 + std::sqrt(1.0 - (heston.rho())*(heston.rho()))*z2)*
						((heston.rho())*z1 + sqrtf(1.0 - (heston.rho())*(heston.rho()))*z2) - 1.0) +
						sqrtf(1.0 - (heston.rho())*(heston.rho()))*
						heston.diffusion1(d_times[i], last, var)*
						((heston.diffusion2(d_times[i], last + 0.5*DIFF_STEP, var) -
							heston.diffusion2(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					d_paths[k] = last_new;
					last = last_new;
					var = var_new;
					i++;
				}
			}
		}

	};


};



#endif ///_HESTON_KERNELS