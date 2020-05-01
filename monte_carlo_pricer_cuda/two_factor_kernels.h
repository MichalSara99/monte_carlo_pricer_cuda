#pragma once
#if !defined(_TWO_FACTOR_KERNELS)
#define _TWO_FACTOR_KERNELS

#include<device_launch_parameters.h>
#include<curand_kernel.h>
#include"mc_types.h"


namespace kernels {

	namespace two_factor_kernels {

		namespace euler_scheme {
			// ==========================================================
			// ===== Kernels for equidistant steps 
			// ==========================================================

			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel1D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPaths, unsigned int nSteps, T dt) {
				// Path idx:
				unsigned int const t_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const total = nPaths * nSteps;

				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init1();
				T var = sdeBuilder.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};

				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					last_new = last + sdeBuilder.drift1(i*dt, last, var)*dt +
						sdeBuilder.diffusion1(i*dt, last, var)*sqrtf(dt)*z1;
					var_new = var + sdeBuilder.drift2(i*dt, last, var)*dt +
						sdeBuilder.diffusion2(i*dt, last, var)*sqrtf(dt)*
						(sdeBuilder.rho() * z1 + sqrt(1.0 - sdeBuilder.rho()*sdeBuilder.rho()) * z2);
					d_paths[t] = last_new;
					last = last_new;
					var = var_new;
				}
			}

			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel2D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nSteps, T dt) {
				// Path idx:
				unsigned int const c_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y*blockIdx.y + threadIdx.y;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth;
				unsigned int const nPaths = nPathsWidth * nPathsHeight;
				unsigned int const total = nPaths * nSteps;

				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init1();
				T var = sdeBuilder.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};

				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					last_new = last + sdeBuilder.drift1(i*dt, last, var)*dt +
						sdeBuilder.diffusion1(i*dt, last, var)*sqrtf(dt)*z1;
					var_new = var + sdeBuilder.drift2(i*dt, last, var)*dt +
						sdeBuilder.diffusion2(i*dt, last, var)*sqrtf(dt)*
						(sdeBuilder.rho() * z1 + sqrt(1.0 - sdeBuilder.rho()*sdeBuilder.rho()) * z2);
					d_paths[t] = last_new;
					last = last_new;
					var = var_new;
				}
			}

			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel3D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
					unsigned int nSteps, T dt) {
				// Path idx:
				unsigned int const c_idx = blockDim.x * blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y * blockIdx.y + threadIdx.y;
				unsigned int const l_idx = blockDim.z * blockIdx.z + threadIdx.z;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth + l_idx * nPathsWidth*nPathsHeight;
				unsigned int const nPaths = nPathsWidth * nPathsHeight * nPathsDepth;
				unsigned int const total = nPaths * nSteps;

				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init1();
				T var = sdeBuilder.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};

				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					last_new = last + sdeBuilder.drift1(i*dt, last, var)*dt +
						sdeBuilder.diffusion1(i*dt, last, var)*sqrtf(dt)*z1;
					var_new = var + sdeBuilder.drift2(i*dt, last, var)*dt +
						sdeBuilder.diffusion2(i*dt, last, var)*sqrtf(dt)*
						(sdeBuilder.rho() * z1 + sqrt(1.0 - sdeBuilder.rho()*sdeBuilder.rho()) * z2);
					d_paths[t] = last_new;
					last = last_new;
					var = var_new;
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
				// Path idx:
				unsigned int const t_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const total = nPaths * size;

				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init1();
				T var = sdeBuilder.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				T dt{};

				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];
					last_new = last + sdeBuilder.drift1(d_times[i], last, var)*dt +
						sdeBuilder.diffusion1(d_times[i], last, var)*sqrtf(dt)*z1;
					var_new = var + sdeBuilder.drift2(d_times[i], last, var)*dt +
						sdeBuilder.diffusion2(d_times[i], last, var)*sqrtf(dt)*
						(sdeBuilder.rho() * z1 + sqrt(1.0 - sdeBuilder.rho()*sdeBuilder.rho()) * z2);
					d_paths[t] = last_new;
					last = last_new;
					var = var_new;
				}
			}

			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel2D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPathsWidth, unsigned int nPathsHeight, T const *d_times, unsigned int size) {
				// Path idx:
				unsigned int const c_idx = blockDim.x*blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y*blockIdx.y + threadIdx.y;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth;
				unsigned int const nPaths = nPathsWidth * nPathsHeight;
				unsigned int const total = nPaths * size;

				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init1();
				T var = sdeBuilder.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				T dt{};

				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];
					last_new = last + sdeBuilder.drift1(d_times[i], last, var)*dt +
						sdeBuilder.diffusion1(d_times[i], last, var)*sqrtf(dt)*z1;
					var_new = var + sdeBuilder.drift2(d_times[i], last, var)*dt +
						sdeBuilder.diffusion2(d_times[i], last, var)*sqrtf(dt)*
						(sdeBuilder.rho() * z1 + sqrt(1.0 - sdeBuilder.rho()*sdeBuilder.rho()) * z2);
					d_paths[t] = last_new;
					last = last_new;
					var = var_new;
				}
			}

			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel3D(SDEBuilder sdeBuilder, T* d_paths, curandState_t *states,
					unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
					T const *d_times, unsigned int size) {
				// Path idx:
				unsigned int const c_idx = blockDim.x * blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y * blockIdx.y + threadIdx.y;
				unsigned int const l_idx = blockDim.z * blockIdx.z + threadIdx.z;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth + l_idx * nPathsWidth*nPathsHeight;
				unsigned int const nPaths = nPathsWidth * nPathsHeight * nPathsDepth;
				unsigned int const total = nPaths * size;

				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init1();
				T var = sdeBuilder.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				T dt{};

				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];
					last_new = last + sdeBuilder.drift1(d_times[i], last, var)*dt +
						sdeBuilder.diffusion1(d_times[i], last, var)*sqrtf(dt)*z1;
					var_new = var + sdeBuilder.drift2(d_times[i], last, var)*dt +
						sdeBuilder.diffusion2(d_times[i], last, var)*sqrtf(dt)*
						(sdeBuilder.rho() * z1 + sqrt(1.0 - sdeBuilder.rho()*sdeBuilder.rho()) * z2);
					d_paths[t] = last_new;
					last = last_new;
					var = var_new;
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
			void generatePathsKernel1D(SDEBuilder sdeBuilder,
					T *d_paths, curandState_t* states,
					unsigned int nPaths, unsigned int nSteps, T dt) {
				// Path index
				unsigned int const t_idx = blockIdx.x * blockDim.x + threadIdx.x;
				unsigned int const total = nPaths * nSteps;

				if (t_idx >= nPaths)
					return;


				T last = sdeBuilder.init1();
				T var = sdeBuilder.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths,++i) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);

					last_new = last +
						sdeBuilder.drift1(i*dt, last, var)*dt +
						sdeBuilder.diffusion1(i*dt, last, var) *
						sqrtf(dt) * z1 +
						0.5*sdeBuilder.diffusion1(i*dt, last, var) *
						((sdeBuilder.diffusion1(i*dt, last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion1(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt)*((z1)*(z1)-1.0) +
						0.5*(sdeBuilder.rho())*sdeBuilder.diffusion2(i*dt, last, var) *
						((sdeBuilder.diffusion1(i*dt, last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion1(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*((z1)*(z1)-1.0) +
						sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*
						sdeBuilder.diffusion2(i*dt, last, var)*
						((sdeBuilder.diffusion1(i *dt, last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion1(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					var_new = var +
						sdeBuilder.drift2(i*dt, last, var)*dt +
						sdeBuilder.diffusion2(i*dt, last, var) *
						sqrtf(dt) *
						(sdeBuilder.rho() * z1 + sqrtf(1.0 - (sdeBuilder.rho() * sdeBuilder.rho())) * z2) +
						0.5*(sdeBuilder.rho())*sdeBuilder.diffusion1(i*dt, last, var)*
						((sdeBuilder.diffusion2(i*dt, last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion2(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt) * ((z1)*(z1)-1.0) +
						0.5 * sdeBuilder.diffusion2(i*dt, last, var)*
						((sdeBuilder.diffusion2(i*dt, last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion2(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP))*
							(dt) * (((sdeBuilder.rho())*z1 + std::sqrt(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*z2)*
						((sdeBuilder.rho())*z1 + sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*z2) - 1.0) +
						sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*
						sdeBuilder.diffusion1(i*dt, last, var)*
						((sdeBuilder.diffusion2(i*dt, last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion2(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					d_paths[t] = last_new;
					last = last_new;
					var = var_new;
				}
			}
			
			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel2D(SDEBuilder sdeBuilder,
					T *d_paths, curandState_t* states,
					unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nSteps, T dt) {
				// Path index
				unsigned int const c_idx = blockDim.x * blockIdx.x + threadIdx.x;
				unsigned int const r_idx = blockDim.y * blockIdx.y + threadIdx.y;
				unsigned int const t_idx = c_idx + r_idx * nPathsWidth;
				unsigned int const nPaths = nPathsWidth * nPathsHeight;
				unsigned int const total = nPaths * nSteps;

				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init1();
				T var = sdeBuilder.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);

					last_new = last +
						sdeBuilder.drift1(i*dt, last, var)*dt +
						sdeBuilder.diffusion1(i*dt, last, var) *
						sqrtf(dt) * z1 +
						0.5*sdeBuilder.diffusion1(i*dt, last, var) *
						((sdeBuilder.diffusion1(i*dt, last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion1(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt)*((z1)*(z1)-1.0) +
						0.5*(sdeBuilder.rho())*sdeBuilder.diffusion2(i*dt, last, var) *
						((sdeBuilder.diffusion1(i*dt, last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion1(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*((z1)*(z1)-1.0) +
						sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*
						sdeBuilder.diffusion2(i*dt, last, var)*
						((sdeBuilder.diffusion1(i *dt, last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion1(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					var_new = var +
						sdeBuilder.drift2(i*dt, last, var)*dt +
						sdeBuilder.diffusion2(i*dt, last, var) *
						sqrtf(dt) *
						(sdeBuilder.rho() * z1 + sqrtf(1.0 - (sdeBuilder.rho() * sdeBuilder.rho())) * z2) +
						0.5*(sdeBuilder.rho())*sdeBuilder.diffusion1(i*dt, last, var)*
						((sdeBuilder.diffusion2(i*dt, last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion2(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt) * ((z1)*(z1)-1.0) +
						0.5 * sdeBuilder.diffusion2(i*dt, last, var)*
						((sdeBuilder.diffusion2(i*dt, last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion2(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP))*
							(dt) * (((sdeBuilder.rho())*z1 + std::sqrt(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*z2)*
						((sdeBuilder.rho())*z1 + sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*z2) - 1.0) +
						sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*
						sdeBuilder.diffusion1(i*dt, last, var)*
						((sdeBuilder.diffusion2(i*dt, last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion2(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					d_paths[t] = last_new;
					last = last_new;
					var = var_new;
				}
			}

			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel3D(SDEBuilder sdeBuilder,
					T *d_paths, curandState_t* states,
					unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
					unsigned int nSteps, T dt) {
				// Path index
				unsigned int const c_idx = blockIdx.x * blockDim.x + threadIdx.x;
				unsigned int const r_idx = blockIdx.y * blockDim.y + threadIdx.y;
				unsigned int const l_idx = blockIdx.z * blockDim.z + threadIdx.z;
				unsigned int const t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight*l_idx;
				unsigned int const nPaths = nPathsWidth * nPathsHeight * nPathsDepth;
				unsigned int const total = nPaths * nSteps;

				if (t_idx >= nPaths)
					return;

				T last = sdeBuilder.init1();
				T var = sdeBuilder.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);

					last_new = last +
						sdeBuilder.drift1(i*dt, last, var)*dt +
						sdeBuilder.diffusion1(i*dt, last, var) *
						sqrtf(dt) * z1 +
						0.5*sdeBuilder.diffusion1(i*dt, last, var) *
						((sdeBuilder.diffusion1(i*dt, last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion1(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt)*((z1)*(z1)-1.0) +
						0.5*(sdeBuilder.rho())*sdeBuilder.diffusion2(i*dt, last, var) *
						((sdeBuilder.diffusion1(i*dt, last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion1(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*((z1)*(z1)-1.0) +
						sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*
						sdeBuilder.diffusion2(i*dt, last, var)*
						((sdeBuilder.diffusion1(i *dt, last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion1(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					var_new = var +
						sdeBuilder.drift2(i*dt, last, var)*dt +
						sdeBuilder.diffusion2(i*dt, last, var) *
						sqrtf(dt) *
						(sdeBuilder.rho() * z1 + sqrtf(1.0 - (sdeBuilder.rho() * sdeBuilder.rho())) * z2) +
						0.5*(sdeBuilder.rho())*sdeBuilder.diffusion1(i*dt, last, var)*
						((sdeBuilder.diffusion2(i*dt, last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion2(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt) * ((z1)*(z1)-1.0) +
						0.5 * sdeBuilder.diffusion2(i*dt, last, var)*
						((sdeBuilder.diffusion2(i*dt, last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion2(i*dt, last, var - 0.5*DIFF_STEP)) / (DIFF_STEP))*
							(dt) * (((sdeBuilder.rho())*z1 + std::sqrt(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*z2)*
						((sdeBuilder.rho())*z1 + sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*z2) - 1.0) +
						sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*
						sdeBuilder.diffusion1(i*dt, last, var)*
						((sdeBuilder.diffusion2(i*dt, last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion2(i*dt, last - 0.5*DIFF_STEP, var)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					d_paths[t] = last_new;
					last = last_new;
					var = var_new;
				}
				
			}



			// ==========================================================
			// ===== Kernels for non-equidistant steps 
			// ==========================================================

			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel1D(SDEBuilder sdeBuilder,
					T *d_paths, curandState_t* states,
					unsigned int nPaths, T const *d_times, unsigned int size) {
				// Path index
				unsigned int const t_idx = blockIdx.x * blockDim.x + threadIdx.x;
				unsigned int const total = nPaths * size;

				if (t_idx >= nPaths)
					return;


				T last = sdeBuilder.init1();
				T var = sdeBuilder.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				T dt{};
				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];

					last_new = last +
						sdeBuilder.drift1(d_times[i], last, var)*dt +
						sdeBuilder.diffusion1(d_times[i], last, var) *
						sqrtf(dt) * z1 +
						0.5*sdeBuilder.diffusion1(d_times[i], last, var) *
						((sdeBuilder.diffusion1(d_times[i], last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion1(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt)*((z1)*(z1)-1.0) +
						0.5*(sdeBuilder.rho())*sdeBuilder.diffusion2(d_times[i], last, var) *
						((sdeBuilder.diffusion1(d_times[i], last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion1(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*((z1)*(z1)-1.0) +
						sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*
						sdeBuilder.diffusion2(d_times[i], last, var)*
						((sdeBuilder.diffusion1(d_times[i], last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion1(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					var_new = var +
						sdeBuilder.drift2(d_times[i], last, var)*dt +
						sdeBuilder.diffusion2(d_times[i], last, var) *
						sqrtf(dt) *
						(sdeBuilder.rho() * z1 + sqrtf(1.0 - (sdeBuilder.rho() * sdeBuilder.rho())) * z2) +
						0.5*(sdeBuilder.rho())*sdeBuilder.diffusion1(d_times[i], last, var)*
						((sdeBuilder.diffusion2(d_times[i], last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion2(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt) * ((z1)*(z1)-1.0) +
						0.5 * sdeBuilder.diffusion2(d_times[i], last, var)*
						((sdeBuilder.diffusion2(d_times[i], last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion2(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP))*
							(dt) * (((sdeBuilder.rho())*z1 + std::sqrt(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*z2)*
						((sdeBuilder.rho())*z1 + sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*z2) - 1.0) +
						sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*
						sdeBuilder.diffusion1(d_times[i], last, var)*
						((sdeBuilder.diffusion2(d_times[i], last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion2(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					d_paths[t] = last_new;
					last = last_new;
					var = var_new;
				}
			}

			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel2D(SDEBuilder sdeBuilder,
					T *d_paths, curandState_t* states,
					unsigned int nPathsWidth, unsigned int nPathsHeight, T const *d_times, unsigned int size) {
				// Path index
				const unsigned int c_idx = blockIdx.x * blockDim.x + threadIdx.x;
				const unsigned int r_idx = blockIdx.y * blockDim.y + threadIdx.y;
				unsigned int const t_idx = c_idx + nPathsWidth * r_idx;
				const unsigned int nPaths = nPathsWidth * nPathsHeight;
				unsigned int const total = nPaths * size;

				if (t_idx >= nPaths)
					return;


				T last = sdeBuilder.init1();
				T var = sdeBuilder.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				T dt{};
				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];

					last_new = last +
						sdeBuilder.drift1(d_times[i], last, var)*dt +
						sdeBuilder.diffusion1(d_times[i], last, var) *
						sqrtf(dt) * z1 +
						0.5*sdeBuilder.diffusion1(d_times[i], last, var) *
						((sdeBuilder.diffusion1(d_times[i], last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion1(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt)*((z1)*(z1)-1.0) +
						0.5*(sdeBuilder.rho())*sdeBuilder.diffusion2(d_times[i], last, var) *
						((sdeBuilder.diffusion1(d_times[i], last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion1(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*((z1)*(z1)-1.0) +
						sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*
						sdeBuilder.diffusion2(d_times[i], last, var)*
						((sdeBuilder.diffusion1(d_times[i], last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion1(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					var_new = var +
						sdeBuilder.drift2(d_times[i], last, var)*dt +
						sdeBuilder.diffusion2(d_times[i], last, var) *
						sqrtf(dt) *
						(sdeBuilder.rho() * z1 + sqrtf(1.0 - (sdeBuilder.rho() * sdeBuilder.rho())) * z2) +
						0.5*(sdeBuilder.rho())*sdeBuilder.diffusion1(d_times[i], last, var)*
						((sdeBuilder.diffusion2(d_times[i], last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion2(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt) * ((z1)*(z1)-1.0) +
						0.5 * sdeBuilder.diffusion2(d_times[i], last, var)*
						((sdeBuilder.diffusion2(d_times[i], last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion2(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP))*
							(dt) * (((sdeBuilder.rho())*z1 + std::sqrt(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*z2)*
						((sdeBuilder.rho())*z1 + sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*z2) - 1.0) +
						sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*
						sdeBuilder.diffusion1(d_times[i], last, var)*
						((sdeBuilder.diffusion2(d_times[i], last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion2(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					d_paths[t] = last_new;
					last = last_new;
					var = var_new;
				}
			}

			template<typename SDEBuilder,
				typename T = SDEBuilder::value_type>
				__global__
				void generatePathsKernel3D(SDEBuilder sdeBuilder,
					T *d_paths, curandState_t* states,
					unsigned int nPathsWidth, unsigned int nPathsHeight, unsigned int nPathsDepth,
					T const *d_times, unsigned int size) {
				// Path index
				unsigned int const c_idx = blockIdx.x * blockDim.x + threadIdx.x;
				unsigned int const r_idx = blockIdx.y * blockDim.y + threadIdx.y;
				unsigned int const l_idx = blockIdx.z * blockDim.z + threadIdx.z;
				unsigned int const t_idx = c_idx + nPathsWidth * r_idx + nPathsWidth * nPathsHeight*l_idx;
				unsigned int const nPaths = nPathsWidth * nPathsHeight * nPathsDepth;
				unsigned int const total = nPaths * size;

				if (t_idx >= nPaths)
					return;


				T last = sdeBuilder.init1();
				T var = sdeBuilder.init2();
				T last_new{};
				T var_new{};
				T z1{};
				T z2{};
				T dt{};
				unsigned int i{ 0 };
				d_paths[t_idx] = last;

				for (unsigned int t = t_idx + nPaths; t < total; t += nPaths, ++i) {
					z1 = curand_normal(&states[t_idx]);
					z2 = curand_normal(&states[t_idx]);
					dt = d_times[i + 1] - d_times[i];

					last_new = last +
						sdeBuilder.drift1(d_times[i], last, var)*dt +
						sdeBuilder.diffusion1(d_times[i], last, var) *
						sqrtf(dt) * z1 +
						0.5*sdeBuilder.diffusion1(d_times[i], last, var) *
						((sdeBuilder.diffusion1(d_times[i], last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion1(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt)*((z1)*(z1)-1.0) +
						0.5*(sdeBuilder.rho())*sdeBuilder.diffusion2(d_times[i], last, var) *
						((sdeBuilder.diffusion1(d_times[i], last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion1(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*((z1)*(z1)-1.0) +
						sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*
						sdeBuilder.diffusion2(d_times[i], last, var)*
						((sdeBuilder.diffusion1(d_times[i], last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion1(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					var_new = var +
						sdeBuilder.drift2(d_times[i], last, var)*dt +
						sdeBuilder.diffusion2(d_times[i], last, var) *
						sqrtf(dt) *
						(sdeBuilder.rho() * z1 + sqrtf(1.0 - (sdeBuilder.rho() * sdeBuilder.rho())) * z2) +
						0.5*(sdeBuilder.rho())*sdeBuilder.diffusion1(d_times[i], last, var)*
						((sdeBuilder.diffusion2(d_times[i], last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion2(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP))*
							(dt) * ((z1)*(z1)-1.0) +
						0.5 * sdeBuilder.diffusion2(d_times[i], last, var)*
						((sdeBuilder.diffusion2(d_times[i], last, var + 0.5*DIFF_STEP) -
							sdeBuilder.diffusion2(d_times[i], last, var - 0.5*DIFF_STEP)) / (DIFF_STEP))*
							(dt) * (((sdeBuilder.rho())*z1 + std::sqrt(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*z2)*
						((sdeBuilder.rho())*z1 + sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*z2) - 1.0) +
						sqrtf(1.0 - (sdeBuilder.rho())*(sdeBuilder.rho()))*
						sdeBuilder.diffusion1(d_times[i], last, var)*
						((sdeBuilder.diffusion2(d_times[i], last + 0.5*DIFF_STEP, var) -
							sdeBuilder.diffusion2(d_times[i], last - 0.5*DIFF_STEP, var)) / (DIFF_STEP)) *
							(dt)*z1*z2;

					d_paths[t] = last_new;
					last = last_new;
					var = var_new;
				}
			}

		}


	}




}






#endif ///_TWO_FACTOR_KERNELS