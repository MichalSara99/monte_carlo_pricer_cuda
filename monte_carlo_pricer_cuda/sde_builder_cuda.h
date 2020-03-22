#pragma once
#if !defined(_SDE_BUILDER_CUDA)
#define _SDE_BUILDER_CUDA

#include<type_traits>
#include<cuda_runtime.h>
#include<curand_kernel.h>
#include"mc_types.h"


// Forward declaration:
namespace fdm_engine_cuda {
	class GBMPathEngineDouble;
	class GBMPathEngineFloat;
	class ABMPathEngineDouble;
	class ABMPathEngineFloat;
	class CEVPathEngineDouble;
	class CEVPathEngineFloat;
	class HestonPathEngineDouble;
	class HestonPathEngineFloat;
}

namespace sde_builder_cuda {

	using fdm_engine_cuda::GBMPathEngineDouble;
	using fdm_engine_cuda::GBMPathEngineFloat;
	using fdm_engine_cuda::ABMPathEngineDouble;
	using fdm_engine_cuda::ABMPathEngineFloat;
	using fdm_engine_cuda::CEVPathEngineDouble;
	using fdm_engine_cuda::CEVPathEngineFloat;
	using fdm_engine_cuda::HestonPathEngineDouble;
	using fdm_engine_cuda::HestonPathEngineFloat;

	// Geometric Brownian Motion
	template<typename T = double>
	class GeometricBrownianMotion {
	private:
		T mu_;
		T sigma_;
		T init_;

		GBMPathEngineDouble _engineConstructor(std::true_type);
		GBMPathEngineFloat _engineConstructor(std::false_type);

	public:
		typedef T value_type;
		explicit GeometricBrownianMotion(T mu, T sigma, T init)
			:mu_{ mu }, sigma_{ sigma }, init_{ init } {}

		GeometricBrownianMotion(GeometricBrownianMotion<T> const &other)
			:mu_{ other.mu_ }, sigma_{ other.sigma_ }, init_{ other.init_ }
		{}

		GeometricBrownianMotion& operator=(GeometricBrownianMotion<T> const &other)
		{
			if (this != &other) {
				mu_ = other.mu_;
				sigma_ = other.sigma_;
				init_ = other.init_;
			}
			return *this;
		}

		enum { FactorCount = 1 };

		auto engineConstructor() {
			return _engineConstructor(std::is_same<T, double>());
		}

		__host__ __device__
			inline T mu()const { return this->mu_; }
		__host__ __device__
			inline T sigma()const { return this->sigma_; }
		__host__ __device__
			inline T init()const { return this->init_; }

		__host__ __device__
			inline T drift(T time, T underlyingPrice)const {
			return mu_ * underlyingPrice;
		}
		__host__ __device__
			inline T diffusion(T time, T underlyingPrice)const {
			return sigma_ * underlyingPrice;
		}
	};


	// Arithmetic Brownian Motion
	template<typename T = double>
	class ArithmeticBrownianMotion {
	private:
		T mu_;
		T sigma_;
		T init_;

		ABMPathEngineDouble _engineConstructor(std::true_type);
		ABMPathEngineFloat _engineConstructor(std::false_type);

	public:
		typedef T value_type;
		ArithmeticBrownianMotion(T mu, T sigma, T initialCondition)
			:mu_{ mu }, sigma_{ sigma }, init_{ initialCondition } {}
		ArithmeticBrownianMotion()
			:ArithmeticBrownianMotion{ 0.0,1.0,1.0 } {}

		ArithmeticBrownianMotion(ArithmeticBrownianMotion<T> const &copy)
			:mu_{ copy.mu_ }, sigma_{ copy.sigma_ }, init_{ copy.init_ } {}

		ArithmeticBrownianMotion& operator=(ArithmeticBrownianMotion<T> const &copy) {
			if (this != &copy) {
				mu_ = copy.mu_;
				sigma_ = copy.sigma_;
				init_ = copy.init_;
			}
			return *this;
		}

		enum { FactorCount = 1 };

		auto engineConstructor() {
			return _engineConstructor(std::is_same<T, double>());
		}


		__host__ __device__
			inline T mu()const { return this->mu_; }
		__host__ __device__
			inline T sigma()const { return this->sigma_; }
		__host__ __device__
			inline T init()const { return this->init_; }

		__host__ __device__
			inline T drift(T time, T underlyingPrice)const {
			return this->mu_;
		}
		__host__ __device__
			inline T diffusion(T time, T underlyingPrice)const {
			return this->sigma_;
		}
	};


	template<typename T = double>
	class ConstantElasticityVariance {
	private:
		T beta_;
		T mu_;
		T sigma_;
		T init_;

		CEVPathEngineDouble _engineConstructor(std::true_type);
		CEVPathEngineFloat _engineConstructor(std::false_type);

	public:
		typedef T value_type;
		ConstantElasticityVariance(T mu, T sigma, T beta, T initialCondition)
			:mu_{ mu }, sigma_{ sigma }, beta_{ beta }, init_{
			initialCondition
		} {}
		ConstantElasticityVariance()
			:ConstantElasticityVariance{ 0.0,1.0,0.5,1.0 } {}

		ConstantElasticityVariance(ConstantElasticityVariance<T> const &copy)
			:mu_{ copy.mu_ }, sigma_{ copy.sigma_ }, beta_{ copy.beta_ },
			init_{ copy.init_ } {}

		ConstantElasticityVariance& operator=(ConstantElasticityVariance<T> const &copy) {
			if (this != &copy) {
				mu_ = copy.mu_;
				sigma_ = copy.sigma_;
				beta_ = copy.beta_;
				init_ = copy.init_;
			}
			return *this;
		}

		enum { FactorCount = 1 };

		auto engineConstructor() {
			return _engineConstructor(std::is_same<T, double>());
		}

		__host__ __device__
			inline T mu()const { return mu_; }
		__host__ __device__
			inline T sigma()const { return sigma_; }
		__host__ __device__
			inline T init()const { return init_; }
		__host__ __device__
			inline T beta()const { return beta_; }

		__host__ __device__
			inline T drift(T time, T underlyingPrice)const {
			return mu_ * underlyingPrice;
		}
		__host__ __device__
			inline T diffusion(T time, T underlyingPrice)const {
			return sigma_ * pow(underlyingPrice, beta_);
		}
	};


	template<typename T>
	class HestonModel {
	private:
		// for first underlying factor:
		T init1_;
		T mu_;
		T sigma_;
		// for second variance factor:
		T init2_;
		T kappa_;
		T theta_;
		T etha_;
		// correlation between factors:
		T rho_;

		HestonPathEngineDouble _engineConstructor(std::true_type);
		HestonPathEngineFloat _engineConstructor(std::false_type);

	public:
		typedef T value_type;
		explicit HestonModel(T mu, T sigma, T kappa, T theta,
			T init1, T init2, T rho = 0.0)
			:mu_{ mu }, sigma_{ sigma }, kappa_{ kappa },
			theta_{ theta }, init2_{ init2 }, init1_{ init1 }, rho_{ rho } {}

		HestonModel(HestonModel<T> const &other)
			:mu_{ other.mu_ }, init1_{ other.init1_ }, sigma_{ other.sigma_ },
			init2_{ other.init2_ }, kappa_{ other.kappa_ }, theta_{ other.theta_ },
			etha_{ other.theta_ }, rho_{ other.rho_ }
		{}

		HestonModel& operator=(HestonModel<T> const &other) {
			if (this != &other) {
				init1_ = other.init1_;
				mu_ = other.mu_;
				sigma_ = other.sigma_;
				init2_ = other.init2_;
				kappa_ = other.kappa_;
				theta_ = other.theta_;
				etha_ = other.etha_;
				rho_ = other.rho_;
			}
			return *this;
		}

		enum { FactorCount = 2 };

		auto engineConstructor() {
			return _engineConstructor(std::is_same<T, double>());
		}

		__host__ __device__
			inline T init1()const { return this->init1_; }
		__host__ __device__
			inline T init2()const { return this->init2_; }
		__host__ __device__
			inline T rho()const { return this->rho_; }
		__host__ __device__
			inline T mu()const { return this->mu_; }
		__host__ __device__
			inline T sigma()const { return this->sigma_; }
		__host__ __device__
			inline T kappa()const { return this->kappa_; }
		__host__ __device__
			inline T theta()const { return this->theta_; }
		__host__ __device__
			inline T etha()const { return this->etha_; }

		__host__ __device__
			inline T drift1(T time, T underlyingPrice, T variance)const {
			return mu_ * underlyingPrice;
		}
		__host__ __device__
			inline T diffusion1(T time, T underlyingPrice, T variance)const {
			return sigma_ * underlyingPrice * sqrt(variance);
		}
		__host__ __device__
			inline T drift2(T time, T underlyingPrice, T variance)const {
			return kappa_ * (theta_ - variance);
		}
		__host__ __device__
			inline T diffusion2(T time, T underlyingPrice, T variance)const {
			return etha_ * sqrt(variance);
		}

	};
}





namespace fdm_engine_cuda {

	using mc_types::PathValuesType;
	using mc_types::GPUConfiguration;
	using mc_types::FDMScheme;

	// ========================================================================================== //
	// ====== FactorCount: 1
	// ====== GBMPathEngine (Geometric Brownian Motion)
	// ========================================================================================== //

	class GBMPathEngineDouble {
	private:
		sde_builder_cuda::GeometricBrownianMotion<double> gbm_;
		void _generate1D(double *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, double dt)const;
		void _generate2D(double *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, double dt)const;
		void _generate3D(double *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, double dt)const;

	public:
		typedef double value_type;
		explicit GBMPathEngineDouble(sde_builder_cuda::GeometricBrownianMotion<double> gbm)
			:gbm_{ gbm } {}

		PathValuesType<PathValuesType<double>> simulate(unsigned int nPaths,
			unsigned int nSteps, double dt, FDMScheme scheme,
			GPUConfiguration config = GPUConfiguration::Grid1D)const;
	};


	class GBMPathEngineFloat {
	private:
		sde_builder_cuda::GeometricBrownianMotion<float> gbm_;
		void _generate1D(float *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, float dt)const;
		void _generate2D(float *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, float dt)const;
		void _generate3D(float *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, float dt)const;

	public:
		typedef double value_type;
		explicit GBMPathEngineFloat(sde_builder_cuda::GeometricBrownianMotion<float> gbm)
			:gbm_{ gbm } {}

		PathValuesType<PathValuesType<float>> simulate(unsigned int nPaths,
			unsigned int nSteps, float dt, FDMScheme scheme,
			GPUConfiguration config = GPUConfiguration::Grid1D)const;

	};
	// ========================================================================================== //
	// ====== FactorCount: 1
	// ====== ABMPathEngine (Arithemtic Brownian Motion)
	// ========================================================================================== //

	class ABMPathEngineDouble {
	private:
		sde_builder_cuda::ArithmeticBrownianMotion<double> abm_;
		void _generate1D(double *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, double dt)const;
		void _generate2D(double *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, double dt)const;
		void _generate3D(double *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, double dt)const;

	public:
		explicit ABMPathEngineDouble(sde_builder_cuda::ArithmeticBrownianMotion<double> abm)
			:abm_{ abm } {}

		PathValuesType<PathValuesType<double>> simulate(unsigned int nPaths,
			unsigned int nSteps, double dt, FDMScheme scheme,
			GPUConfiguration config = GPUConfiguration::Grid1D)const;
	};

	class ABMPathEngineFloat {
	private:
		sde_builder_cuda::ArithmeticBrownianMotion<float> abm_;
		void _generate1D(float *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, float dt)const;
		void _generate2D(float *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, float dt)const;
		void _generate3D(float *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, float dt)const;

	public:
		explicit ABMPathEngineFloat(sde_builder_cuda::ArithmeticBrownianMotion<float> abm)
			:abm_{ abm } {}

		PathValuesType<PathValuesType<float>> simulate(unsigned int nPaths,
			unsigned int nSteps, float dt, FDMScheme scheme,
			GPUConfiguration config = GPUConfiguration::Grid1D)const;

	};

	// ========================================================================================== //
	// ====== FactorCount: 1
	// ====== CEVPathEngine (Constant Elasticity Variance)
	// ========================================================================================== //

	class CEVPathEngineDouble {
	private:
		sde_builder_cuda::ConstantElasticityVariance<double> cev_;
		void _generate1D(double *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, double dt)const;
		void _generate2D(double *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, double dt)const;
		void _generate3D(double *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, double dt)const;

	public:
		explicit CEVPathEngineDouble(sde_builder_cuda::ConstantElasticityVariance<double> cev)
			:cev_{ cev } {}

		PathValuesType<PathValuesType<double>> simulate(unsigned int nPaths,
			unsigned int nSteps, double dt, FDMScheme scheme,
			GPUConfiguration config = GPUConfiguration::Grid1D)const;
	};

	class CEVPathEngineFloat {
	private:
		sde_builder_cuda::ConstantElasticityVariance<float> cev_;
		void _generate1D(float *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, float dt)const;
		void _generate2D(float *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, float dt)const;
		void _generate3D(float *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, float dt)const;

	public:
		explicit CEVPathEngineFloat(sde_builder_cuda::ConstantElasticityVariance<float> cev)
			:cev_{ cev } {}

		PathValuesType<PathValuesType<float>> simulate(unsigned int nPaths,
			unsigned int nSteps, float dt, FDMScheme scheme,
			GPUConfiguration config = GPUConfiguration::Grid1D)const;

	};

	// ========================================================================================== //
	// ====== FactorCount: 2
	// ====== HestonPathEngine
	// ========================================================================================== //

	class HestonPathEngineDouble {
	private:
		sde_builder_cuda::HestonModel<double> heston_;
		void _generate1D(double *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, double dt)const;
		void _generate2D(double *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, double dt)const;
		void _generate3D(double *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, double dt)const;

	public:
		explicit HestonPathEngineDouble(sde_builder_cuda::HestonModel<double> heston)
			:heston_{ heston } {}

		PathValuesType<PathValuesType<double>> simulate(unsigned int nPaths,
			unsigned int nSteps, double dt, FDMScheme scheme,
			GPUConfiguration config = GPUConfiguration::Grid1D)const;
	};

	class HestonPathEngineFloat {
	private:
		sde_builder_cuda::HestonModel<float> heston_;
		void _generate1D(float *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, float dt)const;
		void _generate2D(float *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, float dt)const;
		void _generate3D(float *d_paths, curandState_t *states, FDMScheme scheme,
			unsigned int nPaths, unsigned int nSteps, float dt)const;

	public:
		explicit HestonPathEngineFloat(sde_builder_cuda::HestonModel<float> heston)
			:heston_{ heston } {}

		PathValuesType<PathValuesType<float>> simulate(unsigned int nPaths,
			unsigned int nSteps, float dt, FDMScheme scheme,
			GPUConfiguration config = GPUConfiguration::Grid1D)const;

	};

	// ========================================================================================== //
	// ========================================================================================== //
}

// implementations:
template<typename T>
fdm_engine_cuda::GBMPathEngineDouble sde_builder_cuda::GeometricBrownianMotion<T>::_engineConstructor(std::true_type) {
	return fdm_engine_cuda::GBMPathEngineDouble(*this);
}

template<typename T>
fdm_engine_cuda::GBMPathEngineFloat sde_builder_cuda::GeometricBrownianMotion<T>::_engineConstructor(std::false_type) {
	return fdm_engine_cuda::GBMPathEngineFloat(*this);
}

template<typename T>
fdm_engine_cuda::ABMPathEngineDouble sde_builder_cuda::ArithmeticBrownianMotion<T>::_engineConstructor(std::true_type) {
	return fdm_engine_cuda::ABMPathEngineDouble(*this);
}

template<typename T>
fdm_engine_cuda::ABMPathEngineFloat sde_builder_cuda::ArithmeticBrownianMotion<T>::_engineConstructor(std::false_type) {
	return fdm_engine_cuda::ABMPathEngineFloat(*this);
}

template<typename T>
fdm_engine_cuda::CEVPathEngineDouble sde_builder_cuda::ConstantElasticityVariance<T>::_engineConstructor(std::true_type) {
	return fdm_engine_cuda::CEVPathEngineDouble(*this);
}

template<typename T>
fdm_engine_cuda::CEVPathEngineFloat sde_builder_cuda::ConstantElasticityVariance<T>::_engineConstructor(std::false_type) {
	return fdm_engine_cuda::CEVPathEngineFloat(*this);
}

template<typename T>
fdm_engine_cuda::HestonPathEngineDouble sde_builder_cuda::HestonModel<T>::_engineConstructor(std::true_type) {
	return fdm_engine_cuda::HestonPathEngineDouble(*this);
}

template<typename T>
fdm_engine_cuda::HestonPathEngineFloat sde_builder_cuda::HestonModel<T>::_engineConstructor(std::false_type) {
	return fdm_engine_cuda::HestonPathEngineFloat(*this);
}

#endif ///_SDE_BUILDER_CUDA