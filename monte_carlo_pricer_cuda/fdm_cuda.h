#pragma once
#if !defined(_FDM_CUDA)
#define _FDM_CUDA

#include"sde_builder_cuda.h"
#include"mc_types.h"

namespace fdm_cuda {

	using mc_types::FDMScheme;
	using mc_types::GPUConfiguration;
	using mc_types::PathValuesType;

	template<std::size_t FactorCount,
		typename T>
		class FdmCUDA;


	template<typename T>
	class FdmCUDA<1, T> {
	private:
		T sdeModel_;
		T terminationTime_;
		std::size_t numberSteps_;

	public:
		explicit FdmCUDA(T terminationTime, std::size_t numberSteps = 360)
			:terminationTime_{ terminationTime },
			numberSteps_{ numberSteps }
		{}

		template<typename SDECudaModel,
			typename = typename std::enable_if<SDECudaModel::FactorCount == 1>::type>
			PathValuesType<PathValuesType<T>>
			operator()(SDECudaModel model, std::size_t iterations,
				FDMScheme scheme = FDMScheme::EulerScheme,
				GPUConfiguration config = GPUConfiguration::Grid1D) const {
			T dt = terminationTime_ / static_cast<T>(numberSteps_);
			auto engine = model.engineConstructor();
			return engine.simulate(iterations, this->numberSteps_, dt, scheme, config);
		}
	};


	template<typename T>
	class FdmCUDA<2, T> {
	private:
		T sdeModel_;
		T terminationTime_;
		std::size_t numberSteps_;

	public:
		explicit FdmCUDA(T terminationTime, std::size_t numberSteps = 360)
			:terminationTime_{ terminationTime },
			numberSteps_{ numberSteps }
		{}

		template<typename SDECudaModel,
			typename = typename std::enable_if<SDECudaModel::FactorCount == 2>::type>
			PathValuesType<PathValuesType<T>>
			operator()(SDECudaModel model, std::size_t iterations,
				FDMScheme scheme = FDMScheme::EulerScheme,
				GPUConfiguration config = GPUConfiguration::Grid1D) const {
			T dt = terminationTime_ / static_cast<T>(numberSteps_);
			auto engine = model.engineConstructor();
			return engine.simulate(iterations, this->numberSteps_, dt, scheme, config);
		}
	};



}



#endif ///_FDM_CUDA