#pragma once
#if !defined(_FDM_CUDA)
#define _FDM_CUDA

#include"sde_builder_cuda.h"
#include"mc_types.h"
#include"path_collector.h"

namespace finite_difference_method_cuda {

	using mc_types::FDMScheme;
	using mc_types::GPUConfiguration;
	using mc_types::PathValuesType;
	using mc_types::TimePointsType;

	template<std::size_t FactorCount,
		typename T>
		class FdmCUDA;

	// Partial specialization for one-factor models:
	template<typename T>
	class FdmCUDA<1, T> {
	private:
		T sdeModel_;
		T terminationTime_;
		std::size_t numberSteps_;
		const TimePointsType<T> timePoints_;
		bool timePointsOn_{ false };

	public:
		explicit FdmCUDA(T terminationTime, std::size_t numberSteps = 360)
			:terminationTime_{ terminationTime },
			numberSteps_{ numberSteps }
		{}

		explicit FdmCUDA(TimePointsType<T> const &timePoints)
			:timePoints_{ timePoints },
			timePointsOn_{ true }
		{}

		template<typename SDECudaModel,
				typename = typename std::enable_if<SDECudaModel::FactorCount == 1>::type>
		std::shared_ptr<path_collector::PathCollector<1, T>> const
			operator()(SDECudaModel model, std::size_t iterations,
				FDMScheme scheme = FDMScheme::EulerScheme,
				GPUConfiguration config = GPUConfiguration::Grid1D) const {
			auto engine = model.engineConstructor();
			if (timePointsOn_ == false) {
				T dt = terminationTime_ / static_cast<T>(numberSteps_);
				return engine.simulate(iterations, this->numberSteps_, dt, scheme, config);
			}
			return engine.simulate(iterations, this->timePoints_, scheme, config);
		}
	};

	// Partial specialization for two-factor models:
	template<typename T>
	class FdmCUDA<2, T> {
	private:
		T sdeModel_;
		T terminationTime_;
		std::size_t numberSteps_;
		const TimePointsType<T> timePoints_;
		bool timePointsOn_{ false };

	public:
		explicit FdmCUDA(T terminationTime, std::size_t numberSteps = 360)
			:terminationTime_{ terminationTime },
			numberSteps_{ numberSteps }
		{}

		explicit FdmCUDA(TimePointsType<T> const &timePoints)
			:timePoints_{ timePoints },
			timePointsOn_{ true }
		{}

		template<typename SDECudaModel,
			typename = typename std::enable_if<SDECudaModel::FactorCount == 2>::type>
		std::shared_ptr<path_collector::PathCollector<2, T>> const
			operator()(SDECudaModel model, std::size_t iterations,
				FDMScheme scheme = FDMScheme::EulerScheme,
				GPUConfiguration config = GPUConfiguration::Grid1D) const {
			auto engine = model.engineConstructor();
			if (timePointsOn_ == false) {
				T dt = terminationTime_ / static_cast<T>(numberSteps_);
				return engine.simulate(iterations, this->numberSteps_, dt, scheme, config);
			}
			return engine.simulate(iterations, this->timePoints_, scheme, config);
		}
	};



}



#endif ///_FDM_CUDA