#pragma once
#if !defined(_FDM_SCHEME_H_)
#define _FDM_SCHEME_H_

#include"mc_types.h"
#include"mc_utilities.h"
#include"sde.h"
#include<random>
#include<cassert>

namespace finite_difference_method {


	using mc_types::TimePointsType;
	using mc_utilities::PartialCentralDifference;
	using mc_utilities::withRespectTo;
	using sde::Sde;


	template<typename T, typename ...Ts>
	using asyncKernel1 = std::function<PathValuesType<T>(Ts...)>;

	template<typename T, typename ...Ts>
	using asyncKernel2 = std::function<std::pair<PathValuesType<T>, PathValuesType<T>>(Ts...)>;



	template<std::size_t FactorCount, typename T, typename ...Ts>
	class SchemeBuilder {
	public:
		virtual ~SchemeBuilder() {}
	};

	// Scheme builder for one-factor models
	template<typename T, typename ...Ts>
	class SchemeBuilder<1, T, Ts...> {
	protected:
		std::size_t numberSteps_;
		T delta_;
		std::shared_ptr<Sde<T, Ts...>> model_;

	public:
		SchemeBuilder(std::shared_ptr<Sde<T, Ts...>> const &model,
			T const &delta, std::size_t numberSteps)
			:model_{ model }, delta_{ delta },
			numberSteps_{ numberSteps } {}
		SchemeBuilder(std::shared_ptr<Sde<T, Ts...>> const &model) :
			model_{ model } {}


		virtual PathValuesType<T> simulate(std::random_device::result_type seed) = 0;
		virtual PathValuesType<T> simulateWithTimePoints(std::random_device::result_type seed,
			TimePointsType<T> &timePoints) = 0;
	};

	// Scheme builder for two-factor models:
	template<typename T, typename ...Ts>
	class SchemeBuilder<2, T, Ts...> {
	protected:
		std::size_t numberSteps_;
		T delta_;
		T correlation_;
		std::tuple<std::shared_ptr<Sde<T, Ts...>>, std::shared_ptr<Sde<T, Ts...>>> model_;

	public:
		SchemeBuilder(std::tuple<std::shared_ptr<Sde<T, Ts...>>, std::shared_ptr<Sde<T, Ts...>>> const &model,
			T correlation, T const &delta, std::size_t numberSteps)
			:model_{ model }, delta_{ delta }, correlation_{ correlation },
			numberSteps_{ numberSteps } {}

		SchemeBuilder(std::tuple<std::shared_ptr<Sde<T, Ts...>>, std::shared_ptr<Sde<T, Ts...>>> const &model,
			T correlation)
			:model_{ model }, correlation_{ correlation } {}

		virtual std::pair<PathValuesType<T>, PathValuesType<T>> simulate(std::random_device::result_type seed) = 0;
		virtual std::pair<PathValuesType<T>, PathValuesType<T>> simulateWithTimePoints(std::random_device::result_type seed,
			TimePointsType<T> &timePoints) = 0;

	};


	template<std::size_t FactorCount,
		typename T,
		typename = typename std::enable_if<(std::is_arithmetic<T>::value) &&
		(FactorCount > 0)>::type>
		class EulerScheme {};


	template<typename T>
	class EulerScheme<1, T> :public SchemeBuilder<1, T, T, T> {
	private:
		std::normal_distribution<T> normal_;
		std::mt19937 mt_;

	public:
		EulerScheme(std::shared_ptr<Sde<T, T, T>> const &model,
			T const &delta, std::size_t numberSteps)
			:SchemeBuilder<1, T, T, T>{ model,delta,numberSteps } {
		}

		EulerScheme(std::shared_ptr<Sde<T, T, T>> const &model) :
			SchemeBuilder<1, T, T, T>{ model } {}


		PathValuesType<T> simulate(std::random_device::result_type seed) override {
			this->mt_.seed(seed);
			PathValuesType<T> path(this->numberSteps_);
			path[0] = this->model_->initCondition();
			auto spot = this->model_->initCondition();
			T spotNew{};
			T const dt{ this->delta_ };
			for (std::size_t i = 1; i < path.size(); ++i) {
				spotNew = spot +
					this->model_->drift((i - 1)*dt, spot)*dt +
					this->model_->diffusion((i - 1)*dt, spot) *
					std::sqrt(dt) * normal_(mt_);
				path[i] = spotNew;
				spot = spotNew;
			}
			return path;
		}

		PathValuesType<T> simulateWithTimePoints(std::random_device::result_type seed, TimePointsType<T> &timePoints)override {
			this->mt_.seed(seed);
			assert(!timePoints.empty());
			PathValuesType<T> path(timePoints.size());
			path[0] = this->model_->initCondition();
			auto spot = this->model_->initCondition();
			T spotNew{};
			T dt{};
			T tp{};
			for (std::size_t i = 1; i < path.size(); ++i) {
				dt = timePoints[i] - timePoints[i - 1];
				tp = timePoints[i - 1];
				spotNew = spot +
					this->model_->drift(tp, spot)*dt +
					this->model_->diffusion(tp, spot) *
					std::sqrt(dt) * normal_(mt_);
				path[i] = spotNew;
				spot = spotNew;
			}
			return path;
		}

	};

	template<typename T>
	class EulerScheme<2, T> :public SchemeBuilder<2, T, T, T, T> {
	private:
		std::normal_distribution<T> normal1_;
		std::normal_distribution<T> normal2_;
		std::mt19937 mt_;

	public:
		EulerScheme(std::tuple<std::shared_ptr<Sde<T, T, T, T>>, std::shared_ptr<Sde<T, T, T, T>>> const &model,
			T correlation, T const &delta, std::size_t numberSteps)
			:SchemeBuilder<2, T, T, T, T>{ model,correlation,delta,numberSteps } {}

		EulerScheme(std::tuple<std::shared_ptr<Sde<T, T, T, T>>, std::shared_ptr<Sde<T, T, T, T>>> const &model,
			T correlation) :
			SchemeBuilder<2, T, T, T, T>{ model,correlation } {}

		std::pair<PathValuesType<T>, PathValuesType<T>> simulate(std::random_device::result_type seed)override {
			this->mt_.seed(seed);
			T z1{};
			T z2{};
			PathValuesType<T> path0(this->numberSteps_);
			PathValuesType<T> path1(this->numberSteps_);
			auto firstModel = std::get<0>(this->model_);
			auto secondModel = std::get<1>(this->model_);
			path0[0] = firstModel->initCondition();
			path1[0] = secondModel->initCondition();
			auto firstSpot = firstModel->initCondition();
			T firstSpotNew{};
			auto secondSpot = secondModel->initCondition();
			T secondSpotNew{};
			T const dt{ this->delta_ };

			for (std::size_t i = 1; i < path0.size(); ++i) {
				z1 = normal1_(mt_);
				z2 = normal2_(mt_);
				firstSpotNew = firstSpot +
					firstModel->drift((i - 1)*dt, firstSpot, secondSpot)*dt +
					firstModel->diffusion((i - 1)*dt, firstSpot, secondSpot) *
					std::sqrt(dt) * z1;
				secondSpotNew = secondSpot +
					secondModel->drift((i - 1)*dt, firstSpot, secondSpot)*dt +
					secondModel->diffusion((i - 1)*dt, firstSpot, secondSpot) *
					std::sqrt(dt) *
					(this->correlation_ * z1 + std::sqrt(1.0 - (this->correlation_ * this->correlation_)) * z2);
				path0[i] = firstSpotNew;
				path1[i] = secondSpotNew;
				firstSpot = firstSpotNew;
				secondSpot = secondSpotNew;
			}
			return std::make_pair(path0,path1);
		}

		std::pair<PathValuesType<T>, PathValuesType<T>> simulateWithTimePoints(std::random_device::result_type seed, TimePointsType<T> &timePoints)override {
			this->mt_.seed(seed);
			T z1{};
			T z2{};
			assert(!timePoints.empty());
			PathValuesType<T> path0(timePoints.size());
			PathValuesType<T> path1(timePoints.size());
			auto firstModel = std::get<0>(this->model_);
			auto secondModel = std::get<1>(this->model_);
			path0[0] = firstModel->initCondition();
			path1[0] = secondModel->initCondition();
			auto firstSpot = firstModel->initCondition();
			T firstSpotNew{};
			auto secondSpot = secondModel->initCondition();
			T secondSpotNew{};
			T dt{};
			T tp{};

			for (std::size_t i = 1; i < path0.size(); ++i) {
				z1 = normal1_(mt_);
				z2 = normal2_(mt_);
				dt = timePoints[i] - timePoints[i - 1];
				tp = timePoints[i - 1];
				firstSpotNew = firstSpot +
					firstModel->drift(tp, firstSpot, secondSpot)*dt +
					firstModel->diffusion(tp, firstSpot, secondSpot) *
					std::sqrt(dt) * z1;
				secondSpotNew = secondSpot +
					secondModel->drift(tp, firstSpot, secondSpot)*dt +
					secondModel->diffusion(tp, firstSpot, secondSpot) *
					std::sqrt(dt) *
					(this->correlation_ * z1 + std::sqrt(1.0 - (this->correlation_*this->correlation_)) * z2);
				path0[i] = firstSpotNew;
				path1[i] = secondSpotNew;
				firstSpot = firstSpotNew;
				secondSpot = secondSpotNew;
			}
			return std::make_pair(path0,path1);
		}

	};


	template<std::size_t FactorCount,
		typename T,
		typename = typename std::enable_if<(std::is_arithmetic<T>::value) &&
		(FactorCount > 0)>::type>
		class MilsteinScheme {};


	template<typename T>
	class MilsteinScheme<1, T> :public SchemeBuilder<1, T, T, T> {
	private:
		std::normal_distribution<T> normal_;
		std::mt19937 mt_;
		T step_ = 10e-6;

	public:
		MilsteinScheme(std::shared_ptr<Sde<T, T, T>> const &model,
			T const &delta, std::size_t numberSteps)
			:SchemeBuilder<1, T, T, T>{ model,delta,numberSteps } {}

		MilsteinScheme(std::shared_ptr<Sde<T, T, T>> const &model) :
			SchemeBuilder<1, T, T, T>{ model } {};

		inline T diffusionPrime(T time, T price) {
			auto fun = pcd_(std::bind(&Sde<T, T, T>::diffusion, *(this->model_), std::placeholders::_1, std::placeholders::_2),
				withRespectTo::secondArg);
			return fun(time, price);
		}

		PathValuesType<T> simulate(std::random_device::result_type seed) override {
			this->mt_.seed(seed);
			PathValuesType<T> path(this->numberSteps_);
			path[0] = this->model_->initCondition();
			auto spot = this->model_->initCondition();
			T spotNew{};
			T z{};
			T const dt{ this->delta_ };
			T const stp{ this->step_ };

			for (std::size_t i = 1; i < path.size(); ++i) {
				z = normal_(mt_);
				spotNew = spot +
					this->model_->drift((i - 1)*dt, spot)*dt +
					this->model_->diffusion((i - 1)*dt, spot) *
					std::sqrt(dt) * z +
					0.5*this->model_->diffusion((i - 1) * dt, spot) *
					((this->model_->diffusion((i - 1) * dt, spot + 0.5*stp) -
						this->model_->diffusion((i - 1) * dt, spot - 0.5*stp)) / stp)*
						((std::sqrt(dt)*z)*(std::sqrt(dt)*z) - dt);
				path[i] = spotNew;
				spot = spotNew;
			}
			return path;
		}

		PathValuesType<T> simulateWithTimePoints(std::random_device::result_type seed, TimePointsType<T> &timePoints)override {
			this->mt_.seed(seed);
			assert(!timePoints.empty());
			PathValuesType<T> path(timePoints.size());
			path[0] = this->model_->initCondition();
			auto spot = this->model_->initCondition();
			T spotNew{};
			T z{};
			T dt{};
			T tp{};
			T sqrtdt{};
			for (std::size_t i = 1; i < path.size(); ++i) {
				z = normal_(mt_);
				dt = timePoints[i] - timePoints[i - 1];
				tp = timePoints[i - 1];
				sqrtdt = std::sqrt(dt);
				spotNew = spot +
					this->model_->drift(tp, spot)*dt +
					this->model_->diffusion(tp, spot)*sqrtdt * z +
					0.5*this->model_->diffusion(tp, spot)*
					((this->model_->diffusion(tp, spot + 0.5*(this->step_)) -
						this->model_->diffusion(tp, spot - 0.5*(this->step_))) / (this->step_))*
						((sqrtdt*z)*(sqrtdt*z) -(dt));
				path[i] = spotNew;
				spot = spotNew;
			}
			return path;
		}
	};


	template<typename T>
	class MilsteinScheme<2, T> :public SchemeBuilder<2, T, T, T, T> {
	private:
		std::normal_distribution<T> normal1_;
		std::normal_distribution<T> normal2_;
		std::mt19937 mt_;
		T step_ = 10e-6;

	public:
		MilsteinScheme(std::tuple<std::shared_ptr<Sde<T, T, T, T>>, std::shared_ptr<Sde<T, T, T, T>>> const &model,
			T correlation, T const &delta, std::size_t numberSteps) :
			SchemeBuilder<2, T, T, T, T>{ model,correlation,delta,numberSteps } {
		}

		MilsteinScheme(std::tuple<std::shared_ptr<Sde<T, T, T, T>>, std::shared_ptr<Sde<T, T, T, T>>> const &model,
			T correlation) :
			SchemeBuilder<2, T, T, T, T>{ model,correlation } {}

		std::pair<PathValuesType<T>, PathValuesType<T>> simulate(std::random_device::result_type seed) override {
			mt_.seed(seed);
			T z1{};
			T z2{};

			PathValuesType<T> path0(this->numberSteps_);
			PathValuesType<T> path1(this->numberSteps_);
			auto firstModel = std::get<0>(this->model_);
			auto secondModel = std::get<1>(this->model_);
			path0[0] = firstModel->initCondition();
			path1[0] = secondModel->initCondition();
			auto firstSpot = firstModel->initCondition();
			T firstSpotNew{};
			auto secondSpot = secondModel->initCondition();
			T secondSpotNew{};
			T sqrtdt{ std::sqrt(this->delta_) };
			T const dt{ this->delta_ };
			T const stp{ this->step_ };
			T const corr{ this->correlation_ };

			for (std::size_t i = 1; i < path0.size(); ++i) {
				z1 = normal1_(mt_);
				z2 = normal2_(mt_);

				firstSpotNew = firstSpot +
					firstModel->drift((i - 1)*dt, firstSpot, secondSpot)*dt +
					firstModel->diffusion((i - 1)*dt, firstSpot, secondSpot) *sqrtdt * z1 +
					0.5*firstModel->diffusion((i - 1)*dt, firstSpot, secondSpot) *
					((firstModel->diffusion((i - 1)*dt, firstSpot + 0.5*stp, secondSpot) -
						firstModel->diffusion((i - 1)*dt, firstSpot - 0.5*stp, secondSpot)) / stp)*
					dt*((z1)*(z1)-1.0) +
					0.5*corr*secondModel->diffusion((i - 1)*dt, firstSpot, secondSpot) *
					((firstModel->diffusion((i - 1)*dt, firstSpot, secondSpot + 0.5*stp) -
						firstModel->diffusion((i - 1)*dt, firstSpot, secondSpot - 0.5*stp)) / stp) *
					dt*((z1)*(z1)-1.0) +
					std::sqrt(1.0 - corr * corr)*
					secondModel->diffusion((i - 1)*dt, firstSpot, secondSpot)*
					((firstModel->diffusion((i - 1)*dt, firstSpot, secondSpot + 0.5*stp) -
						firstModel->diffusion((i - 1)*dt, firstSpot, secondSpot - 0.5*stp)) / stp) *
					dt*z1*z2;

				secondSpotNew = secondSpot +
					secondModel->drift((i - 1)*dt, firstSpot, secondSpot)*dt +
					secondModel->diffusion((i - 1)*dt, firstSpot, secondSpot) *sqrtdt *
					(corr * z1 + std::sqrt(1.0 - (corr * corr)) * z2) +
					0.5*corr*firstModel->diffusion((i - 1)*dt, firstSpot, secondSpot)*
					((secondModel->diffusion((i - 1)*dt, firstSpot + 0.5*stp, secondSpot) -
						secondModel->diffusion((i - 1)*dt, firstSpot - 0.5*stp, secondSpot)) / stp)*
						dt * ((z1)*(z1)-1.0) +
					0.5 * secondModel->diffusion((i - 1)*dt, firstSpot, secondSpot)*
					((secondModel->diffusion((i - 1)*dt, firstSpot, secondSpot + 0.5*stp) -
						secondModel->diffusion((i - 1)*dt, firstSpot, secondSpot - 0.5*stp)) / stp)*
					dt * ((corr*z1 + std::sqrt(1.0 - corr * corr)*z2)*
					(corr*z1 + std::sqrt(1.0 - corr * corr)*z2) - 1.0) +
					std::sqrt(1.0 - corr *corr)*
					firstModel->diffusion((i - 1)*dt, firstSpot, secondSpot)*
					((secondModel->diffusion((i - 1)*dt, firstSpot + 0.5*stp, secondSpot) -
						secondModel->diffusion((i - 1)*dt, firstSpot - 0.5*stp, secondSpot)) / stp) *
					dt*z1*z2;

				path0[i] = firstSpotNew;
				path1[i] = secondSpotNew;
				firstSpot = firstSpotNew;
				secondSpot = secondSpotNew;
			}
			return std::make_pair(path0, path1);
		}

		std::pair<PathValuesType<T>, PathValuesType<T>> simulateWithTimePoints(std::random_device::result_type seed, TimePointsType<T> &timePoints)override {
			this->mt_.seed(seed);
			T z1{};
			T z2{};
			assert(!timePoints.empty());
			PathValuesType<T> path0(timePoints.size());
			PathValuesType<T> path1(timePoints.size());
			auto firstModel = std::get<0>(this->model_);
			auto secondModel = std::get<1>(this->model_);
			path0[0] = firstModel->initCondition();
			path1[0] = secondModel->initCondition();
			auto firstSpot = firstModel->initCondition();
			T firstSpotNew{};
			auto secondSpot = secondModel->initCondition();
			T secondSpotNew{};
			T dt{};
			T tp{};
			T const stp{ this->step_ };
			T const corr{ this->correlation_ };

			for (std::size_t i = 1; i < path0.size(); ++i) {
				z1 = normal1_(mt_);
				z2 = normal2_(mt_);
				dt = timePoints[i] - timePoints[i - 1];
				tp = timePoints[i - 1];

				firstSpotNew = firstSpot +
					firstModel->drift(tp, firstSpot, secondSpot)*dt +
					firstModel->diffusion(tp, firstSpot, secondSpot) *
					std::sqrt(dt) * z1 +
					0.5*firstModel->diffusion(tp, firstSpot, secondSpot) *
					((firstModel->diffusion(tp, firstSpot + 0.5*stp, secondSpot) -
						firstModel->diffusion(tp, firstSpot - 0.5*stp, secondSpot)) / stp)*
						dt*((z1)*(z1)-1.0) +
					0.5*corr*secondModel->diffusion(tp, firstSpot, secondSpot) *
					((firstModel->diffusion(tp, firstSpot, secondSpot + 0.5*stp) -
						firstModel->diffusion(tp, firstSpot, secondSpot - 0.5*stp)) / stp) *
						dt*((z1)*(z1)-1.0) +
					std::sqrt(1.0 - corr * corr)*
					secondModel->diffusion(tp, firstSpot, secondSpot)*
					((firstModel->diffusion(tp, firstSpot, secondSpot + 0.5*stp) -
						firstModel->diffusion(tp, firstSpot, secondSpot - 0.5*stp)) / stp) *
						dt*z1*z2;

				secondSpotNew = secondSpot +
					secondModel->drift(tp, firstSpot, secondSpot)*dt +
					secondModel->diffusion(tp, firstSpot, secondSpot) *
					std::sqrt(dt) *
					(corr * z1 + std::sqrt(1.0 - (corr * corr)) * z2) +
					0.5*corr*firstModel->diffusion(tp, firstSpot, secondSpot)*
					((secondModel->diffusion(tp, firstSpot + 0.5*stp, secondSpot) -
						secondModel->diffusion(tp, firstSpot - 0.5*stp, secondSpot)) / stp)*
						dt * ((z1)*(z1)-1.0) +
					0.5 * secondModel->diffusion(tp, firstSpot, secondSpot)*
					((secondModel->diffusion(tp, firstSpot, secondSpot + 0.5*stp) -
						secondModel->diffusion(tp, firstSpot, secondSpot - 0.5*stp)) / stp)*
						dt * ((corr*z1 + std::sqrt(1.0 - corr * corr)*z2)*
					(corr*z1 + std::sqrt(1.0 - corr * corr)*z2) - 1.0) +
					std::sqrt(1.0 - corr * corr)*
					firstModel->diffusion(tp, firstSpot, secondSpot)*
					((secondModel->diffusion(tp, firstSpot + 0.5*stp, secondSpot) -
						secondModel->diffusion(tp, firstSpot - 0.5*stp, secondSpot)) / stp) *
						dt*z1*z2;

				path0[i] = firstSpotNew;
				path1[i] = secondSpotNew;
				firstSpot = firstSpotNew;
				secondSpot = secondSpotNew;
			}
			return std::make_pair(path0,path1);
		}



	};

}



#endif ///_FDM_SCHEME_H_