#pragma once
#if !defined(_EXAMPLES_CUDA_H_)
#define _EXAMPLES_CUDA_H_

#include<iostream>
#include<string>
#include<chrono>
#include"payoff.h"
#include"payoff_strategy.h"
#include"fdm_cuda.h"
#include"sde_builder_cuda.h"
#include<boost/date_time/gregorian/gregorian.hpp>
#include<boost/date_time/gregorian/gregorian_io.hpp>

using namespace finite_difference_method_cuda;
using namespace sde_builder_cuda;
using namespace boost::gregorian;
//===================================================================
//=== equidistant time points
//===================================================================

// Pricing european options 
// using paths from geometric brownian motion  
void europeanOptionsGBMEulerCuda() {

	// First generate paths using GBM 
	double rate{ 0.001 };
	double sigma{ 0.005 };
	double s{ 100.0 };
	double maturityInYears{ 1.0 };
	std::size_t numberSteps{ 720 }; // two times a day
	std::size_t simuls{ 270'000 };

	// Construct the model:
	sde_builder_cuda::GeometricBrownianMotion<> gbm{ rate,sigma,s };
	std::cout << "CUDA:\n";
	std::cout << "Model: " << sde_builder_cuda::GeometricBrownianMotion<>::name() << "\n";
	std::cout << "Factors: " << sde_builder_cuda::GeometricBrownianMotion<>::FactorCount << "\n";
	// Construct the engine: 
	FdmCUDA<sde_builder_cuda::GeometricBrownianMotion<>::FactorCount, double> fdm_gbm{maturityInYears,numberSteps };
	auto start = std::chrono::system_clock::now();
	auto collector = fdm_gbm(gbm,simuls, FDMScheme::EulerScheme);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Euler scheme for GBM<1> ("<<simuls<<") took: " << end << " seconds.\n";

	// last values:
	std::cout << "==============================\n";
	// transform for pricing:
	auto paths_euler = collector->transform(mc_types::SlicingType::PerPathSlicing);
	for (std::size_t t = 0; t < 30; ++t) {
		std::cout << t << " paths: \n";
		std::cout << paths_euler[t][719] << ", ";
		std::cout << "\n";
	}
	std::cout << "==============================\n";


	// Construct call payoff of the option:
	double call_strike{ 100.0 };
	payoff::PlainCallStrategy<> call_strategy{ call_strike };
	auto call_payoff = std::bind(&payoff::PlainCallStrategy<>::payoff, &call_strategy, std::placeholders::_1);
	// Construct put payoff of the option:
	double put_strike{ 100.0 };
	payoff::PlainPutStrategy<> put_strategy{ put_strike };
	auto put_payoff = std::bind(&payoff::PlainPutStrategy<>::payoff, &put_strategy, std::placeholders::_1);

	double lastPrice{ 0.0 };
	double call_sum{ 0.0 };
	double put_sum{ 0.0 };
	for (auto const &path : paths_euler) {
		lastPrice = path.back();
		call_sum += call_payoff(lastPrice);
		put_sum += put_payoff(lastPrice);
	}
	auto call_avg = (call_sum / static_cast<double>(paths_euler.size()));
	auto put_avg = (put_sum / static_cast<double>(paths_euler.size()));

	std::cout << "Call price: " << ((std::exp(-1.0*rate*maturityInYears))*call_avg) << "\n";
	std::cout << "Put price: " << ((std::exp(-1.0*rate*maturityInYears))*put_avg) << "\n";
	std::cout << "=========================================================\n";
}

// Pricing european options 
// using paths from geometric brownian motion  
void europeanOptionsGBMMilsteinCuda() {

	// First generate paths using GBM 
	double rate{ 0.001 };
	double sigma{ 0.005 };
	double s{ 100.0 };
	double maturityInYears{ 1.0 };
	std::size_t numberSteps{ 720 }; // two times a day
	std::size_t simuls{ 270'000 };

	// Construct the model:
	sde_builder_cuda::GeometricBrownianMotion<> gbm{ rate,sigma,s };
	std::cout << "CUDA:\n";
	std::cout << "Model: " << sde_builder_cuda::GeometricBrownianMotion<>::name() << "\n";
	std::cout << "Factors: " << sde_builder_cuda::GeometricBrownianMotion<>::FactorCount << "\n";
	// Construct the engine: 
	FdmCUDA<sde_builder_cuda::GeometricBrownianMotion<>::FactorCount, double> fdm_gbm{maturityInYears,numberSteps };
	auto start = std::chrono::system_clock::now();
	auto collector = fdm_gbm(gbm,simuls, FDMScheme::MilsteinScheme);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Milstein scheme for GBM<1>  (" << simuls << ") took: " << end << " seconds.\n";

	// Construct call payoff of the option:
	double call_strike{ 100.0 };
	payoff::PlainCallStrategy<> call_strategy{ call_strike };
	auto call_payoff = std::bind(&payoff::PlainCallStrategy<>::payoff, &call_strategy, std::placeholders::_1);
	// Construct put payoff of the option:
	double put_strike{ 100.0 };
	payoff::PlainPutStrategy<> put_strategy{ put_strike };
	auto put_payoff = std::bind(&payoff::PlainPutStrategy<>::payoff, &put_strategy, std::placeholders::_1);

	double lastPrice{ 0.0 };
	double call_sum{ 0.0 };
	double put_sum{ 0.0 };
	// transform for pricing:
	auto paths_euler = collector->transform(mc_types::SlicingType::PerPathSlicing);
	for (auto const &path : paths_euler) {
		lastPrice = path.back();
		call_sum += call_payoff(lastPrice);
		put_sum += put_payoff(lastPrice);
	}
	auto call_avg = (call_sum / static_cast<double>(paths_euler.size()));
	auto put_avg = (put_sum / static_cast<double>(paths_euler.size()));

	std::cout << "Call price: " << ((std::exp(-1.0*rate*maturityInYears))*call_avg) << "\n";
	std::cout << "Put price: " << ((std::exp(-1.0*rate*maturityInYears))*put_avg) << "\n";
	std::cout << "=========================================================\n";
}

// Pricing european options 
// using paths from geometric brownian motion  
void europeanOptionsHestonEulerCuda() {

	// First generate paths using Heston Model:
	double rate_d{ 0.005 };
	double rate_f{ 0.004 };
	double mu{ rate_d - rate_f };
	double sigma{ 0.005 };
	double theta{ 0.0155 };
	double kappa{ 0.14 };
	double etha{ 0.012 };
	double s_init{ 100.0 };
	double var_init{ 0.025 };
	double correlation{ 0.0 };
	double maturityInYears{ 2.0 };
	std::size_t numberSteps{ 730 }; // two times a day
	std::size_t simuls{ 250'000 };

	// Construct the model:
	sde_builder_cuda::HestonModel<> heston{ mu,sigma,kappa,theta,etha,s_init,var_init,correlation };
	std::cout << "CUDA:\n";
	std::cout << "Model: " << sde_builder_cuda::HestonModel<>::name() << "\n";
	std::cout << "Factors: " << sde_builder_cuda::HestonModel<>::FactorCount << "\n";
	// Construct the engine: 
	FdmCUDA<sde_builder_cuda::HestonModel<>::FactorCount, double> fdm_heston{ maturityInYears,numberSteps };
	auto start = std::chrono::system_clock::now();
	auto collector = fdm_heston(heston,simuls, FDMScheme::EulerScheme);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Euler scheme for Heston<2>  (" << simuls << ") took: " << end << " seconds.\n";

	// Construct call payoff of the option:
	double call_strike{ 100.0 };
	payoff::PlainCallStrategy<> call_strategy{ call_strike };
	auto call_payoff = std::bind(&payoff::PlainCallStrategy<>::payoff, &call_strategy, std::placeholders::_1);
	// Construct put payoff of the option:
	double put_strike{ 100.0 };
	payoff::PlainPutStrategy<> put_strategy{ put_strike };
	auto put_payoff = std::bind(&payoff::PlainPutStrategy<>::payoff, &put_strategy, std::placeholders::_1);

	double lastPrice{ 0.0 };
	double call_sum{ 0.0 };
	double put_sum{ 0.0 };
	// transform first factor for pricing:
	auto paths_euler = collector->transform(mc_types::SlicingType::PerPathSlicing,0);
	for (auto const &path : paths_euler) {
		lastPrice = path.back();
		call_sum += call_payoff(lastPrice);
		put_sum += put_payoff(lastPrice);
	}
	auto call_avg = (call_sum / static_cast<double>(paths_euler.size()));
	auto put_avg = (put_sum / static_cast<double>(paths_euler.size()));

	std::cout << "Call price: " << ((std::exp(-1.0*mu*maturityInYears))*call_avg) << "\n";
	std::cout << "Put price: " << ((std::exp(-1.0*mu*maturityInYears))*put_avg) << "\n";
	std::cout << "=========================================================\n";
}


// Pricing european options 
// using paths from geometric brownian motion  
void europeanOptionsHestonMilsteinCuda() {

	// First generate paths using Heston Model:
	double rate_d{ 0.005 };
	double rate_f{ 0.004 };
	double mu{ rate_d - rate_f };
	double sigma{ 0.005 };
	double theta{ 0.0155 };
	double kappa{ 0.14 };
	double etha{ 0.012 };
	double s_init{ 100.0 };
	double var_init{ 0.025 };
	double correlation{ 0.0 };
	double maturityInYears{ 1.0 };
	std::size_t numberSteps{ 720 }; // two times a day
	std::size_t simuls{ 250'000 };

	// Construct the model:
	sde_builder_cuda::HestonModel<> heston{ mu,sigma,kappa,theta,etha,s_init,var_init,correlation };
	std::cout << "CUDA:\n";
	std::cout << "Model: " << heston.name() << "\n";
	std::cout << "Factors: " << sde_builder_cuda::HestonModel<>::FactorCount << "\n";
	// Construct the engine: 
	FdmCUDA<sde_builder_cuda::HestonModel<>::FactorCount, double> fdm_heston{maturityInYears,numberSteps };
	auto start = std::chrono::system_clock::now();
	auto collector = fdm_heston(heston,simuls, FDMScheme::MilsteinScheme);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Milstein scheme for Heston<2> (" << simuls << ") took: " << end << " seconds.\n";

	// Construct call payoff of the option:
	double call_strike{ 100.0 };
	payoff::PlainCallStrategy<> call_strategy{ call_strike };
	auto call_payoff = std::bind(&payoff::PlainCallStrategy<>::payoff, &call_strategy, std::placeholders::_1);
	// Construct put payoff of the option:
	double put_strike{ 100.0 };
	payoff::PlainPutStrategy<> put_strategy{ put_strike };
	auto put_payoff = std::bind(&payoff::PlainPutStrategy<>::payoff, &put_strategy, std::placeholders::_1);

	double lastPrice{ 0.0 };
	double call_sum{ 0.0 };
	double put_sum{ 0.0 };
	// transform first factor for pricing:
	auto paths_euler = collector->transform(mc_types::SlicingType::PerPathSlicing, 0);
	for (auto const &path : paths_euler) {
		lastPrice = path.back();
		call_sum += call_payoff(lastPrice);
		put_sum += put_payoff(lastPrice);
	}
	auto call_avg = (call_sum / static_cast<double>(paths_euler.size()));
	auto put_avg = (put_sum / static_cast<double>(paths_euler.size()));

	std::cout << "Call price: " << ((std::exp(-1.0*mu*maturityInYears))*call_avg) << "\n";
	std::cout << "Put price: " << ((std::exp(-1.0*mu*maturityInYears))*put_avg) << "\n";
	std::cout << "=========================================================\n";
}


// Pricing asian options 
// using paths from geometric brownian motion  
void asianOptionsGBMEulerCuda() {

	// First generate paths using GBM 
	double rate{ 0.001 };
	double sigma{ 0.005 };
	double s{ 100.0 };
	double maturityInYears{ 1.0 };
	std::size_t numberSteps{ 720 }; // two times a day
	std::size_t simuls{ 250'000 };

	// Construct the model:
	sde_builder_cuda::GeometricBrownianMotion<> gbm{ rate,sigma,s };
	std::cout << "CUDA:\n";
	std::cout << "Model: " << gbm.name() << "\n";
	std::cout << "Factors: " << sde_builder_cuda::GeometricBrownianMotion<>::FactorCount << "\n";
	// Construct the engine: 
	FdmCUDA<sde_builder_cuda::GeometricBrownianMotion<>::FactorCount, double> fdm_gbm{ maturityInYears,numberSteps };
	auto start = std::chrono::system_clock::now();
	auto collector = fdm_gbm(gbm,simuls, FDMScheme::EulerScheme);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Euler scheme for GBM<1> (" << simuls << ")  took: " << end << " seconds.\n";

	// Construct call payoff of the option:
	double call_strike{ 100.0 };
	payoff::AsianAvgCallStrategy<> call_strategy{ call_strike };
	auto call_payoff = std::bind(&payoff::AsianAvgCallStrategy<>::payoff, &call_strategy, std::placeholders::_1);
	// Construct put payoff of the option:
	double put_strike{ 100.0 };
	payoff::AsianAvgPutStrategy<> put_strategy{ put_strike };
	auto put_payoff = std::bind(&payoff::AsianAvgPutStrategy<>::payoff, &put_strategy, std::placeholders::_1);

	double call_sum{ 0.0 };
	double put_sum{ 0.0 };
	// transform  factor for pricing:
	auto paths_euler = collector->transform(mc_types::SlicingType::PerPathSlicing);
	for (auto const &path : paths_euler) {
		call_sum += call_payoff(path);
		put_sum += put_payoff(path);
	}
	auto call_avg = (call_sum / static_cast<double>(paths_euler.size()));
	auto put_avg = (put_sum / static_cast<double>(paths_euler.size()));

	std::cout << "Asian call price: " << ((std::exp(-1.0*rate*maturityInYears))*call_avg) << "\n";
	std::cout << "Asian put price: " << ((std::exp(-1.0*rate*maturityInYears))*put_avg) << "\n";
	std::cout << "=========================================================\n";
}


// Pricing european options 
// using paths from geometric brownian motion  
void asianOptionsGBMMilsteinCuda() {

	// First generate paths using GBM 
	double rate{ 0.001 };
	double sigma{ 0.005 };
	double s{ 100.0 };
	double maturityInYears{ 1.0 };
	std::size_t numberSteps{ 720 }; // two times a day
	std::size_t simuls{ 250'000 };

	// Construct the model:
	sde_builder_cuda::GeometricBrownianMotion<> gbm{ rate,sigma,s };
	std::cout << "CUDA:\n";
	std::cout << "Model: " << gbm.name() << "\n";
	std::cout << "Factors: " << sde_builder_cuda::GeometricBrownianMotion<>::FactorCount << "\n";
	// Construct the engine: 
	FdmCUDA<sde_builder_cuda::GeometricBrownianMotion<>::FactorCount, double> fdm_gbm{ maturityInYears,numberSteps };
	auto start = std::chrono::system_clock::now();
	auto collector = fdm_gbm(gbm,simuls, FDMScheme::MilsteinScheme);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Milstein scheme for GBM<1> (" << simuls << ") took: " << end << " seconds.\n";

	// Construct call payoff of the option:
	double call_strike{ 100.0 };
	payoff::AsianAvgCallStrategy<> call_strategy{ call_strike };
	auto call_payoff = std::bind(&payoff::AsianAvgCallStrategy<>::payoff, &call_strategy, std::placeholders::_1);
	// Construct put payoff of the option:
	double put_strike{ 100.0 };
	payoff::AsianAvgPutStrategy<> put_strategy{ put_strike };
	auto put_payoff = std::bind(&payoff::AsianAvgPutStrategy<>::payoff, &put_strategy, std::placeholders::_1);

	double call_sum{ 0.0 };
	double put_sum{ 0.0 };
	// transform  factor for pricing:
	auto paths_euler = collector->transform(mc_types::SlicingType::PerPathSlicing);
	for (auto const &path : paths_euler) {
		call_sum += call_payoff(path);
		put_sum += put_payoff(path);
	}
	auto call_avg = (call_sum / static_cast<double>(paths_euler.size()));
	auto put_avg = (put_sum / static_cast<double>(paths_euler.size()));

	std::cout << "Call price: " << ((std::exp(-1.0*rate*maturityInYears))*call_avg) << "\n";
	std::cout << "Put price: " << ((std::exp(-1.0*rate*maturityInYears))*put_avg) << "\n";
	std::cout << "=========================================================\n";
}

// Pricing asian options 
// using paths from geometric brownian motion  
void asianOptionsHestonEulerCuda() {

	// First generate paths using Heston Model:
	double rate_d{ 0.005 };
	double rate_f{ 0.004 };
	double mu{ rate_d - rate_f };
	double sigma{ 0.005 };
	double theta{ 0.0155 };
	double kappa{ 0.14 };
	double etha{ 0.012 };
	double s_init{ 100.0 };
	double var_init{ 0.025 };
	double correlation{ 0.0 };
	double maturityInYears{ 1.0 };
	std::size_t numberSteps{ 720 }; // two times a day
	std::size_t simuls{ 450'000 };

	// Construct the model:
	sde_builder_cuda::HestonModel<> heston{ mu,sigma,kappa,theta,etha,s_init,var_init,correlation };
	std::cout << "CUDA:\n";
	std::cout << "Model: " << heston.name() << "\n";
	std::cout << "Factors: " << sde_builder_cuda::HestonModel<>::FactorCount << "\n";
	// Construct the engine: 
	FdmCUDA<sde_builder_cuda::HestonModel<>::FactorCount, double> fdm_heston{ maturityInYears,numberSteps };
	auto start = std::chrono::system_clock::now();
	auto collector = fdm_heston(heston,simuls, FDMScheme::EulerScheme,GPUConfiguration::Grid2D);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Euler scheme for Heston<2> (" << simuls << ") took: " << end << " seconds.\n";

	// Construct call payoff of the option:
	double call_strike{ 100.0 };
	payoff::AsianAvgCallStrategy<> call_strategy{ call_strike };
	auto call_payoff = std::bind(&payoff::AsianAvgCallStrategy<>::payoff, &call_strategy, std::placeholders::_1);
	// Construct put payoff of the option:
	double put_strike{ 100.0 };
	payoff::AsianAvgPutStrategy<> put_strategy{ put_strike };
	auto put_payoff = std::bind(&payoff::AsianAvgPutStrategy<>::payoff, &put_strategy, std::placeholders::_1);

	double call_sum{ 0.0 };
	double put_sum{ 0.0 };
	// transform first factor for pricing:
	auto paths_euler = collector->transform(mc_types::SlicingType::PerPathSlicing,0);
	for (auto const &path : paths_euler) {
		call_sum += call_payoff(path);
		put_sum += put_payoff(path);
	}
	auto call_avg = (call_sum / static_cast<double>(paths_euler.size()));
	auto put_avg = (put_sum / static_cast<double>(paths_euler.size()));

	std::cout << "Call price: " << ((std::exp(-1.0*mu*maturityInYears))*call_avg) << "\n";
	std::cout << "Put price: " << ((std::exp(-1.0*mu*maturityInYears))*put_avg) << "\n";
	std::cout << "=========================================================\n";
}

// Pricing asian options 
// using paths from geometric brownian motion  
void asianOptionsHestonMilsteinCuda() {

	// First generate paths using Heston Model:
	double rate_d{ 0.005 };
	double rate_f{ 0.004 };
	double mu{ rate_d - rate_f };
	double sigma{ 0.005 };
	double theta{ 0.0155 };
	double kappa{ 0.14 };
	double etha{ 0.012 };
	double s_init{ 100.0 };
	double var_init{ 0.025 };
	double correlation{ 0.0 };
	double maturityInYears{ 1.0 };
	std::size_t numberSteps{ 720 }; // two times a day
	std::size_t simuls{ 550'000 };

	// Construct the model:
	sde_builder_cuda::HestonModel<> heston{ mu,sigma,kappa,theta,etha,s_init,var_init,correlation };
	std::cout << "CUDA:\n";
	std::cout << "Model: " << heston.name() << "\n";
	std::cout << "Factors: " << sde_builder_cuda::HestonModel<>::FactorCount << "\n";
	// Construct the engine: 
	FdmCUDA<sde_builder_cuda::HestonModel<>::FactorCount, double> fdm_heston{ maturityInYears,numberSteps };
	auto start = std::chrono::system_clock::now();
	auto collector = fdm_heston(heston,simuls, FDMScheme::MilsteinScheme,GPUConfiguration::Grid2D);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Milstein scheme for Heston<2> (" << simuls << ") took: " << end << " seconds.\n";

	// Construct call payoff of the option:
	double call_strike{ 100.0 };
	payoff::AsianAvgCallStrategy<> call_strategy{ call_strike };
	auto call_payoff = std::bind(&payoff::AsianAvgCallStrategy<>::payoff, &call_strategy, std::placeholders::_1);
	// Construct put payoff of the option:
	double put_strike{ 100.0 };
	payoff::AsianAvgPutStrategy<> put_strategy{ put_strike };
	auto put_payoff = std::bind(&payoff::AsianAvgPutStrategy<>::payoff, &put_strategy, std::placeholders::_1);

	double call_sum{ 0.0 };
	double put_sum{ 0.0 };
	// transform first factor for pricing:
	auto paths_euler = collector->transform(mc_types::SlicingType::PerPathSlicing, 0);
	for (std::size_t t = 0; t < 30; ++t) {
		std::cout << paths_euler[t][719] << "\n";
	}
	for (auto const &path : paths_euler) {
		call_sum += call_payoff(path);
		put_sum += put_payoff(path);
	}
	auto call_avg = (call_sum / static_cast<double>(paths_euler.size()));
	auto put_avg = (put_sum / static_cast<double>(paths_euler.size()));

	std::cout << "Call price: " << ((std::exp(-1.0*mu*maturityInYears))*call_avg) << "\n";
	std::cout << "Put price: " << ((std::exp(-1.0*mu*maturityInYears))*put_avg) << "\n";
	std::cout << "=========================================================\n";
}

//===================================================================
//=== non-equidistant time points
//===================================================================

// Pricing european options 
// using paths from geometric brownian motion  
void europeanOptionsGBMEulerTPCuda() {

	// First generate paths using GBM 
	double rate{ 0.001 };
	double sigma{ 0.005 };
	double s{ 100.0 };
	double daysInYear{ 365.0 };
	std::size_t numberSteps{ 730 };
	std::size_t simuls{ 270'000 };

	// Generate fixing dates:
	int size = numberSteps;
	std::vector<date> fixingDates;
	date today(day_clock::local_day());
	fixingDates.push_back(today);
	for (int i = 1; i < size; ++i) {
		fixingDates.emplace_back(today + date_duration(i));
	}
	std::vector<double> timePoints;

	// Generate time points:
	for (auto const &d : fixingDates) {
		timePoints.push_back(((d - today).days()) / daysInYear);
	}

	// Construct the model:
	sde_builder_cuda::GeometricBrownianMotion<> gbm{ rate,sigma,s };
	std::cout << "CUDA:\n";
	std::cout << "Model: " << sde_builder_cuda::GeometricBrownianMotion<>::name() << "\n";
	std::cout << "Factors: " << sde_builder_cuda::GeometricBrownianMotion<>::FactorCount << "\n";
	// Construct the engine: 
	FdmCUDA<sde_builder_cuda::GeometricBrownianMotion<>::FactorCount, double> fdm_gbm{timePoints };
	auto start = std::chrono::system_clock::now();
	auto collector = fdm_gbm(gbm,simuls, FDMScheme::EulerScheme);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Euler scheme for GBM<1> (" << simuls << ") took: " << end << " seconds.\n";

	// transform factor for pricing:
	auto paths_euler = collector->transform(mc_types::SlicingType::PerPathSlicing);
	// last values:
	for (std::size_t t = 0; t < 30; ++t) {
		std::cout << t << " paths: \n";
		std::cout << paths_euler[t][719] << ", ";
		std::cout << "\n";
	}
	std::cout << "==============================\n";


	// Construct call payoff of the option:
	double call_strike{ 100.0 };
	payoff::PlainCallStrategy<> call_strategy{ call_strike };
	auto call_payoff = std::bind(&payoff::PlainCallStrategy<>::payoff, &call_strategy, std::placeholders::_1);
	// Construct put payoff of the option:
	double put_strike{ 100.0 };
	payoff::PlainPutStrategy<> put_strategy{ put_strike };
	auto put_payoff = std::bind(&payoff::PlainPutStrategy<>::payoff, &put_strategy, std::placeholders::_1);

	double lastPrice{ 0.0 };
	double call_sum{ 0.0 };
	double put_sum{ 0.0 };
	for (auto const &path : paths_euler) {
		lastPrice = path.back();
		call_sum += call_payoff(lastPrice);
		put_sum += put_payoff(lastPrice);
	}
	auto call_avg = (call_sum / static_cast<double>(paths_euler.size()));
	auto put_avg = (put_sum / static_cast<double>(paths_euler.size()));

	std::cout << "Call price: " << ((std::exp(-1.0*rate*timePoints.back()))*call_avg) << "\n";
	std::cout << "Put price: " << ((std::exp(-1.0*rate*timePoints.back()))*put_avg) << "\n";
	std::cout << "=========================================================\n";
}

// Pricing european options 
// using paths from geometric brownian motion  
void europeanOptionsGBMMilsteinTPCuda() {

	// First generate paths using GBM 
	double rate{ 0.001 };
	double sigma{ 0.005 };
	double s{ 100.0 };
	double daysInYear{ 365.0 };
	std::size_t numberSteps{ 730 }; // two times a day
	std::size_t simuls{ 270'000 };

	// Generate fixing dates:
	int size = numberSteps;
	std::vector<date> fixingDates;
	date today(day_clock::local_day());
	fixingDates.push_back(today);
	for (int i = 1; i < size; ++i) {
		fixingDates.emplace_back(today + date_duration(i));
	}
	std::vector<double> timePoints;

	// Generate time points:
	for (auto const &d : fixingDates) {
		timePoints.push_back(((d - today).days()) / daysInYear);
	}

	// Construct the model:
	sde_builder_cuda::GeometricBrownianMotion<> gbm{ rate,sigma,s };
	std::cout << "CUDA:\n";
	std::cout << "Model: " << sde_builder_cuda::GeometricBrownianMotion<>::name() << "\n";
	std::cout << "Factors: " << sde_builder_cuda::GeometricBrownianMotion<>::FactorCount << "\n";
	// Construct the engine: 
	FdmCUDA<sde_builder_cuda::GeometricBrownianMotion<>::FactorCount, double> fdm_gbm{ timePoints };
	auto start = std::chrono::system_clock::now();
	auto collector = fdm_gbm(gbm, simuls, FDMScheme::MilsteinScheme);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Milstein scheme for GBM<1>  (" << simuls << ") took: " << end << " seconds.\n";

	// Construct call payoff of the option:
	double call_strike{ 100.0 };
	payoff::PlainCallStrategy<> call_strategy{ call_strike };
	auto call_payoff = std::bind(&payoff::PlainCallStrategy<>::payoff, &call_strategy, std::placeholders::_1);
	// Construct put payoff of the option:
	double put_strike{ 100.0 };
	payoff::PlainPutStrategy<> put_strategy{ put_strike };
	auto put_payoff = std::bind(&payoff::PlainPutStrategy<>::payoff, &put_strategy, std::placeholders::_1);

	double lastPrice{ 0.0 };
	double call_sum{ 0.0 };
	double put_sum{ 0.0 };
	// transform factor for pricing:
	auto paths_euler = collector->transform(mc_types::SlicingType::PerPathSlicing);
	for (auto const &path : paths_euler) {
		lastPrice = path.back();
		call_sum += call_payoff(lastPrice);
		put_sum += put_payoff(lastPrice);
	}
	auto call_avg = (call_sum / static_cast<double>(paths_euler.size()));
	auto put_avg = (put_sum / static_cast<double>(paths_euler.size()));

	std::cout << "Call price: " << ((std::exp(-1.0*rate*timePoints.back()))*call_avg) << "\n";
	std::cout << "Put price: " << ((std::exp(-1.0*rate*timePoints.back()))*put_avg) << "\n";
	std::cout << "=========================================================\n";
}

// Pricing european options 
// using paths from geometric brownian motion  
void europeanOptionsHestonEulerTPCuda() {

	// First generate paths using Heston Model:
	double rate_d{ 0.005 };
	double rate_f{ 0.004 };
	double mu{ rate_d - rate_f };
	double sigma{ 0.005 };
	double theta{ 0.0155 };
	double kappa{ 0.14 };
	double etha{ 0.012 };
	double s_init{ 100.0 };
	double var_init{ 0.025 };
	double correlation{ 0.0 };
	double daysInYear{ 365.0 };
	std::size_t numberSteps{ 730 }; // two times a day
	std::size_t simuls{ 250'000 };

	// Generate fixing dates:
	int size = numberSteps;
	std::vector<date> fixingDates;
	date today(day_clock::local_day());
	fixingDates.push_back(today);
	for (int i = 1; i < size; ++i) {
		fixingDates.emplace_back(today + date_duration(i));
	}
	std::vector<double> timePoints;

	// Generate time points:
	for (auto const &d : fixingDates) {
		timePoints.push_back(((d - today).days()) / daysInYear);
	}

	// Construct the model:
	sde_builder_cuda::HestonModel<> heston{ mu,sigma,kappa,theta,etha,s_init,var_init,correlation };
	std::cout << "CUDA:\n";
	std::cout << "Model: " << sde_builder_cuda::HestonModel<>::name() << "\n";
	std::cout << "Factors: " << sde_builder_cuda::HestonModel<>::FactorCount << "\n";
	// Construct the engine: 
	FdmCUDA<sde_builder_cuda::HestonModel<>::FactorCount, double> fdm_heston{ timePoints };
	auto start = std::chrono::system_clock::now();
	auto collector = fdm_heston(heston, simuls, FDMScheme::EulerScheme);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Euler scheme for Heston<2>  (" << simuls << ") took: " << end << " seconds.\n";

	// Construct call payoff of the option:
	double call_strike{ 100.0 };
	payoff::PlainCallStrategy<> call_strategy{ call_strike };
	auto call_payoff = std::bind(&payoff::PlainCallStrategy<>::payoff, &call_strategy, std::placeholders::_1);
	// Construct put payoff of the option:
	double put_strike{ 100.0 };
	payoff::PlainPutStrategy<> put_strategy{ put_strike };
	auto put_payoff = std::bind(&payoff::PlainPutStrategy<>::payoff, &put_strategy, std::placeholders::_1);

	double lastPrice{ 0.0 };
	double call_sum{ 0.0 };
	double put_sum{ 0.0 };
	// transform first factor for pricing:
	auto paths_euler = collector->transform(mc_types::SlicingType::PerPathSlicing,0);
	for (auto const &path : paths_euler) {
		lastPrice = path.back();
		call_sum += call_payoff(lastPrice);
		put_sum += put_payoff(lastPrice);
	}
	auto call_avg = (call_sum / static_cast<double>(paths_euler.size()));
	auto put_avg = (put_sum / static_cast<double>(paths_euler.size()));

	std::cout << "Call price: " << ((std::exp(-1.0*mu*timePoints.back()))*call_avg) << "\n";
	std::cout << "Put price: " << ((std::exp(-1.0*mu*timePoints.back()))*put_avg) << "\n";
	std::cout << "=========================================================\n";
}


#endif ///_EXAMPLES_CUDA_H_