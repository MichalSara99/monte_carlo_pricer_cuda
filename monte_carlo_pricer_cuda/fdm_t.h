#pragma once
#if !defined(_FDM_H_)
#define _FDM_T_H_

#include"sde_builder.h"
#include"fdm.h"
#include<iostream>
#include<chrono>


using namespace finite_difference_method;

void fdm_gbm() {

	double rate{ 0.001 };
	double sigma{ 0.005 };
	double s{ 100.0 };

	sde_builder::GeometricBrownianMotion<> gbm{ rate,sigma,s };
	std::cout << "Number of factors: " << sde_builder::GeometricBrownianMotion<>::FactorCount << "\n";
	auto sde = gbm.model();
	auto diffusion = gbm.diffusion();
	auto drift = gbm.drift();

	constexpr std::size_t factors = sde_builder::GeometricBrownianMotion<>::FactorCount;
	Fdm<factors, double> gbm_fdm{ sde,1.0,160 };
	auto times = gbm_fdm.timeResolution();

	std::cout << "timing: \n";
	auto start = std::chrono::system_clock::now();
	auto euler_collector = gbm_fdm(500'000);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Euler took: " << end << " seconds\n";
	start = std::chrono::system_clock::now();
	auto milstein_collector = gbm_fdm(500'000, FDMScheme::MilsteinScheme);
	end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Milstein took: " << end << " seconds\n";
	std::cout << "\n";

	// transform for pricing:
	auto paths_euler = euler_collector->transform(mc_types::SlicingType::PerPathSlicing);
	// last values:
	for (std::size_t t = 0; t < 30; ++t) {
		std::cout << t << " paths: \n";
		std::cout << paths_euler[t][159] << ", ";
		std::cout << "\n";
	}
	std::cout << "==============================\n";
	// To see few generated values:
	for (std::size_t t = 0; t < 30; ++t) {
		std::cout << t << " paths: \n";
		for (std::size_t v = 0; v < 10; ++v) {
			std::cout << paths_euler[t][v] << ", ";
		}
		std::cout << "\n";
	}

}

void fdm_heston() {

	float r_d{ 0.05f };
	float r_f{ 0.01f };
	float mu = r_d - r_f;
	float sigma{ 0.01f };
	float theta{ 0.015f };
	float kappa{ 0.12f };
	float etha{ 0.012f };
	float stock_init{ 100.0f };
	float var_init{ 0.025f };


	sde_builder::HestonModel<float> hest{ mu,sigma,kappa,theta,etha,stock_init,var_init };
	std::cout << "Number of factors: " << sde_builder::HestonModel<>::FactorCount << "\n";

	constexpr std::size_t factors = sde_builder::HestonModel<>::FactorCount;
	Fdm<sde_builder::HestonModel<>::FactorCount, float> heston_fdm{ hest.model(),1.0,0.8f,2 * 360 };
	auto times = heston_fdm.timeResolution();

	std::cout << "timing: \n";
	auto start = std::chrono::system_clock::now();
	auto euler_collector = heston_fdm(70'000);
	auto end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
	std::cout << "Euler took: " << end << " seconds\n";
	{
		start = std::chrono::system_clock::now();
		auto milstein_collector = heston_fdm(70'000, FDMScheme::MilsteinScheme);
		end = std::chrono::duration<double>(std::chrono::system_clock::now() - start).count();
		std::cout << "Milstein took: " << end << " seconds\n";
		std::cout << "\n";
	}

	// transform for pricing:
	auto paths_euler = euler_collector->transform(mc_types::SlicingType::PerPathSlicing, 0);

	// To see few generated values:
	for (std::size_t t = 0; t < 30; ++t) {
		std::cout << t << " paths: \n";
		for (std::size_t v = 0; v < 10; ++v) {
			std::cout << paths_euler[t][v] << ", ";
		}
		std::cout << "\n";
	}
	// To see few generated values:
	//for (std::size_t t = 0; t < 30; ++t) {
	//	std::cout << t << " paths: \n";
	//	for (std::size_t v = 0; v < 10; ++v) {
	//		std::cout << paths_milstein[t][v] << ", ";
	//	}
	//	std::cout << "\n";
	//}
}





#endif ///_FDM_T_H_

