#pragma once
#if !defined(_FDM_CUDA_T)
#define _FDM_CUDA_T

#include<iostream>
#include<chrono>
#include<algorithm>
#include"sde_builder_cuda.h"
#include"fdm_cuda.h"

using namespace sde_builder_cuda;
using namespace finite_difference_method_cuda;

void gbm_cuda() {

	double rate{ 0.001 };
	double sigma{ 0.005 };
	double s{ 100.0 };

	double maturity{ 1.0 };
	std::size_t numberSteps{ 2 * 360 };
	std::size_t numberIterations{ 350'000 };
	double strike{ 100.0 };

	sde_builder_cuda::GeometricBrownianMotion<double> gbm{ rate,sigma,s };
	FdmCUDA<1, double> fdm{ maturity,numberSteps };

	{
		auto begin = std::chrono::system_clock::now();
		auto paths = fdm(gbm, numberIterations);
		auto end = std::chrono::system_clock::now();
		std::cout << "Generating on 1D grid took: " << std::chrono::duration<double>(end - begin).count() << "\n";


		std::cout << "===============================\n";

		double sum_call{ 0.0 };
		double sum_put{ 0.0 };
		for (std::size_t t = 0; t < paths.size(); ++t) {
			sum_call += max(0.0, paths[t][719] - strike);
			sum_put += max(0.0, strike - paths[t][719]);
		}
		auto call_price = std::exp(-1.0*rate*maturity)*(sum_call / static_cast<double>(paths.size()));
		auto put_price = std::exp(-1.0*rate*maturity)*(sum_put / static_cast<double>(paths.size()));
		std::cout << "Call price: " << call_price << "\n";
		std::cout << "Put price: " << put_price << "\n";

	}
	{
		auto begin = std::chrono::system_clock::now();
		auto paths = fdm(gbm, numberIterations, FDMScheme::EulerScheme, GPUConfiguration::Grid2D);
		auto end = std::chrono::system_clock::now();

		std::cout << "Generating on 2D grid took: " << std::chrono::duration<double>(end - begin).count() << "\n";

		std::cout << "=============================\n";
		for (std::size_t t = 0; t < 30; ++t) {
			std::cout << paths[t][719] << "\n";
		}


		std::cout << "===============================\n";

		double sum_call{ 0.0 };
		double sum_put{ 0.0 };
		for (std::size_t t = 0; t < paths.size(); ++t) {
			sum_call += max(0.0, paths[t][719] - strike);
			sum_put += max(0.0, strike - paths[t][719]);
		}
		auto call_price = std::exp(-1.0*rate*maturity)*(sum_call / static_cast<double>(paths.size()));
		auto put_price = std::exp(-1.0*rate*maturity)*(sum_put / static_cast<double>(paths.size()));
		std::cout << "Call price: " << call_price << "\n";
		std::cout << "Put price: " << put_price << "\n";
	}

	auto begin = std::chrono::system_clock::now();
	auto paths = fdm(gbm, numberIterations, FDMScheme::EulerScheme, GPUConfiguration::Grid3D);
	auto end = std::chrono::system_clock::now();

	std::cout << "Generating on 3D grid took: " << std::chrono::duration<double>(end - begin).count() << "\n";

	std::cout << "=============================\n";
	for (std::size_t t = 0; t < 30; ++t) {
		std::cout << paths[t][719] << "\n";
	}


	std::cout << "===============================\n";

	double sum_call{ 0.0 };
	double sum_put{ 0.0 };
	for (std::size_t t = 0; t < paths.size(); ++t) {
		sum_call += max(0.0, paths[t][719] - strike);
		sum_put += max(0.0, strike - paths[t][719]);
	}
	auto call_price = std::exp(-1.0*rate*maturity)*(sum_call / static_cast<double>(paths.size()));
	auto put_price = std::exp(-1.0*rate*maturity)*(sum_put / static_cast<double>(paths.size()));
	std::cout << "Call price: " << call_price << "\n";
	std::cout << "Put price: " << put_price << "\n";

}

void heston_cuda() {

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

	std::size_t numberSteps{ 720 };
	std::size_t numberIterations{ 250'000 };
	double strike{ 100.0 };

	sde_builder_cuda::HestonModel<double> heston{ mu,sigma,kappa,theta,etha,s_init,var_init,correlation };
	FdmCUDA<2, double> fdm{ maturityInYears,numberSteps };

	{
		auto begin = std::chrono::system_clock::now();
		auto paths = fdm(heston, numberIterations, FDMScheme::MilsteinScheme);
		auto end = std::chrono::system_clock::now();
		std::cout << "Generating on 1D grid took: " << std::chrono::duration<double>(end - begin).count() << "\n";


		std::cout << "===============================\n";

		double sum_call{ 0.0 };
		double sum_put{ 0.0 };
		for (std::size_t t = 0; t < paths.size(); ++t) {
			sum_call += max(0.0, paths[t][719] - strike);
			sum_put += max(0.0, strike - paths[t][719]);
		}
		auto call_price = std::exp(-1.0*mu*maturityInYears)*(sum_call / static_cast<double>(paths.size()));
		auto put_price = std::exp(-1.0*mu*maturityInYears)*(sum_put / static_cast<double>(paths.size()));
		std::cout << "Call price: " << call_price << "\n";
		std::cout << "Put price: " << put_price << "\n";

	}


	{
		auto begin = std::chrono::system_clock::now();
		auto paths = fdm(heston, numberIterations,FDMScheme::MilsteinScheme,GPUConfiguration::Grid2D);
		auto end = std::chrono::system_clock::now();

		std::cout << "Generating on 2D grid took: " << std::chrono::duration<double>(end - begin).count() << "\n";

		std::cout << "=============================\n";
		for (std::size_t t = 0; t < 30; ++t) {
			std::cout << paths[t][719] << "\n";
		}


		std::cout << "===============================\n";

		double sum_call{ 0.0 };
		double sum_put{ 0.0 };
		for (std::size_t t = 0; t < paths.size(); ++t) {
			sum_call += max(0.0, paths[t][719] - strike);
			sum_put += max(0.0, strike - paths[t][719]);
		}
		auto call_price = std::exp(-1.0*mu*maturityInYears)*(sum_call / static_cast<double>(paths.size()));
		auto put_price = std::exp(-1.0*mu*maturityInYears)*(sum_put / static_cast<double>(paths.size()));
		std::cout << "Call price: " << call_price << "\n";
		std::cout << "Put price: " << put_price << "\n";
	}

	auto begin = std::chrono::system_clock::now();
	auto paths = fdm(heston, numberIterations, FDMScheme::MilsteinScheme, GPUConfiguration::Grid3D);
	auto end = std::chrono::system_clock::now();

	std::cout << "Generating on 3D grid took: " << std::chrono::duration<double>(end - begin).count() << "\n";

	std::cout << "=============================\n";
	for (std::size_t t = 0; t < 30; ++t) {
		std::cout << paths[t][719] << "\n";
	}


	std::cout << "===============================\n";

	double sum_call{ 0.0 };
	double sum_put{ 0.0 };
	for (std::size_t t = 0; t < paths.size(); ++t) {
		sum_call += max(0.0, paths[t][719] - strike);
		sum_put += max(0.0, strike - paths[t][719]);
	}
	auto call_price = std::exp(-1.0*mu*maturityInYears)*(sum_call / static_cast<double>(paths.size()));
	auto put_price = std::exp(-1.0*mu*maturityInYears)*(sum_put / static_cast<double>(paths.size()));
	std::cout << "Call price: " << call_price << "\n";
	std::cout << "Put price: " << put_price << "\n";

}



#endif ///_FDM_CUDA_T