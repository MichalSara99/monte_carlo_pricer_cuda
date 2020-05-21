#pragma once
#if !defined(_OPTION_UTILITIES)
#define _OPTION_UTILITIES

#include<functional>




namespace option_utilities {

	enum class DistributionType { CDF, PDF };

	constexpr static double pi{ 3.14159265358979323846 };


	template<typename T>
	struct BlackScholesMertonComponents {
	public:
		static inline auto d1() {
			return [](T underlying,T strike,T constOfCarry,T sig,T maturity)->T {
				return ((std::log(underlying / strike) + (constOfCarry + 0.5*sig*sig)*maturity)
					/ (sig * std::sqrt(maturity)));
			};
		}
		static inline auto d2() {
			return [](T underlying, T strike, T constOfCarry, T sig, T maturity)->T {
				return ((std::log(underlying / strike) + (constOfCarry - 0.5*sig*sig)*maturity)
					/ (sig * std::sqrt(maturity)));
			};
		}
	};




	template<typename T>
	struct Distribution {
	private:
		static std::function<T(T)> _normal_cdf();
		static std::function<T(T)> _normal_pdf();


	public:
		static inline auto normal(DistributionType type) {
			if (type == DistributionType::CDF) {
				return _normal_cdf();
			}
			else {
				return _normal_pdf();
			}
		}



	};


}



template<typename T>
std::function<T(T)> option_utilities::Distribution<T>::_normal_cdf() {
	// Hart algorithm (1968)
	return [](T x) {
		if (x > 6.0) { return 1.0; }
		if (x < -6.0) { return 0.0; }

		T const b1{ 0.31938153 };
		T const b2{ -0.356563782 };
		T const b3{ 1.781477937 };
		T const b4{ -1.821255978 };
		T const b5{ 1.330274429 };
		T const p{ 0.2316419 };
		T const c2{ 0.3989423 };

		T const a = std::abs(x);
		T const t = 1.0 / (1.0 + a * p);
		T const b = c2 * std::exp(-0.5*x*x);
		T n = ((((b5*t + b4)*t + b3)*t + b2)*t + b1)*t;
		n = 1.0 - b * n;
		if (x < 0.0) { n = 1.0 - n; }
		return n;
	};
}


template<typename T>
std::function<T(T)> option_utilities::Distribution<T>::_normal_pdf() {
	return [](T x) {
		T tmp = 1.0 / std::sqrt(2.0*pi);
		return tmp * std::exp(-0.5*x*x);
	};
}







#endif ///_OPTION_UTILITIES