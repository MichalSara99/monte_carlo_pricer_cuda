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
			return [](T underlying,T strike,T constOfCarry,T sig,T maturity) {
				(std::log(underlying / strike) + (constOfCarry + 0.5*sig*sig)*maturity)
					/ (sig * std::sqrt(maturity));
			};
		}
		static inline auto d2() {
			return [](T underlying, T strike, T constOfCarry, T sig, T maturity) {
				(std::log(underlying / strike) + (constOfCarry - 0.5*sig*sig)*maturity)
					/ (sig * std::sqrt(maturity));
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
		const T a1 = 0.0352624965998911;
		const T	a2 = 0.700383064443688;
		const T	a3 = 6.37396220353165;
		const T	a4 = 33.912866078383;
		const T	a5 = 112.079291497871;
		const T	a6 = 221.213596169931;
		const T a7 = 220.206867912376;

		const T b1 = 0.0883883476483184;
		const T b2 = 1.75566716318264;
		const T	b3 = 16.064177579207;
		const T	b4 = 86.7807322029461;
		const T	b5 = 296.564248779674;
		const T	b6 = 637.333633378831;
		const T	b7 = 793.826512519948;
		const T	b8 = 440.413735824752;

		const T y = std::abs(x);
		T p = {};
		if (y > 37) {
			p = 0.0;
		}
		else {
			const T expon{ -0.5*y*y };
			if (y < 7.07106781186547) {
				const T A = (((((a1*y + a2)*y + a3)*y + a4)*y + a5)*y + a6)*y + a7;
				const T B = ((((((b1*y + b2)*y + b3)*y + b4)*y + b5)*y + b6)*y + b7)*y + b8;
				p = ((A / B)*expon);
			}
			else {
				const T C = y + 1.0 / (y + 2.0 / (y + 3.0 / (y + 4.0 / (y + 0.65))));
				const T D = 2.506628274631;
				p = (expon / (D*C));
			}
		}
		if (x > 0.0)
			return 1.0 - p;
		else
			return p;
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