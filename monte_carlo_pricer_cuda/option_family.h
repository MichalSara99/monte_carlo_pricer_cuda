#pragma once
#if !defined(_OPTION_FAMILY)
#define _OPTION_FAMILY

#include"product.h"
#include"option_utilities.h"

namespace option_family {

	using product::OptionProduct;
	using option_utilities::BlackScholesMertonComponents;
	using option_utilities::Distribution;
	using option_utilities::DistributionType;


	namespace black_scholes_merton {

		// =============================================================================
		// ====================== General Black-Scholes-Merton Option ==================
		// =============================================================================

		template<typename T>
		class BlackScholesMertonOption :public OptionProduct<T> {
		protected:
			// cost of carry:
			T b_;
			// settlement rate:
			T r1_;
			// risk-free rate:
			T r_;
			// strike
			T x_;
			// time to settlement:
			T t1_;
			// time to maturity: 
			T T_;
			// volatility:
			T vol_;
			std::function<T(T, T, T, T, T)> d1_ = BlackScholesMertonComponents<T>::d1();
			std::function<T(T, T, T, T, T)> d2_ = BlackScholesMertonComponents<T>::d2();
			std::function<T(T)> cdf_ = Distribution<T>::normal(DistributionType::CDF);

		public:
			typedef T value_type;
			explicit BlackScholesMertonOption() = default;
			explicit BlackScholesMertonOption(T costOfCarry, T settleRate,
				T riskFreeRate, T strike, T timeToSettlement,
				T timeToMaturity, T volatility)
				:b_{ costOfCarry }, r1_{ settleRate },
				r_{ riskFreeRate }, x_{ strike },
				t1_{ timeToSettlement }, T_{ timeToMaturity },
				vol_{ volatility } {}

			inline void setCostOfCarry(T value) { b_ = value; }
			inline void setSettleRate(T value) { r1_ = value; }
			inline void setOptionRiskFreeRate(T value) { r_ = value; }
			inline void setStrike(T value) { x_ = value; }
			inline void setTimeToSettlement(T value) { t1_ = value; }
			inline void setTimeToMaturity(T value) { T_ = value; }
			inline void setVolatility(T value) { vol_ = value; }

			inline T put(T underlying) {
				T const s = underlying;
				T const T2 = T_ + t1_;
				T const d1 = d1_(s, x_, b_, vol_, T_);
				T const d2 = d2_(s, x_, b_, vol_, T_);
				T const main = (x_ * cdf_(-1.0*d2) - s * std::exp(b_*T_)*cdf_(-1.0*d1));
				return (std::exp(r1_*t1_)*std::exp(-1.0*r_*T2)*main);
			}

			inline T call(T underlying) {
				T const s = underlying;
				T const T2 = T_ + t1_;
				T const d1 = d1_(s, x_, b_, vol_, T_);
				T const d2 = d2_(s, x_, b_, vol_, T_);
				T const main = (s * std::exp(b_*T_) * cdf_(d1) - x_ * cdf_(d2));
				return (std::exp(r1_*t1_)*std::exp(-1.0*r_*T2)*main);
			}

		};


		// =============================================================================
		// ====================== General Black-Scholes-Merton Deltas ==================
		// =============================================================================


		template<typename T>
		class BlackScholesMertonDelta :public OptionProduct<T> {
		protected:
			// cost of carry:
			T b_;
			// settlement rate:
			T r1_;
			// risk-free rate:
			T r_;
			// strike
			T x_;
			// time to settlement:
			T t1_;
			// time to maturity: 
			T T_;
			// volatility:
			T vol_;
			std::function<T(T)> d1_ = BlackScholesMertonComponents<T>::d1();
			std::function<T(T)> d2_ = BlackScholesMertonComponents<T>::d2();
			std::function<T(T)> cdf_ = Distribution<T>::normal(DistributionType::CDF);

		public:
			typedef T value_type;
			explicit BlackScholesMertonDelta() = default;
			explicit BlackScholesMertonDelta(T costOfCarry, T settleRate,
				T riskFreeRate, T strike, T timeToSettlement,
				T timeToMaturity, T volatility)
				:b_{ costOfCarry }, r1_{ settleRate },
				r_{ riskFreeRate }, x_{ strike },
				t1_{ timeToSettlement }, T_{ timeToMaturity },
				vol_{ volatility } {}

			inline void setCostOfCarry(T value) { b_ = value; }
			inline void setSettleRate(T value) { r1_ = value; }
			inline void setOptionRiskFreeRate(T value) { r_ = value; }
			inline void setStrike(T value) { x_ = value; }
			inline void setTimeToSettlement(T value) { t1_ = value; }
			inline void setTimeToMaturity(T value) { T_ = value; }
			inline void setVolatility(T value) { vol_ = value; }

			inline T put(T underlying) {
				T const s = underlying;
				T const T2 = T_ + t1_;
				T const d1 = d1_(s, x_, b_, vol_, T_);
				T const main = std::exp(b_*T_) * cdf_(d1);
				return (std::exp(r1_*t1_)*std::exp(-1.0*r_*T2)*main);
			}

			inline T call(T underlying) {
				T const s = underlying;
				T const T2 = T_ + t1_;
				T const d1 = d1_(s, x_, b_, vol_, T_);
				T const main = std::exp(b_*T_) * cdf_(d1);
				return (std::exp(r1_*t1_)*std::exp(-1.0*r_*T2)*main);
			}

		};







	}




	// =============================================================================
	// ============================ General Barrier Option =========================
	// =============================================================================



}




#endif ///_OPTION_FAMILY