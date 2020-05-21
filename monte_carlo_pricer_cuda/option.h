#pragma once
#if !defined(_OPTION)
#define _OPTION

#include"product.h"
#include"option_utilities.h"

namespace option {

	using product::OptionProduct;
	using option_utilities::BlackScholesMertonComponents;
	using option_utilities::Distribution;
	using option_utilities::DistributionType;



	// =============================================================================
	// ====================== General Black-Scholes-Merton Option ==================
	// =============================================================================

	template<typename T>
	class BlackScholesMertonOption :public OptionProduct<T> {
	protected:
		T costOfCarry_;
		T settleRate_;
		T optionRiskFreeRate_;
		T strike_;
		T timeToSettlement_;
		T timeToMaturity_;
		T volatility_;
		std::function<T(T)> d1 = BlackScholesMertonComponents<T>::d1();
		std::function<T(T)> d2 = BlackScholesMertonComponents<T>::d2();
		std::function<T(T)> cdf = Distribution<T>::normal(DistributionType::CDF);

	public:
		explicit BlackScholesMertonOption() = default;
		explicit BlackScholesMertonOption(T costOfCarry, T settleRate,
			T riskFreeRate, T strike, T timeToSettlement,
			T timeToMaturity, T volatility)
			:costOfCarry_{ costOfCarry }, settleRate_{ settleRate },
			riskFreeRate_{ riskFreeRate }, strike_{ strike },
			timeToSettlement_{ timeToSettlement }, timeToMaturity_{ timeToMaturity },
			volatility_{ volatility } {}

		inline void setCostOfCarry(T value) { costOfCarry_ = value; }
		inline void setSettleRate(T value) { settleRate_ = value; }
		inline void setOptionRiskFreeRate(T value) { optionRiskFreeRate_ = value; }
		inline void setStrike(T value) { strike_ = value; }
		inline void setTimeToSettlement(T value) { timeToSettlement_ = value; }
		inline void setTimeToMaturity(T value) { timeToMaturity_ = value; }
		inline void setVolatility(T value) { volatility_ = value; }

	};



	// =============================================================================
	// ============================ General Barrier Option =========================
	// =============================================================================



}




#endif ///_OPTION