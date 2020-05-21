#pragma once
#if !defined(_OPTION)
#define _OPTION


#include"option_family.h"

namespace option {

	using namespace option_family;


	template<typename T = double,
		template<typename> typename OptionFamily = black_scholes_merton::BlackScholesMertonOption>
	class EuropeanCall {
	private:
		OptionFamily<T> family_;

	public:
		explicit EuropeanCall(){}

		inline void setCostOfCarry(T value) {family_.setCostOfCarry(value);}
		inline void setSettleRate(T value) { family_.setSettleRate(value); }
		inline void setOptionRiskFreeRate(T value) {family_.setOptionRiskFreeRate(value);}
		inline void setStrike(T value) { family_.setStrike(value); }
		inline void setTimeToSettlement(T value) { family_.setTimeToSettlement(value); }
		inline void setTimeToMaturity(T value) { family_.setTimeToMaturity(value); }
		inline void setVolatility(T value) { family_.setVolatility(value); }
		inline T price(T underlying) { return family_.call(underlying); }

	};


	template<typename T = double,
		template<typename> typename OptionFamily = black_scholes_merton::BlackScholesMertonOption>
	class EuropeanPut {
	private:
		OptionFamily<T> family_;

	public:
		explicit EuropeanPut() {}

		inline void setCostOfCarry(T value) { family_.setCostOfCarry(value); }
		inline void setSettleRate(T value) { family_.setSettleRate(value); }
		inline void setOptionRiskFreeRate(T value) { family_.setOptionRiskFreeRate(value); }
		inline void setStrike(T value) { family_.setStrike(value); }
		inline void setTimeToSettlement(T value) { family_.setTimeToSettlement(value); }
		inline void setTimeToMaturity(T value) { family_.setTimeToMaturity(value); }
		inline void setVolatility(T value) { family_.setVolatility(value); }
		inline T price(T underlying) { return family_.put(underlying); }

	};








}







#endif ///_OPTION


