#pragma once
#if !defined(_OPTION_T)
#define _OPTION_T

#include<iostream>
#include"option.h"
#include"option_family.h"


void testBSM() {
	
	using option::EuropeanCall;
	using option::EuropeanPut;


	auto S{ 130.0 };
	auto K{ 120.0 };
	auto settle_t{ 2 / 365.0 };
	auto maturity{ 0.5 };
	auto settle_r{ 0.5 };
	auto r{ 0.1 };
	auto b{ 0.1 };
	auto vol{ 0.12 };

	EuropeanCall<> bsm_call;
	bsm_call.setStrike(K);
	bsm_call.setTimeToSettlement(settle_t);
	bsm_call.setTimeToMaturity(maturity);
	bsm_call.setSettleRate(settle_r);
	bsm_call.setOptionRiskFreeRate(r);
	bsm_call.setCostOfCarry(b);
	bsm_call.setVolatility(vol);
	auto price = bsm_call.price(S);
	std::cout << "European call price: " << price << "\n";


}







#endif ///_OPTION_T