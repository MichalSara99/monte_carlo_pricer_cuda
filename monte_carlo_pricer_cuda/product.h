#pragma once
#if !defined(_OPTION_PRODUCT)
#define _OPTION_PRODUCT

#include"option_utilities.h"


namespace product {


	template<typename T>
	class OptionProduct {
	public:
		virtual ~OptionProduct(){}

		virtual T call(T underlying) = 0;

		virtual T put(T underlying) = 0;
	};






}






#endif ////_OPTION_PRODUCT