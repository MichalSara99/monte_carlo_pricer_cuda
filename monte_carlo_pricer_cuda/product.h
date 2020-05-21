#pragma once
#if !defined(_OPTION_PRODUCT)
#define _OPTION_PRODUCT

#include"option_utilities.h"


namespace product {

	template<typename T>
	class OptionProduct {
	public:
		virtual ~OptionProduct(){}

		virtual T price(T underlying) = 0;
		
		virtual T operator()(T underlying) {
			return price(underlying);
		}
	};






}






#endif ////_OPTION_PRODUCT