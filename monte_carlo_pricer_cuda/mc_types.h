#pragma once
#if !defined(_MC_TYPES_H_)
#define _MC_TYPES_H_

#include<vector>
#include<type_traits>
#include<functional>
#include<tuple>

// For 1D paths:
#define THREADS_PER_BLOCK 512
// For 2D paths:
// Note that 32 * 16 = 512 like in the 1D path
#define THREADS_2D_PER_BLOCK_X 32
#define THREADS_2D_PER_BLOCK_Y 16
// For 3D paths:
// Note that 8 * 8 * 8 = 512 like in the 1D path
#define THREADS_3D_PER_BLOCK_X 8
#define THREADS_3D_PER_BLOCK_Y 8
#define THREADS_3D_PER_BLOCK_Z 8


namespace mc_types {

	template<typename ReturnType, typename ArgType>
	using PayoffFunType = std::function<ReturnType(ArgType)>;

	template<typename T>
	using TimePointsType = std::vector<T>;

	template<typename T>
	using PathValuesType = std::vector<T>;

	template<typename T, typename ...Args>
	using SdeComponent = std::function<T(Args...)>;

	// For models
	// 0: drift lambda 1: diffusion lambda
	template<typename T, typename ...Ts>
	using ISde = std::tuple<SdeComponent<T, Ts...>, SdeComponent<T, Ts...>>;

	enum class SdeModelType { oneFactor, twoFactor };

	enum class FDMScheme { EulerScheme, MilsteinScheme };

	enum class GPUConfiguration { Grid1D, Grid2D, Grid3D };

}



#endif ///_MC_TYPES_H_