#pragma once
#if !defined(_PATH_COLLECTOR)
#define _PATH_COLLECTOR

#include<memory>
#include<cassert>
#include"mc_types.h"


namespace path_collector {

	using mc_types::SlicingType;

	template<std::size_t FactorCount,
		typename T>
	class PathCollector {
	};


	// Partial specialization for one-factor models:
	template<typename T>
	class PathCollector<1,T> {
	private:
		std::size_t numberPaths_;
		std::size_t pathSize_;

		std::unique_ptr<T> factorPaths_;

		void timeSlicing(std::vector<std::vector<T>> &result);
		void pathSlicing(std::vector<std::vector<T>> &result);

	public:
		explicit PathCollector(std::unique_ptr<T> factorPaths,
			std::size_t numberPaths,std::size_t pathSize)
			:factorPaths_{ std::move(factorPaths) },
			numberPaths_{ numberPaths }, pathSize_{pathSize}{}

		virtual ~PathCollector(){}

		std::vector<std::vector<T>> const transform(SlicingType type)const {
			std::vector<std::vector<T>> collection;
			if (type == SlicingType::PerPathSlicing) {
				collection.resize(numberPaths_);
				pathSlicing(collection);
				return collection;
			}
			collection.resize(pathSize_);
			timeSlicing(collection);
			return collection;
		}


	};


	// Partial specialization for two-factor models:
	template<typename T>
	class PathCollector<2, T> {
	private:
		std::size_t numberPaths_;
		std::size_t pathSize_;

		std::unique_ptr<T> factorPaths1_;
		std::unique_ptr<T> factorPaths2_;

		void timeSlicing(std::vector<std::vector<T>> &result, std::size_t factorIdx);
		void pathSlicing(std::vector<std::vector<T>> &result, std::size_t factorIdx);

	public:
		explicit PathCollector(std::unique_ptr<T> factorPaths1,std::unique_ptr<T> factorPaths2,
						std::size_t numberPaths, std::size_t pathSize)
			:factorPaths1_{ std::move(factorPaths1) }, factorPaths2_{std::move(factorPaths2)},
			numberPaths_{numberPaths},pathSize_{pathSize}{}

		virtual ~PathCollector() {}


		std::vector<std::vector<T>> const transform(SlicingType type, std::size_t factorIndex)const {
			assert((factorIndex >= 0 && factorIndex <= 1));
			std::vector<std::vector<T>> collection;
			if (type == SlicingType::PerPathSlicing) {
				collection.resize(numberPaths_);
				pathSlicing(collection, factorIndex);
				return collection;
			}
			collection.resize(pathSize_);
			timeSlicing(collection, factorIndex);
			return collection;
		}

	};
}




template<typename T>
void path_collector::PathCollector<1, T>::timeSlicing(std::vector<std::vector<T>> &result) {
	throw std::exception("noit yet implemented");
}

template<typename T>
void path_collector::PathCollector<1, T>::pathSlicing(std::vector<std::vector<T>> &result) {
	throw std::exception("noit yet implemented");
}


template<typename T>
void path_collector::PathCollector<2, T>::timeSlicing(std::vector<std::vector<T>> &result, std::size_t factorIdx) {
	throw std::exception("noit yet implemented");
}

template<typename T>
void path_collector::PathCollector<2, T>::pathSlicing(std::vector<std::vector<T>> &result, std::size_t factorIdx) {
	throw std::exception("noit yet implemented");
}








#endif ///_PATH_COLLECTOR