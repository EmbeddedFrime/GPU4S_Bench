#pragma once
#include <chrono>

//Create an class for a shorter call
// chrono timestamps for kernel timing (CLBlast event profiling unreliable on Android)
class Clock
{
private:
	std::chrono::high_resolution_clock::time_point _timePointA, _timePointB;
public:
		
	void start(){
		_timePointA = std::chrono::high_resolution_clock::now();
	}

	void end(){
		_timePointB = std::chrono::high_resolution_clock::now();
	}

	float getElapsed(){
		return std::chrono::duration<float, std::milli>(_timePointB - _timePointA).count() * 1000000.0f;
	}
};
