CXX = g++
CXXFLAGS = -std=c++20 -O2 -Wall -Iinclude

all: main

main: src/main.cpp src/tensor.cpp src/functional.cpp src/utils.cpp src/tensor_iterator.cpp
	$(CXX) $(CXXFLAGS) $^ -o main

clean:
	rm -f main

