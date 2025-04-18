CXX = g++
CXXFLAGS = -std=c++20 -O2 -Wall -Iinclude -g

all: main

main: src/main.cpp src/tensor.cpp src/functional.cpp src/utils.cpp
	$(CXX) $(CXXFLAGS) $^ -o main

clean:
	rm -f main

