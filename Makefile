CXX = g++
CXXFLAGS = -std=c++20 -O2 -Wall -Iinclude

all: main

main: src/main.cpp src/tensor.cpp src/functional.cpp
	$(CXX) $(CXXFLAGS) $^ -o main

clean:
	rm -f main

