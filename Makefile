CXX = g++
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -Wpedantic -Iinclude

SRC_DIR = src

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/ops/*.cpp)

MAIN_EXE := main

all: $(MAIN_EXE)

$(MAIN_EXE): $(SRC_FILES)
	$(CXX) $(CXXFLAGS) $^ -o $@

debug: CXXFLAGS = -std=c++20 -O0 -g -Wall -Wextra -Wpedantic -Iinclude
debug: $(SRC_FILES)
	$(CXX) $(CXXFLAGS) $^ -o $(MAIN_EXE)

clean:
	rm -f $(MAIN_EXE)
