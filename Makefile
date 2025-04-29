CXX = g++
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -Wpedantic -Iinclude

SRC_DIR = src

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/ops/*.cpp $(SRC_DIR)/modules/*.cpp)

MAIN_EXE := main

TEST_SRC = $(filter-out $(SRC_DIR)/main.cpp, $(SRC_FILES)) tests/test_main.cpp
TEST_EXE = test_main

all: $(MAIN_EXE)

$(MAIN_EXE): $(SRC_FILES)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TEST_EXE): $(TEST_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

test: $(TEST_EXE)
	./$(TEST_EXE)

debug: CXXFLAGS = -std=c++20 -O0 -g -Wall -Wextra -Wpedantic -Iinclude
debug: $(SRC_FILES)
	$(CXX) $(CXXFLAGS) $^ -o $(MAIN_EXE)

clean:
	rm -f $(MAIN_EXE) $(TEST_EXE)
