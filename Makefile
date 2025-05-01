# Detect OS and set default compiler
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # macOS - use clang++ explicitly
    CXX = clang++

    # Detect architecture
    UNAME_M := $(shell uname -m)
    ifeq ($(UNAME_M),arm64)
        # M1/ARM Mac specific flags
        CXXFLAGS_ARCH = -arch arm64
    else ifeq ($(UNAME_M),x86_64)
        # Intel Mac specific flags
        CXXFLAGS_ARCH = -arch x86_64
    endif
else
    # Linux and others - use g++
    CXX = g++
    CXXFLAGS_ARCH =
endif

# Initialize CXXFLAGS and LDFLAGS
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -Wpedantic -Iinclude $(CXXFLAGS_ARCH)
LDFLAGS = $(CXXFLAGS_ARCH)

# Check for OpenMP support
# Add this near the top of your Makefile
USE_OMP ?= 1  # Default to enabled, can be overridden from command line

# Modify your OpenMP section
ifeq ($(UNAME_S),Darwin)
    # macOS - check if we're using Apple Clang
    ifeq ($(shell $(CXX) --version | grep -q "Apple clang"; echo $$?),0)
        # Apple Clang - check if OpenMP is installed via Homebrew
        ifneq ($(shell brew --prefix libomp 2>/dev/null),)
            BREW_LIBOMP_PREFIX := $(shell brew --prefix libomp)
            ifeq ($(USE_OMP),1)
                CXXFLAGS += -Xpreprocessor -fopenmp -I$(BREW_LIBOMP_PREFIX)/include -DUSE_OMP
                LDFLAGS += -L$(BREW_LIBOMP_PREFIX)/lib -lomp
                $(info Building with OpenMP support)
            else
                CXXFLAGS += -Wno-unknown-pragmas
                $(info Building without OpenMP support)
            endif
        else
            $(warning OpenMP not found. Install with: brew install libomp)
            CXXFLAGS += -Wno-unknown-pragmas
        endif
    else
        # Assume non-Apple Clang or GCC - should have OpenMP support
        ifeq ($(USE_OMP),1)
            CXXFLAGS += -fopenmp -DUSE_OMP
            LDFLAGS += -fopenmp
            $(info Building with OpenMP support)
        else
            CXXFLAGS += -Wno-unknown-pragmas
            $(info Building without OpenMP support)
        endif
    endif
else
    # Linux and others - assume OpenMP is available
    ifeq ($(USE_OMP),1)
        CXXFLAGS += -fopenmp -DUSE_OMP
        LDFLAGS += -fopenmp
        $(info Building with OpenMP support)
    else
        CXXFLAGS += -Wno-unknown-pragmas
        $(info Building without OpenMP support)
    endif
endif

# Check compiler version and capabilities
CXX_VERSION := $(shell $(CXX) --version)
$(info Using compiler: $(CXX_VERSION))

# Verify C++20 support
ifneq ($(shell $(CXX) -std=c++20 -dM -E - < /dev/null 2>&1 | grep -c "__cplusplus >= 202002L"),1)
    $(warning This compiler might not fully support C++20)
endif

SRC_DIR = src
BUILD_DIR = build

SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/ops/*.cpp $(SRC_DIR)/modules/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))

MAIN_EXE := main

TEST_SRC = $(filter-out $(SRC_DIR)/main.cpp, $(SRC_FILES)) tests/test_main.cpp
TEST_OBJ = $(filter-out $(BUILD_DIR)/main.o, $(OBJ_FILES)) $(BUILD_DIR)/test_main.o
TEST_EXE = test_main

# Create build directory structure
$(shell mkdir -p $(BUILD_DIR) $(BUILD_DIR)/ops $(BUILD_DIR)/modules)

all: $(MAIN_EXE)

$(MAIN_EXE): $(OBJ_FILES)
	$(CXX) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/test_main.o: tests/test_main.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TEST_EXE): $(TEST_OBJ)
	$(CXX) $^ -o $@ $(LDFLAGS)

test: $(TEST_EXE)
	./$(TEST_EXE)

debug: CXXFLAGS += -O0 -g -DDEBUG
debug: $(MAIN_EXE)

clean:
	rm -rf $(BUILD_DIR) $(MAIN_EXE) $(TEST_EXE)

.PHONY: all test debug clean
