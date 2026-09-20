# Testsuite target.
TARGET := testsuite

# Root directory of the implementation under test, i.e. the exercises directory holding this
# 'test' directory. Override with 'make ML_DIR=<path>' to test a tree elsewhere.
ML_DIR := ..

# Directory holding the yrgo-test framework, checked out as a submodule at the root of this
# repository. Run 'git submodule update --init --recursive' once if it's empty.
YRGO_TEST_DIR := ../../../../libs/test
YRGO_TEST_LIB := $(YRGO_TEST_DIR)/libyrgo_test.a

# Source files under test - the dense layer interface and stub are header-only, while the dense
# layer and the neural network have source files. Update this list as new source files are added
# to the implementation.
# Note that main.cpp is excluded: the test suite provides its own main() in testsuite.cpp.
SOURCE_FILES := $(ML_DIR)/source/ml/dense_layer/dense.cpp \
                $(ML_DIR)/source/ml/neural_network/shallow.cpp \

# Test files - update this list as new test files are added to this directory.
TEST_FILES := ml/dense_layer/dense_test.cpp \
              ml/dense_layer/stub_test.cpp \
              ml/neural_network/shallow_test.cpp \
              testsuite.cpp \

# All files.
ALL_FILES := $(SOURCE_FILES) $(TEST_FILES)

# Include directory of the implementation under test.
INC_DIR := $(ML_DIR)/include

# C++ compiler.
CXX_COMPILER := g++

# C++ compiler flags.
CXX_FLAGS := -std=c++17 -Wall -Werror -I$(INC_DIR) -I$(YRGO_TEST_DIR)/include

# Build and run the test suite as default.
default: build run

# Build the yrgo-test framework.
$(YRGO_TEST_LIB):
	@$(MAKE) -C $(YRGO_TEST_DIR) lib

# Build the test suite.
build: $(YRGO_TEST_LIB)
	@$(CXX_COMPILER) $(ALL_FILES) -o $(TARGET) $(CXX_FLAGS) -L$(YRGO_TEST_DIR) -lyrgo_test

# Run the test suite.
run:
	@./$(TARGET)

# Clean the test suite.
clean:
	@rm -f $(TARGET)

.PHONY: default build run clean
