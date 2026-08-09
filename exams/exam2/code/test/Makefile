# Test target.
TARGET := ml_tests

# C++ compiler.
CXX_COMPILER := g++

# Machine learning directories.
ML_INC := ../include
ML_SRC := ../source/ml

# Test framework submodule.
TEST_DIR := ../../../../libs/test
TEST_LIB := $(TEST_DIR)/libyrgo_test.a

# C++ compiler flags.
CXX_FLAGS := -Wall -Werror -std=c++17 -I$(ML_INC) -I$(TEST_DIR)/include

# Linker flags.
LD_FLAGS := -L$(TEST_DIR) -lyrgo_test

# Library source files.
ML_FILES := $(ML_SRC)/cnn/cnn.cpp \
            $(ML_SRC)/conv_layer/conv.cpp \
            $(ML_SRC)/conv_layer/max_pool.cpp \
            $(ML_SRC)/dense_layer/dense.cpp \
            $(ML_SRC)/factory/factory.cpp \
            $(ML_SRC)/flatten_layer/flatten.cpp \
            $(ML_SRC)/random/generator.cpp \
            $(ML_SRC)/utils.cpp \

# Test files.
TEST_FILES := main.cpp \
              ml/dense_layer/test_dense.cpp \
              ml/cnn/test_cnn.cpp \

# All source files.
SRC_FILES := $(ML_FILES) $(TEST_FILES)

# Build and run the application by default:
default: build run

# Build the test library.
$(TEST_LIB):
	@$(MAKE) -C $(TEST_DIR) lib

# Build the application.
build: $(TEST_LIB)
	@$(CXX_COMPILER) $(SRC_FILES) -o $(TARGET) $(CXX_FLAGS) $(LD_FLAGS)

# Run the application.
run:
	@./$(TARGET)

# Clean the application.
clean:
	@rm -f $(TARGET)
	@$(MAKE) -C $(TEST_DIR) clean
