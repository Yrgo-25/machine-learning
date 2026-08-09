# Target application.
TARGET := app

# C++ compiler.
CXX_COMPILER := g++

# Source files.
SOURCE_FILES := source/main.cpp \
                source/ml/cnn/cnn.cpp \
                source/ml/conv_layer/conv.cpp \
				source/ml/conv_layer/max_pool.cpp \
				source/ml/dense_layer/dense.cpp \
				source/ml/factory/factory.cpp \
				source/ml/flatten_layer/flatten.cpp \
				source/ml/random/generator.cpp \
				source/ml/utils.cpp \

# Include directory.
INCLUDE_DIR := -I include

# Compiler flags.
# Comment out the -DSTUB flag for using the real implementation.
COMPILER_FLAGS := -Wall -Werror -std=c++17 -DSTUB

# Build and run the target as default.
default: build run

# Build the target.
build:
	@$(CXX_COMPILER) $(SOURCE_FILES) -o $(TARGET) $(COMPILER_FLAGS) $(INCLUDE_DIR)

# Run the target.
run:
	@./$(TARGET)

# Clean the target.
clean:
	@rm -f $(TARGET)
