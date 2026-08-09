# Application target.
TARGET := max_pool_demo

# C++ compiler.
CXX_COMPILER := g++

# Source files.
SOURCE_FILES := max_pool_demo.cpp \

# Compiler flags.
# TODO: Uncomment this line once the implementation is finished.
#CXX_FLAGS := -std=c++17 -Wall -Werror -DMAX_POOL_LAYER_IMPLEMENTED

# Build and run the application as default.
default: build run

# Build the application.
build:
	@$(CXX_COMPILER) $(SOURCE_FILES) -o $(TARGET) $(CXX_FLAGS)

# Run the application.
run:
	@./$(TARGET)

# Clean the application.
clean:
	@rm -f $(TARGET)
