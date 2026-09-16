# Target application.
TARGET := dense_layer

# C++ compiler.
CXX_COMPILER := g++

# C++ compiler flags.
CXX_FLAGS := -Wall -Werror -std=c++17 -Iinclude

# Source files.
SRC_FILES := source/main.cpp \

# Build and run the target by default.
default: build run

# Build the target.
build:
	@$(CXX_COMPILER) $(SRC_FILES) -o $(TARGET) $(CXX_FLAGS)

# Run the target.
run:
	@./$(TARGET)

# Clean the target.
clean:
	@rm -f $(TARGET)

.PHONY: default build run clean
