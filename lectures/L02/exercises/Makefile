# Target application.
TARGET := linreg

# C++ compiler.
CXX_COMPILER := g++

# C++ compiler flags.
CXX_FLAGS := -Wall -Werror -std=c++17 -Iinclude

# Source files.
SRC_FILES := source/main.cpp \
             source/ml/lin_reg/fixed.cpp \

# Build and run the application by default.
default: build run

# Build the application.
build:
	@$(CXX_COMPILER) $(SRC_FILES) -o $(TARGET) $(CXX_FLAGS)

# Run the application.
run:
	@./$(TARGET)

# Clean the application.
clean:
	@rm -f $(TARGET)
