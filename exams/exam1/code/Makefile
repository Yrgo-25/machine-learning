# Target application.
TARGET := ml_exam1

# C++ compiler.
CXX_COMPILER := g++

# C++ compiler flags.
# Comment out -DSTUB to use ml::dense_layer::Dense instead of ml::dense_layer::Stub.
CXX_FLAGS := -Wall -Werror -std=c++17 -Iinclude -DSTUB

# Source files.
SRC_FILES := source/main.cpp \
             source/ml/dense_layer/dense.cpp \
			 source/ml/neural_network/shallow.cpp \
			 source/ml/helpers.cpp \

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
