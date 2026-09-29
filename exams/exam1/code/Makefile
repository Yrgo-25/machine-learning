# Target application.
TARGET := ml_exam1

# C++ compiler.
CXX_COMPILER := g++

# C++ compiler flags.
# Add -DSTUB to build the demo with ml::dense_layer::Stub instead of ml::dense_layer::Dense.
CXX_FLAGS := -Wall -Werror -std=c++17 -Iinclude

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

# Number of demo runs performed by the check target.
CHECK_RUNS := 5

# Run the demo several times, to see how often the network learns the pattern.
# The random generator is seeded with the current time in whole seconds, hence the pause.
check: build
	@for i in $$(seq 1 $(CHECK_RUNS)); do \
		echo "--- Run $$i of $(CHECK_RUNS) ---"; \
		./$(TARGET); \
		sleep 1; \
	done
	@echo "Expected: one value per prediction, {0.0}, {1.0}, {1.0}, {0.0}, in most of the runs."

# Clean the application.
clean:
	@rm -f $(TARGET)
