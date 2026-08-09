# Application target.
TARGET := max_pool_demo

# C compiler.
C_COMPILER := gcc

# Source files.
SOURCE_FILES := source/max_pool_demo.c \
                source/ml/matrix.c \
				source/ml/max_pool_layer.c \

# Main include directory.
INCLUDE_DIR := -I include/

# Compiler flags.
C_FLAGS := -std=c11 -Wall -Werror

# Linked libraries.
LINK_LIBS := -lm

# Build and run the application as default.
default: build run

# Build the application.
build:
	@$(C_COMPILER) $(SOURCE_FILES) -o $(TARGET) $(C_FLAGS) $(INCLUDE_DIR) $(LINK_LIBS)

# Run the application.
run:
	@./$(TARGET)

# Clean the application.
clean:
	@rm -f $(TARGET)
