# Define the compiler and the default flags
NVCC = nvcc
CXXFLAGS = -lcudart -lcublas -lcudnn

# Define the target executables and output directories
TARGET = vgg
TARGET_OPTIMIZED = vgg_optimized
OUTPUT_DIR = bin

# Define the source files
SRCS = vgg.cu vgg_optimized.cu

# Rule to build the target executables from source files
$(TARGET): vgg.cu
	$(NVCC) $< -o $(OUTPUT_DIR)/$@ $(CXXFLAGS)
	@rm -f $(OUTPUT_DIR)/$(TARGET_OPTIMIZED).exp $(OUTPUT_DIR)/$(TARGET_OPTIMIZED).lib

$(TARGET_OPTIMIZED): vgg_optimized.cu
	$(NVCC) $< -o $(OUTPUT_DIR)/$@
	@rm -f $(OUTPUT_DIR)/$(TARGET_OPTIMIZED).exp $(OUTPUT_DIR)/$(TARGET_OPTIMIZED).lib

# Rule to build all targets
all: $(TARGET) $(TARGET_OPTIMIZED)

# Rule to clean the build directory
clean:
	rm -f $(OUTPUT_DIR)/$(TARGET) $(OUTPUT_DIR)/$(TARGET_OPTIMIZED)
