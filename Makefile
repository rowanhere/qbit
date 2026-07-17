NVCC ?= nvcc
CXX ?= g++
TARGET ?= qbminer
ARCH ?= sm_89
GENCODE ?=

ifeq ($(strip $(GENCODE)),)
NVCC_ARCH_FLAGS := -arch=$(ARCH)
else
NVCC_ARCH_FLAGS := $(foreach cc,$(GENCODE),-gencode arch=compute_$(cc),code=sm_$(cc))
endif

NVCCFLAGS ?= -O3 -std=c++17 $(NVCC_ARCH_FLAGS) -lineinfo -cudart=static
LDFLAGS ?= -pthread

all: $(TARGET)

$(TARGET): src/qbminer.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
