NVCC ?= nvcc
CXX ?= g++
TARGET ?= qbminer
ARCH ?= sm_89

NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo
LDFLAGS ?=

all: $(TARGET)

$(TARGET): src/qbminer.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
