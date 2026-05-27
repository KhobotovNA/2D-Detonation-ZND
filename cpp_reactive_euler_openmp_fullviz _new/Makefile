CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
OMPFLAGS ?= -fopenmp

SRC = src/main.cpp
BIN = reactive_euler_2d_openmp

all: $(BIN)

$(BIN): $(SRC) src/*.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(SRC) -o $(BIN)

run-test: $(BIN)
	./$(BIN) --nx 160 --ny 32 --final 0.5 --save-dt 0.25 --threads 4 --out cpp_openmp_test_output

clean:
	rm -f $(BIN)
	rm -rf cpp_openmp_output cpp_openmp_test_output
