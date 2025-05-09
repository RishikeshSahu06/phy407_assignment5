CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall

all: ising_model

ising_model: ising_model.cpp
	$(CXX) $(CXXFLAGS) -o ising_model ising_model.cpp

run: ising_model
	./ising_model

plot: run
	python plot_ising_results.py

clean:
	rm -f ising_model ising_data.txt ising_model_results.png