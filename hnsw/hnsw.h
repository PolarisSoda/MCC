#ifndef HNSW_H
#define HNSW_H
#include <random>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <atomic>
#include <omp.h>

using namespace std;

struct Item {
	// Constructor
	Item() : values(vector<double>()) {}
	Item(vector<double> _values) : values(_values) {}
	vector<double> values;

	// Assume L2 distance
	// [ADDED] Can i do openmp with it?
	
	Item& operator=(const Item& other) {
        if(this != &other) values = other.values;
        return *this;
    }

	double dist(Item& other) {
		double result = 0.0;
		int sz = (int)values.size();

		// Ensure both items have the same dimension
		if (sz != other.values.size()) {
			#pragma omp critical(printer)
			{
				cerr << "Dimension mismatch: " << sz << " vs " << other.values.size() << endl;
			}
			
			exit(1); // or handle the error appropriately
		}

		for (int i = 0; i < sz; i++) {
			double diff = values[i] - other.values[i];
			result += diff * diff;
		}
		return result;
	}
};

struct HNSWGraph {
	HNSWGraph(int _M, int _MMax, int _MMax0, int _efConstruction, int _ml):M(_M),MMax(_MMax),MMax0(_MMax0),efConstruction(_efConstruction),ml(_ml){
		layerEdgeLists.push_back(unordered_map<int, vector<int>>());
		layer_lock = vector<omp_lock_t>(ml);
		for(int i=0; i<ml; i++) omp_init_lock(&layer_lock[i]);
	}
	
	// Number of neighbors
	int M;
	// Max number of neighbors in layers >= 1
	int MMax;
	// Max number of neighbors in layers 0
	int MMax0;
	// Search numbers in construction
	int efConstruction;
	// Max number of layers
	int ml;

	vector<omp_lock_t> layer_lock;

	vector<int> heights;
	
	// number of items
	atomic<int> itemNum = 0;
	// actual vector of the items
	vector<Item> items;
	// adjacent edge lists in each layer
	vector<unordered_map<int, vector<int>>> layerEdgeLists;
	// enter node id
	int enterNode;

	default_random_engine generator;

	// methods
	void addEdge(int st, int ed, int lc);
	vector<int> searchLayer(Item& q, int ep, int ef, int lc);
	void Insert(Item& q);
	vector<int> KNNSearch(Item& q, int K);


	// This will not be used....
	void printGraph() {
		for (int l = 0; l < layerEdgeLists.size(); l++) {
			cout << "Layer:" << l << endl;
			for (auto it = layerEdgeLists[l].begin(); it != layerEdgeLists[l].end(); ++it) {
				cout << it->first << ":";
				for (auto ed: it->second) cout << ed << " ";
				cout << endl;
			}
		}
	}
};

#endif
