#include "hnsw.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>
#include <omp.h>
#include <atomic>

using namespace std;

vector<int> HNSWGraph::searchLayer(Item& q, int ep, int ef, int lc) {
	set<pair<double, int>> candidates;
	set<pair<double, int>> nearestNeighbors;
	unordered_set<int> isVisited;

	double td = q.dist(items[ep]);
	candidates.insert(make_pair(td, ep));
	nearestNeighbors.insert(make_pair(td, ep));
	isVisited.insert(ep);

	while (!candidates.empty()) {
		auto ci = candidates.begin(); 
		candidates.erase(candidates.begin());
		int nid = ci->second;
		auto fi = nearestNeighbors.end(); fi--;

		if (ci->first > fi->first) break;

		int sz = layerEdgeLists[lc][nid].size();

		struct alignas(64) Aligned {
			bool value = false;
			double distance;
			int id;
			char padding[52];
		};
		vector<Aligned> distances(sz);

		#pragma omp parallel shared(distances)
		{
			std::unordered_set<int> localVisited;

			#pragma omp for
			for (int j = 0; j < sz; j++) {
				int ed = layerEdgeLists[lc][nid][j];

				// Check in thread-local visited set to prevent duplicate work
				if (localVisited.find(ed) != localVisited.end() || isVisited.find(ed) != isVisited.end()) continue;
				localVisited.insert(ed);  // Mark as visited in thread-local set

				distances[j].distance = q.dist(items[ed]);
				distances[j].id = ed;
				distances[j].value = true;
			}

			// No need to merge `localVisited` into `isVisited` here, since we're not updating `isVisited` in parallel
		}

		// Step 2: Serial section to update global isVisited and process candidates
		for (auto tt : distances) {
			if (!tt.value) continue;

			isVisited.insert(tt.id); // Update global isVisited

			if ((tt.distance < fi->first) || nearestNeighbors.size() < ef) {
				auto temp = make_pair(tt.distance, tt.id);
				candidates.insert(temp);
				nearestNeighbors.insert(temp);

				if (nearestNeighbors.size() > ef) nearestNeighbors.erase(fi);
			}
		}
	}
	vector<int> results;
	for(auto &p: nearestNeighbors) results.push_back(p.second);
	return results;
}

vector<int> HNSWGraph::KNNSearch(Item& q, int K) {
	int maxLyer = layerEdgeLists.size() - 1;
	int ep = enterNode;
	for (int l = maxLyer; l >= 1; l--) ep = searchLayer(q, ep, 1, l)[0];
	return searchLayer(q, ep, K, 0);
}

void HNSWGraph::addEdge(int st, int ed, int lc) {
	if (st == ed) return;
	layerEdgeLists[lc][st].push_back(ed);
	layerEdgeLists[lc][ed].push_back(st);
}

void HNSWGraph::Insert(Item& q) {
	int nid = items.size();
	itemNum++; items.push_back(q);

	// sample layer
	int maxLyer = layerEdgeLists.size() - 1;
	int l = 0;
	uniform_real_distribution<double> distribution(0.0,1.0);
	while(l < ml && (1.0 / ml <= distribution(generator))) {
		l++;
		if (layerEdgeLists.size() <= l) layerEdgeLists.push_back(unordered_map<int, vector<int>>());
	}
	if (nid == 0) {
		enterNode = nid;
		return;
	}

	// search up layer entrance
	int ep = enterNode;
	for (int i = maxLyer; i > l; i--) ep = searchLayer(q, ep, 1, i)[0];

	for (int i = min(l, maxLyer); i >= 0; i--) {
		int MM = l == 0 ? MMax0 : MMax; //현재 레이어에서의 최대 neightbor수를 찾는다.
		vector<int> neighbors = searchLayer(q, ep, efConstruction, i); //neightbor의 목록을 찾고.
		vector<int> selectedNeighbors = vector<int>(neighbors.begin(), neighbors.begin()+min(int(neighbors.size()), M)); //그 중에서 상위 M개를 가져온다. 

		//for (int n: selectedNeighbors) addEdge(n, nid, i); //전부다 연결한 다음에

		int sz = selectedNeighbors.size();
		for(int j=0; j<sz; j++) {
			int n = selectedNeighbors[j];
			addEdge(n,nid,i);
		}
		
		for(int j=0; j<sz; j++) {
			int n = selectedNeighbors[j];
			if (layerEdgeLists[i][n].size() > MM) {
				vector<pair<double, int>> distPairs;
				for (int nn: layerEdgeLists[i][n]) distPairs.emplace_back(items[n].dist(items[nn]), nn);
				sort(distPairs.begin(), distPairs.end());
				layerEdgeLists[i][n].clear();
				for (int d = 0; d < min(int(distPairs.size()), MM); d++) layerEdgeLists[i][n].push_back(distPairs[d].second);
			}
		}

		for (int n: selectedNeighbors) { //연결한 Neighbor들을 전부 탐색하여
			if (layerEdgeLists[i][n].size() > MM) {
				vector<pair<double, int>> distPairs;
				for (int nn: layerEdgeLists[i][n]) distPairs.emplace_back(items[n].dist(items[nn]), nn);
				sort(distPairs.begin(), distPairs.end());
				layerEdgeLists[i][n].clear();
				for (int d = 0; d < min(int(distPairs.size()), MM); d++) layerEdgeLists[i][n].push_back(distPairs[d].second);
			}
		}
		ep = selectedNeighbors[0];
	}
	if (l == layerEdgeLists.size() - 1) enterNode = nid;
}
