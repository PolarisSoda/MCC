#include "hnsw.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>
#include <omp.h>

using namespace std;

vector<int> HNSWGraph::searchLayer(Item& q, int ep, int ef, int lc) {
	set<pair<double, int>> candidates; //후보군
	set<pair<double, int>> nearestNeighbors; //판명난 nearestNeighbor?
	unordered_set<int> isVisited; //방문했다.

	double td = q.dist(items[ep]); //item q와 items[ep]간의 거리.
	candidates.insert(make_pair(td, ep));
	nearestNeighbors.insert(make_pair(td, ep));
	isVisited.insert(ep);

	while(!candidates.empty()) {
		auto ci = candidates.begin(); candidates.erase(candidates.begin()); //set에서 거리가 가장 작은 친구를 가져오고,
		int nid = ci->second; // 그 친구의 nid를 적는다.
		auto fi = nearestNeighbors.end(); fi--; //nearestNeightbor의 가장 끝의 iterator를 가져온다.

		if(ci->first > fi->first) break; //만약 neareastNeighbor의 가장 큰값보다 candidate가 크면 죽는다.

		//lc는 현재 레이어의 층수를 말하는 것이다.
		for(int ed: layerEdgeLists[lc][nid]) {
			if(isVisited.find(ed) != isVisited.end()) continue; //중복 체크.

			fi = nearestNeighbors.end(); fi--;
			isVisited.insert(ed);
			td = q.dist(items[ed]);
			if ((td < fi->first) || nearestNeighbors.size() < ef) {
				candidates.insert(make_pair(td, ed));
				nearestNeighbors.insert(make_pair(td, ed));
				if(nearestNeighbors.size() > ef) nearestNeighbors.erase(fi);
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

//
/*
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

	// number of items
	int itemNum;
	// actual vector of the items
	vector<Item> items;
	// adjacent edge lists in each layer
	vector<unordered_map<int, vector<int>>> layerEdgeLists;
	// enter node id
	int enterNode;
*/

void HNSWGraph::Insert(Item& q) {
	int nid = items.size();
	itemNum++; items.push_back(q); //item을 넣고 item에 번호를 부여해준다일까요.

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
	// Until Here is Safe though.

	// search up layer entrance
	int ep = enterNode;
	for (int i = maxLyer; i > l; i--) ep = searchLayer(q, ep, 1, i)[0];
	for (int i = min(l, maxLyer); i >= 0; i--) {
		int MM = l == 0 ? MMax0 : MMax;
		vector<int> neighbors = searchLayer(q, ep, efConstruction, i); //neighbor를 efConsturction만큼 찾는다.
		vector<int> selectedNeighbors = vector<int>(neighbors.begin(), neighbors.begin()+min(int(neighbors.size()), M)); //최대 M개 까지의 이웃을 선택한다.
		for (int n: selectedNeighbors) addEdge(n, nid, i); //그것으로 Edge를 추가하다.

		//모든 이웃에 대해서 가지고 있는 이웃의 숫자가 MM보다 크다면
		//여기서 지워지는데 참조하므로 segfault가 발생할 가능성이 크다.
		for (int n: selectedNeighbors) {
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
