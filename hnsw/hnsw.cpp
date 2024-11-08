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
#include <cstdlib>

using namespace std;

void HNSWGraph::SearchWorker(int thread_id,vector<set<pair<double,int>>>& local_candidates,vector<set<pair<double,int>>>& local_nearestNeighbors,unordered_set<int>& isVisited,omp_lock_t &lock_isVisited,int lc,int ef, Item& q) {
	using_thread++;

	while (!local_candidates[thread_id].empty()) {
		auto ci = local_candidates[thread_id].begin(); local_candidates[thread_id].erase(local_candidates[thread_id].begin());
		int nid = ci->second;
		auto fi = local_nearestNeighbors[thread_id].end(); fi--;

		if (ci->first > fi->first) break;

		for (int ed: layerEdgeLists[lc][nid]) {
			int atmpt = 2;
			bool continued = false;
			while(atmpt) {
				if(omp_test_lock(&lock_isVisited)) {
					if (isVisited.find(ed) != isVisited.end()) {
						continued = true;
						omp_unset_lock(&lock_isVisited);
						break;
					} else {
						isVisited.insert(ed);
						omp_unset_lock(&lock_isVisited);
						break;
					}
				}
				atmpt--;
			}

			if(continued) continue;
		
			fi = local_nearestNeighbors[thread_id].end(); fi--;
			double td = q.dist(items[ed]);

			if ((td < fi->first) || local_nearestNeighbors[thread_id].size() < ef) {
				if(using_thread >= 40) {
					local_candidates[thread_id].insert(make_pair(td, ed));
					local_nearestNeighbors[thread_id].insert(make_pair(td, ed));
					if (local_nearestNeighbors[thread_id].size() > ef) local_nearestNeighbors[thread_id].erase(fi);
				} else {
					#pragma omp task firstprivate(td,ed)
					{	
						int new_thread_id = omp_get_thread_num();
						local_candidates[new_thread_id].insert(make_pair(td,ed));
						local_nearestNeighbors[new_thread_id].insert(make_pair(td,ed));
						SearchWorker(new_thread_id,local_candidates,local_nearestNeighbors,isVisited,lock_isVisited,lc,ef,q);
					}
				}
			}
		}
	}
	using_thread--;
}

vector<int> HNSWGraph::searchLayer(Item& q, int ep, int ef, int lc) {
	set<pair<double, int>> candidates;
	set<pair<double, int>> nearestNeighbors;
	unordered_set<int> isVisited;

	double td = q.dist(items[ep]);

	candidates.insert(make_pair(td, ep));
	nearestNeighbors.insert(make_pair(td, ep));
	isVisited.insert(ep);

	while (!candidates.empty()) {
		auto ci = candidates.begin(); candidates.erase(candidates.begin());
		int nid = ci->second;
		auto fi = nearestNeighbors.end(); fi--;

		if (ci->first > fi->first) break;

		for (int ed: layerEdgeLists[lc][nid]) {
			if (isVisited.find(ed) != isVisited.end()) continue;
			fi = nearestNeighbors.end(); fi--;
			isVisited.insert(ed);
			td = q.dist(items[ed]);
			if ((td < fi->first) || nearestNeighbors.size() < ef) {
				candidates.insert(make_pair(td, ed));
				nearestNeighbors.insert(make_pair(td, ed));
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
	for (int i = maxLyer; i > l; i--) ep = searchLayer(q, ep, 1, i)[0]; //query가 얼마 안걸렸던 것처럼 이것도 사실 별로 안걸린다. 아마도.

	// #pragma omp parallel num_threads(40)
	// {
	// 	#pragma omp single
	// 	{
	// 		for (int i = min(l, maxLyer); i >= 0; i--) {
	// 			int MM = l == 0 ? MMax0 : MMax;
	// 			vector<int> neighbors = searchLayer(q, ep, efConstruction, i); //neightbor의 목록을 찾고.
	// 			vector<int> selectedNeighbors = vector<int>(neighbors.begin(), neighbors.begin()+min(int(neighbors.size()), M)); //그 중에서 상위 M개를 가져온다. 

	// 			for(int n: selectedNeighbors) addEdge(n, nid, i); //전부다 연결한 다음에

	// 			int sz = selectedNeighbors.size();

	// 			for(int j=0; j<sz; j++) {
	// 				#pragma omp task firstprivate(i, j, sz, selectedNeighbors, MM)
	// 				{
	// 					int n = selectedNeighbors[j];
	// 					if (layerEdgeLists[i][n].size() > MM) {
	// 						int resize_random = rand()%2;
	// 						if(resize_random) {
	// 							layerEdgeLists[i][n].resize(min(int(layerEdgeLists[i][n].size()), MM));
	// 						} else {
	// 							vector<pair<double, int>> distPairs;
	// 							for (int nn: layerEdgeLists[i][n]) distPairs.emplace_back(items[n].dist(items[nn]), nn);
	// 							sort(distPairs.begin(), distPairs.end());
	// 							layerEdgeLists[i][n].clear();
	// 							for (int d = 0; d < min(int(distPairs.size()), MM); d++) layerEdgeLists[i][n].push_back(distPairs[d].second);
	// 						}
	// 					}
	// 				}
	// 			}
	// 			ep = selectedNeighbors[0];
	// 		}
	// 	}
	// }
	
	int lsz = min({l,maxLyer,40});
	#pragma omp parallel num_threads(lsz)
	{
		#pragma omp for
		for (int i = min(l, maxLyer); i >= 0; i--) {
			int MM = l == 0 ? MMax0 : MMax;
			vector<int> neighbors = searchLayer(q, ep, efConstruction, i);
			vector<int> selectedNeighbors = vector<int>(neighbors.begin(), neighbors.begin()+min(int(neighbors.size()), M));

			for(int n: selectedNeighbors) addEdge(n, nid, i); //전부다 연결한 다음에

			int sz = selectedNeighbors.size();
			int tn = min(sz,omp_get_num_threads());

			#pragma omp parallel for num_threads(2)
			for(int j=0; j<sz; j++) {
				int n = selectedNeighbors[j];
				if (layerEdgeLists[i][n].size() > MM) {
					int resize_random = rand()%2;
					if(resize_random) {
						layerEdgeLists[i][n].resize(min(int(layerEdgeLists[i][n].size()), MM));
					} else {
						vector<pair<double, int>> distPairs;
						for (int nn: layerEdgeLists[i][n]) distPairs.emplace_back(items[n].dist(items[nn]), nn);
						sort(distPairs.begin(), distPairs.end());
						layerEdgeLists[i][n].clear();
						for (int d = 0; d < min(int(distPairs.size()), MM); d++) layerEdgeLists[i][n].push_back(distPairs[d].second);
					}
				}
			}
		}
	}
	// for (int i = min(l, maxLyer); i >= 0; i--) {
	// 	int MM = l == 0 ? MMax0 : MMax; //현재 레이어에서의 최대 neightbor수를 찾는다.
	// 	vector<int> neighbors = searchLayer(q, ep, efConstruction, i); //neightbor의 목록을 찾고.
	// 	vector<int> selectedNeighbors = vector<int>(neighbors.begin(), neighbors.begin()+min(int(neighbors.size()), M)); //그 중에서 상위 M개를 가져온다. 

	// 	for (int n: selectedNeighbors) addEdge(n, nid, i); //전부다 연결한 다음에

	// 	int sz = selectedNeighbors.size();
	// 	int tn = min(sz,omp_get_num_threads());

	// 	#pragma omp parallel for num_threads(tn)
	// 	for(int j=0; j<sz; j++) {
	// 		int n = selectedNeighbors[j];
	// 		if (layerEdgeLists[i][n].size() > MM) {
	// 			int resize_random = rand()%2;
	// 			if(resize_random) {
	// 				layerEdgeLists[i][n].resize(min(int(layerEdgeLists[i][n].size()), MM));
	// 			} else {
	// 				vector<pair<double, int>> distPairs;
	// 				for (int nn: layerEdgeLists[i][n]) distPairs.emplace_back(items[n].dist(items[nn]), nn);
	// 				sort(distPairs.begin(), distPairs.end());
	// 				layerEdgeLists[i][n].clear();
	// 				for (int d = 0; d < min(int(distPairs.size()), MM); d++) layerEdgeLists[i][n].push_back(distPairs[d].second);
	// 			}
	// 		}
	// 	}

		
	// 	// for(int n : selectedNeighbors) { //연결한 Neighbor들을 전부 탐색하여
	// 	// 	if (layerEdgeLists[i][n].size() > MM) {
	// 	// 		vector<pair<double, int>> distPairs;
	// 	// 		for (int nn: layerEdgeLists[i][n]) distPairs.emplace_back(items[n].dist(items[nn]), nn);
	// 	// 		sort(distPairs.begin(), distPairs.end());
	// 	// 		layerEdgeLists[i][n].clear();
	// 	// 		for (int d = 0; d < min(int(distPairs.size()), MM); d++) layerEdgeLists[i][n].push_back(distPairs[d].second);
	// 	// 	}
	// 	// }
	// 	ep = selectedNeighbors[0];
	// }
	if (l == layerEdgeLists.size() - 1) enterNode = nid;
}
