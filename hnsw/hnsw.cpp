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
	priority_queue<pair<double,int>,vector<pair<double,int>>,compare_less> candidates;
	priority_queue<pair<double,int>,vector<pair<double,int>>,compare_greater> nearestNeighbors;
	unordered_set<int> isVisited;

	double td = q.dist(items[ep]);

	candidates.push(make_pair(td, ep));
	nearestNeighbors.push(make_pair(td, ep));
	isVisited.insert(ep);

	while (!candidates.empty()) {
		auto ci = candidates.top();
		candidates.pop();
		int nid = ci.second; //dist가 가장 작은 친구의 nid를 가져온다.

		auto fi = nearestNeighbors.top();
		nearestNeighbors.pop();

		if(ci.first > fi.first) break; //만약 candidate의 min dist가 nearestNeighbor의 max dist보다 크면 접는다.

		vector<int> cp_layerEdgeLists = layerEdgeLists[lc][nid];
		int sz = cp_layerEdgeLists.size();

		struct alignas(64) Aligned {
			bool value = false;
			double distance;
			int id;
			char padding[51];
		};
		vector<Aligned> check_edge(sz);

		int fi_first = fi.first;
		int nsz = nearestNeighbors.size();
		int add_count = 0;

		#pragma omp parallel for shared(check_edge) firstprivate(td) reduction(+:add_count)
		for(int j=0; j<sz; j++) {
			int ed = cp_layerEdgeLists[j];
			td = q.dist(items[ed]);
			
			bool visited;
			#pragma omp critical (isVisited)
			{	

				if(!(visited = (isVisited.find(ed) != isVisited.end()))) {
					isVisited.insert(ed);
				}
			}
			if(visited) continue;

			if((td < fi_first) || nsz < ef) {
				check_edge[j].value = true;
				check_edge[j].distance = td;
				check_edge[j].id = ed;
				add_count++;
			}
		}

		for(int j=0; j<sz; j++) {
			if(check_edge[j].value == false) continue;
			candidates.push(make_pair(check_edge[j].distance, check_edge[j].id));
			nearestNeighbors.push(make_pair(check_edge[j].distance, check_edge[j].id));
			if(nearestNeighbors.size() > ef) nearestNeighbors.pop();
		}
		

		// for (int ed: layerEdgeLists[lc][nid]) { //현재 lc레이어의 nid의 Edge들을 탐색한다.

		// 	if (isVisited.find(ed) != isVisited.end()) continue; //만약 방문했으면 걍 continue.

		// 	fi = nearestNeighbors.end(); fi--; //nearestNeighbor의 dist가 가장큰 친구의 iterator를 가져온다.
		// 	isVisited.insert(ed); //visited 체크.
		// 	td = q.dist(items[ed]); //td를 더 짧은 거리로 업데이트 한다.

		// 	if ((td < fi->first) || nearestNeighbors.size() < ef) { //만약 nearestNeighbor에 들어갈 조건이 되고. ef보다 사이즈가 작다면?
		// 		candidates.insert(make_pair(td, ed)); //cand에 집어넣고
		// 		nearestNeighbors.insert(make_pair(td, ed)); //neares에도 집어넣고
		// 		if (nearestNeighbors.size() > ef) nearestNeighbors.erase(fi); //안 넘게 지워버린다.
		// 	}
		// }
	}
	vector<int> results;

	int tcnt = 0;
	while(!nearestNeighbors.empty()) {
		if(tcnt > ef) break; 
		results.push_back(nearestNeighbors.top().second);
		nearestNeighbors.pop();
		tcnt++;
	}
	//for(auto &p: nearestNeighbors) results.push_back(p.second); //결론적으로 q와 가장 가까운 순서대로 neighbor들의 nid를 가져오게 된다.
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

		// for (int n: selectedNeighbors) addEdge(n, nid, i); //전부다 연결한 다음에

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

		// for (int n: selectedNeighbors) { //연결한 Neighbor들을 전부 탐색하여
		// 	if (layerEdgeLists[i][n].size() > MM) {
		// 		vector<pair<double, int>> distPairs;
		// 		for (int nn: layerEdgeLists[i][n]) distPairs.emplace_back(items[n].dist(items[nn]), nn);
		// 		sort(distPairs.begin(), distPairs.end());
		// 		layerEdgeLists[i][n].clear();
		// 		for (int d = 0; d < min(int(distPairs.size()), MM); d++) layerEdgeLists[i][n].push_back(distPairs[d].second);
		// 	}
		// }
		ep = selectedNeighbors[0];
	}
	if (l == layerEdgeLists.size() - 1) enterNode = nid;
}
