#include "hnsw.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <sstream>
#include <random>
#include <vector>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <set>
#include <fstream>
#include <shared_mutex>

using namespace std;

vector<Item> loadFile(const string& filename, int numItems, int dim) {
	vector<Item> items;
	ifstream file(filename, ios::binary);
	if (!file) {
		cerr << "Error opening file: " << filename << endl;
		exit(1);
	}
	for (int i = 0; i < numItems; i++) {
		vector<float> temp_float(dim);
		file.read(reinterpret_cast<char*>(temp_float.data()), dim * sizeof(float));
		vector<double> temp_double(dim);
		for(int j = 0; j < dim; j++){
			temp_double[j] = static_cast<double>(temp_float[j]);
		}


		items.emplace_back(temp_double);
	}
	
	return items;
}

vector<vector<int>> loadGroundTruth(const string& filename, int numQueries, int K) {
	vector<vector<int>> ground_truth(numQueries, vector<int>(K));
	ifstream file(filename, ios::binary);

	if (!file) {
		cerr << "Error opening ground truth file: " << filename << endl;
		exit(1);
	}

	for (int i = 0; i < numQueries; i++) {
		file.read(reinterpret_cast<char*>(ground_truth[i].data()), K * sizeof(int));
		file.seekg((100 - K) * sizeof(int), ios::cur);
	}
    return ground_truth;
}

void randomTest(int numItems, int dim, int numQueries, int K, int numThreads, int workload) {
	ostringstream vectorsetFile;
	vectorsetFile << "/home/swe3021/hnsw_dataset/vectorset_" << workload << ".100K.fbin";
	ostringstream groundTruthFile;
	groundTruthFile << "/home/swe3021/hnsw_dataset/ground_truth_" << workload << ".ibin";
	string queryFilename = "/home/swe3021/hnsw_dataset/queries.1K.fbin";

	vector<Item> randomItems = loadFile(vectorsetFile.str(), numItems, dim);

	vector<Item> queries = loadFile(queryFilename, numQueries, dim);

	vector<vector<int>> ground_truth = loadGroundTruth(groundTruthFile.str(), numQueries, K);

	double begin = omp_get_wtime();

	// Building Phase
	
	double begin_build = omp_get_wtime();
	cout << "START BUILDING INDEX" << endl;

	//Original 20 30 30 30 4
	HNSWGraph myHNSWGraph(20, 20, 20, 20, 5); //We are Able to modify this parameters.
	// Number of neighbors
	// Max number of neighbors in layers >= 1
	// Max number of neighbors in layers 0
	// Search numbers in construction
	// Max number of layers
	
	for(int i=0; i<numItems; i++) {
		if(i % 10000 == 0) cout << "." << std::flush;
		myHNSWGraph.Insert(randomItems[i]);
	}
	
	// #pragma omp parallel
	// {
	// 	#pragma omp single
	// 	{
	// 		for(int i=0; i<numItems; i++) {
	// 			if(i % 10000 == 0) cout << "." << std::flush;
    //             #pragma omp task firstprivate(i)
    //             {
    //                 myHNSWGraph.Insert(randomItems[i]);
    //             }
    //         }
	// 	}
	// }

	cout << endl;

	cout << "END BUILDING INDEX" << endl << endl;
	double build_time = omp_get_wtime() - begin_build;
	
	cout << "Build Time: " << build_time << " sec" << endl << endl;

	// Query Phase
	vector<double> hnsw_times(omp_get_max_threads(), 0.0);
	vector<vector<int>> all_knns(numQueries);

	double begin_query = omp_get_wtime();
	cout << "START QUERY" << endl;

	double local_hnsw_begin_time = omp_get_wtime();

	for (int i = 0; i < numQueries; i++) {
		// HNSW Search
		Item query = queries[i];
		all_knns[i] = myHNSWGraph.KNNSearch(query, K);
	}

	int thread_id = omp_get_thread_num();
	hnsw_times[thread_id] = omp_get_wtime() - local_hnsw_begin_time;
	double query_time = omp_get_wtime() - begin_query;

	cout << "END QUERY" << endl;

	for (int i = 0; i < min(5, numQueries); i++) {
    		cout << "Query " << i << " KNN Results: ";
    		for (int index : all_knns[i]) {
        		cout << index << " ";
		}
		cout << endl;
	}
	for(int i = 0; i < numThreads; i++){
		cout << "Thread " << i + 1 << " hnsw_time: " << hnsw_times[i] << " sec" << endl;
	}

	cout << "Query Time: " << query_time << " sec" << endl;
	cout << "query/sec: " << numQueries / query_time << endl << endl;

	cout << "Total Duration: " << omp_get_wtime() - begin << " sec" << endl;

	// recall@K 계산
	double total_recall = 0.0;
	for (int i = 0; i < numQueries; i++) {
		set<int> knn_set(all_knns[i].begin(), all_knns[i].end());
		set<int> ground_truth_set(ground_truth[i].begin(), ground_truth[i].end());

		Item query = queries[i]; 
			
//		cout << "Query " << i << " Comparison:" << endl;
//		cout << "KNN Set  |  Ground Truth Set  |  L2 Distance (KNN)  |  L2 Distance (Ground Truth)" << endl;
//		cout << "--------------------------------------------------------------------------" << endl;

		auto knn_it = knn_set.begin();
		auto gt_it = ground_truth_set.begin();
	
		while (knn_it != knn_set.end() && gt_it != ground_truth_set.end()) {	
                	Item knn_vector = randomItems[*knn_it];
                	Item gt_vector = randomItems[*gt_it];

                	double l2_distance_knn = knn_vector.dist(query);
                	double l2_distance_gt = gt_vector.dist(query);
				
//			cout << *knn_it << "         |  " << *gt_it << "         |  " << l2_distance_knn << "         |  " << l2_distance_gt << endl;
			
			++knn_it;
			++gt_it;
		}

		vector<int> intersection;
		set_intersection(knn_set.begin(), knn_set.end(),
		ground_truth_set.begin(), ground_truth_set.end(),
		back_inserter(intersection));

		double recall = static_cast<double>(intersection.size()) / K;
		total_recall += recall;
		//cout << "racall: " << recall << endl;
		//cout << "--------------------------------------------------------------------------" << endl;
	}
	double average_recall = total_recall / numQueries;
	cout << "Average recall@K: " << average_recall << endl;

}

int main(int argc, char* argv[]) {
	int opt;
	int opt_cnt = 0;
	int dimensions = 96;
	int numPoints = 100000;
	int numQueries = 1000;
	int k = 20;
	int numThreads = 40;
	int workload = -1;

	while((opt = getopt(argc, argv, "d:n:q:k:t:w:")) != -1) {
		switch(opt) {
			case 'd':
				//dimensions = atoi(optarg);
				break;
			case 'n':
				//numPoints = atoi(optarg);
				break;
			case 'q':
				//numQueries = atoi(optarg);
				break;
			case 'k':
			    k = atoi(optarg);
				break;
			case 't':
				numThreads = atoi(optarg);
				break;
			case 'w':
				workload = atoi(optarg);
				break;
			default:
				break;
		}
		opt_cnt++;
	}
	omp_set_num_threads(numThreads);

	if(argc < 2){
		fprintf(stderr, "Usage: %s -k {number of nearest neighbors} -t {number of threads} -w {workload num} \nFor example, %s -k10 -t10 -w1\n", argv[0], argv[0]);
		exit(0);
	}

	randomTest(numPoints, dimensions, numQueries, k, numThreads, workload);
	return 0;
}

