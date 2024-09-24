/*
 * main.cpp
 * Serial version
 * Compile with -O2
 */

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <time.h>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include <atomic>
#include <queue>
#include "skiplist.h"

using namespace std;

//큐에 들어가서 스레드가 이것을 가지고 식별합니다.
struct info {
    char c;
    int a[4];
};
//구간 당 0~16384, 총 구간 131072개. 1 << 17
//구간 당 0~32767, 총 구간 65536개. 1 << 16
//구간 당 0~65535, 총 구간 32768개. 1 << 15
constexpr int itv_len = 1<<16;
constexpr int buf_size = hardware_destructive_interference_size; //false sharing 방지를 위한 align amount
vector<skiplist<int,int>> V_skiplist; //SkipList들을 가지고 있는 Vector
CC_queue<info> task_queue; //Task를 관리하는 Concurrent Queue
vector<pii> status; //[Master 전용], 각 skiplist에 들어간 R,W 쿼리 관리.


alignas(buf_size) Lock SL_LOCK[1<<16]; 
int round[1<<16], finished[1<<16];
pii bb[1<<16];
Lock bb_lock[1<<16];
vector<thread> thread_V;
//skiplist<int, int> skiplist_arr[1<<16];

int main(int argc,char* argv[]) {
    /*- INIT PAGE -*/
    int count = 0;
    struct timespec start, stop;
    
    for(int i=0; i<itv_len; i++) {
        int left = itv_len*i;
        int right = left + itv_len-1;
        cout << left << " " << right << endl;
        skiplist<int,int> templist(i*itv_len,1<<15-1);
        V_skiplist.push_back(templist);
    }
    exit(0);
    // check and parse command line options
    if(argc != 3) {
        printf("Usage: %s <infile> <number of thread>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *fn = argv[1];
    int num_threads = stoi(argv[2]);
    thread_V.resize(num_threads);

    clock_gettime(CLOCK_REALTIME,&start);

    // load input file
    FILE* fin = fopen(fn, "r");
    char action;
    long num;
    while(fscanf(fin, "%c %ld\n", &action, &num) > 0) {
        if(action == 'i') {            // insert
            /*
            [TODO]
            1. EXCLUSIVE LOCK를 잡는다.
            */
            //list.insert(num,num);
        } else if (action == 'q') {      // qeury
            /*
            [TODO]
            2. SHARED LOCK을 잡는다.
            */
            //if(list.find(num)!=num) cout << "ERROR: Not Found: " << num << endl;
        } else if (action == 'w') {     // wait
            // wait until previous operations finish
            // No action will be acted.
        } else if (action == 'p') {     // wait
            // wait until previous operations finish
	        //cout << list.printList() << endl;
            //p는 어떻게 할 것인가?
            //어찌돼었든 중간에 한번 끝내야 한다.
            //섞여들어가니까.
        } else {
            printf("ERROR: Unrecognized action: '%c'\n", action);
            exit(EXIT_FAILURE);
        }
	    count++;
    }
    fclose(fin);

    for(int i=0; i<num_threads; i++) thread_V[i].join();

    clock_gettime(CLOCK_REALTIME,&stop);

    // print results
    double elapsed_time = (stop.tv_sec - start.tv_sec) + ((double) (stop.tv_nsec - start.tv_nsec))/BILLION ;
    cout << "Elapsed time: " << elapsed_time << " sec" << "\n";
    cout << "Throughput: " << (double) count / elapsed_time << " ops (operations/sec)" << "\n";

    return (EXIT_SUCCESS);
}

void thread_function(int lo) {
    //1. Check Queue First



    while(bb_lock[lo].try_lock_shared() == 1) {
        bb_lock[lo].unlock_shared();
    }
}
void thread_write() {
    /*
    EX Lock을 잡는다.
    넣는다.
    푼다.
    */
}

void thread_read() {
    /*
    SHARED를 잡는다.
    값을 가져온다.
    만약 값 앞쪽이 설정한 경계면이라면 앞쪽 Shared 역시 잡는다.
    바로 헤드 바로앞으로 간다.
    */
}