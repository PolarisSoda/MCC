/*
 * main.cpp
 * Serial version
 * Compile with -O2
 */

#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <time.h>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include "skiplist.h"
#include "cc_queue.h"

using namespace std;

typedef shared_mutex Lock;
typedef unique_lock<Lock> WriteLock;
typedef shared_lock<Lock> ReadLock;
typedef pair<int,int> pii;

//구간 당 0~16384, 총 구간 131072개. 1 << 17
//구간 당 0~32767, 총 구간 65536개. 1 << 16
//구간 당 0~65535, 총 구간 32768개. 1 << 15

constexpr int buf_size = hardware_destructive_interference_size; //false sharing 방지를 위한 align amount
alignas(buf_size) Lock SL_LOCK[1<<16]; 
int round[1<<16], finished[1<<16];
pii bb[1<<16];
Lock bb_lock[1<<16];
vector<thread> thread_V;
cc_queue task_queue;

//일단은 thread create_destroy부터 해보고, 그다음에 threadpool로 넘어가겠습니다.
//일단은 아무렇게나 thread create를 하겠죠.
//thread가 완료 했으면
//지난번에 이걸 돌렸어야 했거나, 이미 돌렸다는 것을 어케 알 것인가?
/*
i 20
i 15
i 10
q 10
q 20
q 30
i 30

그 round에서 수행되야 하는 가장 마지막 q의 경우는 알 수 있지.
q끼리는 크게 괘념치 않는다.
서로 반대되거나 i면 문제지.

system한 array가 있다.
이것은 main_thread가 job을 받으면서 기록을 하는 것이다.
이것이 수행되어야 한다라는 표시이다.
그것을 제외하고 thread들이 표시하는 array가 있다. 이것은 thread가 완수한 목적을 주는 것이다. 
항상 O - O를 잡는다.
항상 O 를 잡는다.
20 -> 1 
15 -> 2
10 -> 3
10q ->
*/

int main(int argc, char* argv[]) {
    int count = 0;
    struct timespec start, stop;
    skiplist<int, int> list(0,INT_MAX);

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
            list.insert(num,num);
        } else if (action == 'q') {      // qeury
            /*
            [TODO]
            2. SHARED LOCK을 잡는다.
            */
            if(list.find(num)!=num) cout << "ERROR: Not Found: " << num << endl;
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
    clock_gettime(CLOCK_REALTIME,&stop);

    // print results
    double elapsed_time = (stop.tv_sec - start.tv_sec) + ((double) (stop.tv_nsec - start.tv_nsec))/BILLION ;
    cout << "Elapsed time: " << elapsed_time << " sec" << "\n";
    cout << "Throughput: " << (double) count / elapsed_time << " ops (operations/sec)" << "\n";

    return (EXIT_SUCCESS);
}

void thread_function(int lo) {
    //1. Check Queue First
    node ret;
    while(Queue_Dequeue(&task_queue,&ret) != 0);
    if(ret.a == 'M') return;

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