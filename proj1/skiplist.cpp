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
#include <condition_variable>
#include "skiplist.h"

using namespace std;

//큐에 들어가서 스레드가 이것을 가지고 식별합니다.
struct info {
    char inst;
    int target; //넣거나 찾을 숫자.
    int prereq; //가동하는데 필요한 조건.
    info() : inst('p'), target(INT_MAX), prereq(INT_MAX) {}
    info(char c,int tg,int pr) : inst(c), target(tg), prereq(pr) {} 
};

skiplist<int,int> list(0,INT_MAX); //SkipList
CC_queue<info> task_queue; //Task를 관리하는 Concurrent Queue
int BB_w = 0, BB_r = 0; //We divided Billboard into two part.
mutex BB_wlock,BB_rlock; //Billboard lock.
condition_variable BB_w_cv, BB_r_cv; //Billboard Condition Variables
vector<thread> V_thread; //Thread List

void thread_function(int lo) {
    while(true) {
        info task;
        while(task_queue.Dequeue(&task) == -1); //task를 하나 가져왔음.
        if(task.inst == 'p') return ;

        if(task.inst == 'i') {
            unique_lock<mutex> rk(BB_rlock);
            BB_r_cv.wait(rk,[&]{return BB_r == task.prereq;});
            rk.unlock();
            
            //write something.
            list.insert(task.target,task.target);

            unique_lock<mutex> wk(BB_wlock);
            BB_w++;
            BB_w_cv.notify_all();
            wk.unlock();
        } else {
            unique_lock<mutex> wk(BB_wlock);
            BB_w_cv.wait(wk,[&]{return BB_w == task.prereq;});
            wk.unlock();
            
            //read something.
            pair<int,int> ret = list.pair_find(task.target);
            if(ret.first == -1) cout << "ERROR: Not Found: " << task.target << endl;
            else cout << ret.first << " " << ret.second << "\n";

            unique_lock<mutex> rk(BB_rlock);
            BB_r++;
            BB_r_cv.notify_all();
            rk.unlock();
        }
    }
    
}

int main(int argc,char* argv[]) {
    /*- INIT PAGE -*/
    int count = 0, w_count = 0, r_count = 0;
    struct timespec start, stop;

    // check and parse command line options
    if(argc != 3) {
        printf("Usage: %s <infile> <number of thread>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *fn = argv[1];
    int num_threads = stoi(argv[2]);
    V_thread.resize(num_threads);
    clock_gettime(CLOCK_REALTIME,&start);

    // load input file
    FILE* fin = fopen(fn, "r");
    char action;
    long num;
    for(int i=0; i<num_threads; i++) V_thread[i] = thread(thread_function,i);

    //i i i i i i q q i i i
    //0 0 0 0 0 0 6 6 2 2 2
    //write: 이전 read만 다 수행되었으면 수행할 수 있음.
    //read : 이전 write만 다 수행되었으면 수행할 수 있음.

    while(fscanf(fin, "%c %ld\n", &action, &num) > 0) {
        if(action == 'i') {
            //insert(write)
            task_queue.Enqueue(info(action,num,r_count));
            w_count++;
        } else if (action == 'q') {
            //query(read)
            task_queue.Enqueue(info(action,num,w_count));
            r_count++;
        } else if (action == 'w') {
            // wait until previous operations finish
            // No action will be acted.
        } else if (action == 'p') {     // wait
            for(int i=0; i<num_threads; i++) task_queue.Enqueue(info());
            for(int i=0; i<num_threads; i++) V_thread[i].join();
            cout << list.printList() << "\n";
            for(int i=0; i<num_threads; i++) V_thread[i] = thread(thread_function,i);
        } else {
            printf("ERROR: Unrecognized action: '%c'\n", action);
            exit(EXIT_FAILURE);
        }
	    count++;
    }

    fclose(fin);
    for(int i=0; i<num_threads; i++) task_queue.Enqueue(info());
    for(int i=0; i<num_threads; i++) V_thread[i].join();

    clock_gettime(CLOCK_REALTIME,&stop);

    // print results
    double elapsed_time = (stop.tv_sec - start.tv_sec) + ((double) (stop.tv_nsec - start.tv_nsec))/BILLION ;
    cout << "Elapsed time: " << elapsed_time << " sec" << "\n";
    cout << "Throughput: " << (double) count / elapsed_time << " ops (operations/sec)" << "\n";

    return (EXIT_SUCCESS);
}