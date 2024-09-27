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
#include <condition_variable>
using namespace std;

//큐에 들어가서 스레드가 이것을 가지고 식별합니다.
struct info {
    char inst;
    int task_num;
    int target;
    pii ticket;
    info() : inst(0), task_num(0), target(0), ticket({0,0}) {}
    info(char c,int tn,int tg,pii t) : inst(c), task_num(tn), target(tg), ticket(t) {} 
};

constexpr int buf_size = 64;
skiplist<int,int> list(0,INT_MAX); //SkipList
CC_queue<info> task_queue; //Task를 관리하는 Concurrent Queue
pii status; //[Master 전용], 각 skiplist에 들어간 R,W 쿼리 관리.
pii Billboard = {0,0}; //[Worker 전용] 어디까지 완수되었는지 적는 용도.
mutex BB_lock; //Billboard lock.
condition_variable BB_cv;

vector<thread> V_thread;

void thread_function(int lo) {
    while(true) {
        info task;
        while(task_queue.Dequeue(&task) == -1); //task를 하나 가져왔음.
        if(task.inst == 'p') break; //task가 p일 경우, 즉시 종료함.

        //we need to check billboard
        //RW 생각하라고
        if(task.inst == 'i') {
            unique_lock<mutex> lk(BB_lock);
            // BB_cv.wait(lk,[&]{return Billboard.first == task.ticket.first;});
            BB_lock.unlock();

            // list.insert(task.target,task.target);

            lk.lock();
            // Billboard.second = max(Billboard.second,task.task_num);
            // BB_cv.notify_all();
            lk.unlock();
        } else {
            /*
            unique_lock<mutex> lk(BB_lock);
            BB_cv.wait(lk,[&]{return Billboard.second == task.ticket.second;});
            BB_lock.unlock();

            // pair<int,int> ret = list.pair_find(task.target);
            // if(ret.first == -1) cout << "ERROR: Not Found: " << task.target << "\n";
            // else cout << ret.first << " " << ret.second << "\n";

            lk.lock();
            Billboard.first = max(Billboard.second,task.task_num);
            BB_cv.notify_all();
            lk.unlock();
            */
        }
    }
    
}

int main(int argc,char* argv[]) {
    /*- INIT PAGE -*/
    int count = 0;
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
    cout << "MAKE SOMETHING\n";
    while(fscanf(fin, "%c %ld\n", &action, &num) > 0) {
        cout << action << " ";
        if(action == 'i') {
            task_queue.Enqueue(info(action,count,num,status));
            status.second++;
        } else if (action == 'q') {
            task_queue.Enqueue(info(action,count,num,status));
            status.first++;
            //if(list.find(num)!=num) cout << "ERROR: Not Found: " << num << endl;
        } else if (action == 'w') {
            // wait until previous operations finish
            // No action will be acted.
        } else if (action == 'p') {     // wait
            
            task_queue.Dequeue(NULL);
        } else {
            printf("ERROR: Unrecognized action: '%c'\n", action);
            exit(EXIT_FAILURE);
        }
	    count++;
    }
    fclose(fin);
    cout << "Done to Enqueue queue!\n";
    for(int i=0; i<num_threads; i++) task_queue.Enqueue(info('p',0,0,{0,0}));
    for(int i=0; i<num_threads; i++) V_thread[i].join();
    

    //for(int i=0; i<num_threads; i++) V_thread[i].join();
    clock_gettime(CLOCK_REALTIME,&stop);

    // print results
    double elapsed_time = (stop.tv_sec - start.tv_sec) + ((double) (stop.tv_nsec - start.tv_nsec))/BILLION ;
    cout << "Elapsed time: " << elapsed_time << " sec" << "\n";
    cout << "Throughput: " << (double) count / elapsed_time << " ops (operations/sec)" << "\n";

    return (EXIT_SUCCESS);
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