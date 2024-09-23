#include <cstdlib>
#include <pthread.h>

template <typename T> 
struct array { 
  size_t x; 
  T *ary; 
}; 

struct node {
    char a;
    int bb[4];
    node* next;
};

struct cc_queue {
    node* head;
    node* tail;
    pthread_mutex_t headLock;
    pthread_mutex_t tailLock;
};

void Queue_Init(cc_queue* q) {
    node *temp = new node;
    temp->next = NULL;
    q->head = q->tail = temp;
    pthread_mutex_init(&q->headLock,NULL);
    pthread_mutex_init(&q->tailLock,NULL);
}

void Queue_Enqueue(cc_queue* q,node value) {
    node *temp = new node;
    *temp = value;
    temp->next = NULL;
    
    pthread_mutex_lock(&q->tailLock);
    q->tail->next = temp;
    q->tail = temp;
    pthread_mutex_unlock(&q->tailLock);
}

int Queue_Dequeue(cc_queue* q,node* value) {
    pthread_mutex_lock(&q->headLock);
    node *temp = q->head;
    node *newHead = temp->next;
    if(newHead == NULL) {
        pthread_mutex_unlock(&q->headLock);
        return -1;
    }
    value = temp;
    q->head = newHead;
    pthread_mutex_unlock(&q->headLock);
    delete(temp);
    return 0;
}