#include <iostream>
#include <sstream>
#include <mutex>
#include <shared_mutex>

#define BILLION 1'000'000'000L

using namespace std;

typedef pair<int,int> pii;

template<class K,class V,int MAXLEVEL> 
class skiplist_node {
public:
    skiplist_node() {
        for(int i=1; i<=MAXLEVEL; i++) {
            forwards[i] = NULL;
        }
    }

    //Constructor with key.
    skiplist_node(K searchKey) : key(searchKey) {
        for(int i=1; i<=MAXLEVEL; i++) {
            forwards[i] = NULL;
        }
    }

    //Constructor with key and value.
    skiplist_node(K searchKey,V val) : key(searchKey),value(val) {
        for(int i=1; i<=MAXLEVEL; i++) {
            forwards[i] = NULL;
        }
    }
 
    virtual ~skiplist_node() {}
 
    K key;
    V value;
    skiplist_node<K,V,MAXLEVEL>* forwards[MAXLEVEL+1];
    recursive_mutex node_lock;
};
 
///////////////////////////////////////////////////////////////////////////////
 
template<class K,class V,int MAXLEVEL=16>
class skiplist {
public:
    typedef K KeyType;
    typedef V ValueType;
    typedef skiplist_node<K,V,MAXLEVEL> NodeType;
    
    skiplist(K minKey,K maxKey):m_pHeader(NULL),m_pTail(NULL),
                                max_curr_level(1),max_level(MAXLEVEL),
                                m_minKey(minKey),m_maxKey(maxKey)
    {
        m_pHeader = new NodeType(m_minKey);
        m_pTail = new NodeType(m_maxKey);
        for (int i=1; i<=MAXLEVEL; i++) {
            m_pHeader->forwards[i] = m_pTail;
        }
    }
    
    virtual ~skiplist() {
        NodeType* currNode = m_pHeader->forwards[1];
        while ( currNode != m_pTail ) {
            NodeType* tempNode = currNode;
            currNode = currNode->forwards[1];
            delete tempNode;
        }
        delete m_pHeader;
        delete m_pTail;
    }
    
    //어차피 모든 write가 된 다음에 하니까 상관 안씀.
    pair<K,K> pair_find(K searchKey) {
        NodeType* currNode = m_pHeader;
        for(int level=max_level; level>=1; level--) {
            while(currNode->forwards[level]->key < searchKey) {
                currNode = currNode->forwards[level];
            }
        }
        currNode = currNode->forwards[1];
        if(currNode->key == searchKey) {
            pair<K,K> ret = {searchKey,INT_MAX};
            if(currNode->forwards[1]) ret.second = currNode->forwards[1]->key;
            return ret;
        } else {
            return {-1,-1};
        }
    }

    void insert(K searchKey,V newValue) {
        while(true) {
            NodeType *previous[MAXLEVEL+1], *follower[MAXLEVEL+1];
            NodeType* currNode = m_pHeader;
            bool same = false;
            
            for(int level=max_level; level>=1; level--) {
                NodeType* nextNode = currNode->forwards[level];
                while(nextNode->key < searchKey) {
                    currNode = nextNode, nextNode = currNode->forwards[level];
                }
                if(nextNode->key == searchKey) same = true;
                previous[level] = currNode;
                follower[level] = nextNode;
            }
            if(same) return ;
            
            int new_level = this->randomLevel();
            int locked_level = 0;
            bool error = false;

            for(int level=1; level<=new_level; level++) {
                if(error) break;
                previous[level]->node_lock.lock();
                locked_level = level;
                if(previous[level]->forwards[level] != follower[level]) error = true;
            }

            if(!error) {
                NodeType* newNode= new NodeType(searchKey,newValue);
                for(int level=1; level<=new_level; level++) {
                    newNode->forwards[level] = follower[level];
                }
                for(int level=1; level<=new_level; level++) {
                    previous[level]->forwards[level] = newNode;
                }
                for(int i=1; i<=locked_level; i++) previous[i]->node_lock.unlock();
                break;
            } else {
                for(int i=1; i<=locked_level; i++) previous[i]->node_lock.unlock();
                continue;
            }
        }
    }
 
    bool empty() const {
        return ( m_pHeader->forwards[1] == m_pTail );
    }
 
    std::string printList() {
	    int i=0;
        std::stringstream sstr;
        NodeType* currNode = m_pHeader->forwards[1];
        while (currNode != m_pTail) {
            //sstr << "(" << currNode->key << "," << currNode->value << ")" << endl;
            sstr << currNode->key << " ";
            currNode = currNode->forwards[1];
	        i++;
	        if(i>200) break;
        }
        return sstr.str();
    }
 
    const int max_level;
 
protected:
    double uniformRandom() {
        return rand() / double(RAND_MAX);
    }
 
    int randomLevel() {
        int level = 1;
        double p = 0.5;
        while (uniformRandom() < p && level < MAXLEVEL ) {
            level++;
        }
        return level;
    }
    K m_minKey;
    K m_maxKey;
    int max_curr_level;
    skiplist_node<K,V,MAXLEVEL>* m_pHeader;
    skiplist_node<K,V,MAXLEVEL>* m_pTail;
};
 
///////////////////////////////////////////////////////////////////////////////

template<class T> class CC_queue {
public:
//structure defined
struct qnode {
    T value;
    qnode* next;
};
//class method defined
CC_queue() {
    qnode* temp = new qnode;
    temp->next = NULL;
    this->head = this->tail = temp;
}
~CC_queue() {
    T trash;
    while (Dequeue(&trash) == 0);
    delete this->head;
}

void Enqueue(T input) {
    qnode *new_node = new qnode;
    new_node->value = input;
    new_node->next = NULL;

    unique_lock<mutex> lk(this->tailLock);
    this->tail->next = new_node;
    this->tail = new_node;
    lk.unlock();
}

int Dequeue(T* value) {
    unique_lock<mutex> lk(this->headLock);
    qnode *temp = this->head;
    qnode *newHead = temp->next;
    if(newHead == NULL) {
        lk.unlock();
        return -1;
    }
    *value = newHead->value;
    this->head = newHead;
    lk.unlock();

    delete(temp);
    return 0;
}

private:
    qnode* head;
    qnode* tail;
    mutex tailLock;
    mutex headLock;
};