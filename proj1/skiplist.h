#include <iostream>
#include <sstream>

#define BILLION 1'000'000'000L

using namespace std;

/*
0 ~ 2,147,483,647
대충 쪼개볼가요?
21474
그 이상에는 
*/
template<class K,class V,int MAXLEVEL> 
class skiplist_node {
public:
    /*
    K key;
    V value;
    skiplist_node<K,V,MAXLEVEL>* forwards[MAXLEVEL+1];
    */
    //Constructor with nothing
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
};
 
///////////////////////////////////////////////////////////////////////////////
 
template<class K,class V,int MAXLEVEL=16>
class skiplist {
public:
    /*
    K m_minKey;
    K m_maxKey;
    int max_curr_level;
    skiplist_node<K,V,MAXLEVEL>* m_pHeader;
    skiplist_node<K,V,MAXLEVEL>* m_pTail;
    */
    typedef K KeyType;
    typedef V ValueType;
    typedef skiplist_node<K,V,MAXLEVEL> NodeType;
    
    //저희는 skiplist<int,int> list(0,INT_MAX)를 사용합니다.
    //그렇기 때문에? Header는 0이고 Tail은 INT_MAX의 값을 가질 것입니다.
    //그리고 node에는 forward밖에 없습니다.
    //처음이니까 모든 높이에 대해서 head -> tail을 해 놓습니다.
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
    
    //Destructor in skiplist
    //1이 제일 낮은 레벨인 것 같네요?
    //1은 모든 node를 포함하고 있습니다.
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
    
    //Key와 Value의 차이는 무엇인가?
    //나도 잘 모르겠습니다.
    //대충 각 레벨별로 얘가 들어갈 위치 바로 왼쪽을 pointer의 형태로 저장한 것이라 보면 될 것 같음.
    
    void insert(K searchKey,V newValue) {
        skiplist_node<K,V,MAXLEVEL>* update[MAXLEVEL]; //포인터를 잠시 담아두는 배열, 실제 메모리를 차지하는 것은 아님.
        NodeType* currNode = m_pHeader;
        for(int level=max_curr_level; level >=1; level--) {
            while ( currNode->forwards[level]->key < searchKey ) {
                currNode = currNode->forwards[level];
            }
            update[level] = currNode;
        } //전체적으로 레벨들을 순회하면서 직전의 포인터들을 저장합니다.
        currNode = currNode->forwards[1]; //

        if ( currNode->key == searchKey ) {
            currNode->value = newValue;
        } //KICK IT
        else {
            int newlevel = randomLevel();
            if ( newlevel > max_curr_level ) {
                for ( int level = max_curr_level+1; level <= newlevel; level++ ) {
                    update[level] = m_pHeader;
                }
                max_curr_level = newlevel;
            }
            currNode = new NodeType(searchKey,newValue);
            for ( int lv=1; lv<=max_curr_level; lv++ ) {
                currNode->forwards[lv] = update[lv]->forwards[lv];
                update[lv]->forwards[lv] = currNode;
            }
        }
    }
 
    void erase(K searchKey)
    {
        skiplist_node<K,V,MAXLEVEL>* update[MAXLEVEL];
        NodeType* currNode = m_pHeader;
        for(int level=max_curr_level; level >=1; level--) {
            while ( currNode->forwards[level]->key < searchKey ) {
                currNode = currNode->forwards[level];
            }
            update[level] = currNode;
        }
        currNode = currNode->forwards[1];
        if ( currNode->key == searchKey ) {
            for ( int lv = 1; lv <= max_curr_level; lv++ ) {
                if ( update[lv]->forwards[lv] != currNode ) {
                    break;
                }
                update[lv]->forwards[lv] = currNode->forwards[lv];
            }
            delete currNode;
            // update the max level
            while ( max_curr_level > 1 && m_pHeader->forwards[max_curr_level] == NULL ) {
                max_curr_level--;
            }
        }
    }
 
    //const NodeType* find(K searchKey)
    V find(K searchKey)
    {
        NodeType* currNode = m_pHeader;
        for(int level=max_curr_level; level >=1; level--) {
            while ( currNode->forwards[level]->key < searchKey ) {
                currNode = currNode->forwards[level];
            }
        }
        currNode = currNode->forwards[1];
        if ( currNode->key == searchKey ) {
            return currNode->value;
        }
        else {
            //return NULL;
            return -1;
        }
    }
 
    bool empty() const
    {
        return ( m_pHeader->forwards[1] == m_pTail );
    }
 
    std::string printList()
    {
	int i=0;
        std::stringstream sstr;
        NodeType* currNode = m_pHeader->forwards[1];
        while ( currNode != m_pTail ) {
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
    double uniformRandom()
    {
        return rand() / double(RAND_MAX);
    }
 
    int randomLevel() {
        int level = 1;
        double p = 0.5;
        while ( uniformRandom() < p && level < MAXLEVEL ) {
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