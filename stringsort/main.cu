#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda_runtime.h>

using namespace std;

constexpr int MAX_LEN = 30;

void radix_sort_cuda(char input[][MAX_LEN]) {

}

int main(int argc, char* argv[]) {
    char tmpStr[30];
    int i, j, N, pos, range, ret;

    if(argc<5) {
	    cout << "Usage: " << argv[0] << " filename number_of_strings pos range" << endl;
	    return 0;
    }

    ifstream inputfile(argv[1]);

    if(!inputfile.is_open()) {
	    cout << "Unable to open file" << endl;
	    return 0;
    }

    ret=sscanf(argv[2],"%d", &N);
    if(ret==EOF || N<=0) {
	    cout << "Invalid number" << endl;
	    return 0;
    }

    ret=sscanf(argv[3],"%d", &pos);
    if(ret==EOF || pos<0 || pos>=N) {
	    cout << "Invalid position" << endl;
	    return 0;
    }

    ret=sscanf(argv[4],"%d", &range);
    if(ret==EOF || range<0 || (pos+range)>N) {
	    cout << "Invalid range" << endl;
	    return 0;
    }

    auto strArr = new char[N][30];
    memset(strArr,0,sizeof(strArr));

    for(i=0; i<N; i++) inputfile>>strArr[i];

    inputfile.close();

    // Upper Code is the section that get data.

    for(int i=0; i<10; i++) cout << strArr[i] << endl;
    exit(0);
    
    radix_sort_cuda(strArr);

    cout<<"\nStrings (Names) in Alphabetical order from position " << pos << ": " << endl;
    for(i=pos; i<N && i<(pos+range); i++)
        cout<< i << ": " << strArr[i]<<endl;
    cout<<endl;

    delete[] strArr;

    return 0;
}
