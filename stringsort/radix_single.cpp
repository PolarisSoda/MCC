#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>

using namespace std;

constexpr int MAX_LEN = 64;
constexpr int CHAR_RANGE = 122 - 64 + 1;
constexpr int NUM_THREADS = 512;
//65 ~ 122

void kernel_function(char* device_input, char* device_output, int N) {
    //N is total amount of string.
    //we have NUM_THREADS 512
    //each THREAD HAVE 196 strings.

    int histogram[CHAR_RANGE];
    int offset[CHAR_RANGE];
    int count[CHAR_RANGE];

    for(int pos=MAX_LEN-1; pos>=0; pos--) {
        memset(histogram,0,sizeof(histogram));
        memset(count,0,sizeof(count));

        for(int i=0; i<N; i++) {
            char now = device_input[i*MAX_LEN + pos];
            histogram[now-64]++;
        }

        offset[0] = 0;
        for(int i=0; i<CHAR_RANGE-1; i++) offset[i+1] = offset[i] + histogram[i];
        
        for(int i=0; i<N; i++) {
            char now = device_input[i*MAX_LEN + pos];
            int index = now-64;
            int after_index = offset[index] + count[index]++;
            for(int j=0; j<MAX_LEN; j++) device_output[after_index*MAX_LEN + j] = device_input[i*MAX_LEN + j];
        }

        /*
        for(int i=0; i<N; i++) {
            for(int j=0; j<MAX_LEN; j++) {
                char now = device_output[i*MAX_LEN+j];
                cout << now;
            }
            cout << endl;
        }
        cout << endl;
        */

        char* temp = device_input;
        device_input = device_output;
        device_output = temp;
    }
}


void radix_sort_cuda(char* host_input, char* host_output, int N) {
    // First we have to copy these data to device.
    size_t data_size = N * MAX_LEN * sizeof(char);

    kernel_function(host_input,host_output,N);
}

int main(int argc, char* argv[]) {
    int N, pos, range, ret;

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

    auto strArr = new char[N*MAX_LEN];
    auto output = new char[N*MAX_LEN];

    memset(strArr, 64, N * MAX_LEN);
    for (int i=0; i<N; i++) {
        inputfile >> std::setw(MAX_LEN) >> &strArr[i * MAX_LEN];
        int length = strlen(&strArr[i*MAX_LEN]);
        strArr[i*MAX_LEN + length] = 64;
    }
    inputfile.close();

    // Upper Code is the section that get data.
    radix_sort_cuda(strArr,output,N);

    cout << "\nStrings (Names) in Alphabetical order from position " << pos << ": " << "\n";
    for(int i=pos; i<N && i<(pos+range); i++) {
        cout << i << ": ";
        for(int j=0; j<MAX_LEN; j++) {
            char now = strArr[i*MAX_LEN+j];
            if(now != '@') cout << now;
        }
        cout << endl;
    }
        
    cout << "\n";

    delete[] strArr;

    return 0;
}
