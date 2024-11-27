#include <bits/stdc++.h>

using namespace std;

vector<string> V;

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

    for (int i=0; i<N; i++) {
        string temp;
        inputfile >> temp;
        V.push_back(temp);
    }
    inputfile.close();
    sort(V.begin(),V.end());

    cout << "\nStrings (Names) in Alphabetical order from position " << pos << ": " << "\n";

    for(int i=pos; i<N && i<(pos+range); i++) {
        cout << i << ": ";
        cout << V[i];
        cout << "\n";
    }
    cout << "\n";
    return 0;
}
