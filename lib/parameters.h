#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include <string>
#include <vector>

using namespace std;

extern int alphabet_size;
extern bool MERGE_SINKS;
extern int STATE_COUNT;
extern int SYMBOL_COUNT;
extern int SINK_COUNT;
extern float CORRECTION;
extern float CHECK_PARAMETER;
extern bool USE_SINKS;
extern bool USE_LOWER_BOUND;
extern float LOWER_BOUND;
extern int alphabet_size;
extern int GREEDY_METHOD;
extern int APTA_BOUND;
extern int CLIQUE_BOUND;
extern bool EXTEND_ANY_RED;
extern bool MERGE_SINKS_PRESOLVE;
extern int OFFSET;
extern int EXTRA_STATES;
extern bool TARGET_REJECTING;
extern bool SYMMETRY_BREAKING;
extern bool FORCING;
extern bool MERGE_SINKS_DSOLVE;
extern string eval_string;
extern bool MERGE_MOST_VISITED;
extern bool MERGE_BLUE_BLUE;
extern bool RED_FIXED;
extern bool MERGE_WHEN_TESTING;
extern bool DEPTH_FIRST;
extern int RANGE;
extern int STORE_MERGES;
extern int STORE_MERGES_KEEP_CONFLICT;
extern int STORE_MERGES_SIZE_THRESHOLD;
extern double STORE_MERGES_RATIO_THRESHOLD;
extern string COMMAND;
extern int STREAM_COUNT;
extern bool EXCEPTION4OVERLAP;
extern string EVALPAR;
extern float LSH_KL_THRESHOLD;
extern int LSH_M;
extern int LSH_L;
extern int LSH_D;
extern int LSH_N;
extern int LSH_C;
extern int LSH_T;
extern int LSH_W;

class parameters {
public:
    string command;
    string dfa_file;
    vector <string> dfa_data;
    string dot_file;
    string sat_program;
    string hName;
    string hData;
    int runs;
    int sinkson;
    int seed;
    int sataptabound;
    int satdfabound;
    float lower_bound;
    int satextra;
    int mergesinks;
    int satmergesinks;
    int method;
    int extend;
    int heuristic;
    int symbol_count;
    int state_count;
    int sink_count;
    float correction;
    float extrapar;
    int satplus;
    int satfinalred;
    int symmetry;
    int forcing;
    int blueblue;
    int finalred;
    int largestblue;
    int testmerge;
    int shallowfirst;
    string mode;
    string evalpar;
    int batchsize;
    float delta;
    float epsilon;
    int debugging;
    float klthreshold;
    int lsh_m = 1000;
    int lsh_l = 1;
    int lsh_d = 20;
    int lsh_n = 100;
    int lsh_c = 8;
    int lsh_t = 2;
    int lsh_w = lsh_d;

    parameters();


};

void init_with_params(parameters *);

#endif
