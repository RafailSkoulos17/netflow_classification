#include "parameters.h"

int alphabet_size = 0;
bool MERGE_SINKS = 0;
int STATE_COUNT = 0;
int SYMBOL_COUNT = 0;
int SINK_COUNT = 0;
float CORRECTION = 0.0;
float CHECK_PARAMETER = 0.0;
bool USE_SINKS = 0;
float MINIMUM_SCORE = 0;
bool USE_LOWER_BOUND = 0;
float LOWER_BOUND = 0;
int GREEDY_METHOD = 0;
int APTA_BOUND = 0;
int CLIQUE_BOUND = 0;
bool EXTEND_ANY_RED = 0;
bool MERGE_SINKS_PRESOLVE = 0;
bool MERGE_SINKS_DSOLVE = 0;
int OFFSET = 1;
int EXTRA_STATES = 0;
bool TARGET_REJECTING = 0;
bool SYMMETRY_BREAKING = 0;
bool FORCING = 0;
bool MERGE_MOST_VISITED = 0;
bool MERGE_BLUE_BLUE = 0;
bool RED_FIXED = 0;
bool MERGE_WHEN_TESTING = 0;
bool DEPTH_FIRST = 0;
float LSH_KL_THRESHOLD = 0;
int LSH_M = 0;
int LSH_L = 0;
int LSH_D = 0;
int LSH_N = 0;
int LSH_C = 0;


bool EXCEPTION4OVERLAP = false;

int RANGE = 100;
string eval_string;

int STORE_MERGES = 0;
int STORE_MERGES_KEEP_CONFLICT = 1;
double STORE_MERGES_RATIO_THRESHOLD = 0.3;
int STORE_MERGES_SIZE_THRESHOLD = 100;

parameters::parameters() {
    command = string("");
    batchsize = 1000;
    epsilon = 0.3;
    delta = 0.95;
    mode = "batch";
    evalpar = "";
    dot_file = "dfa";
    sat_program = "";
    hName = "default";
    hData = "evaluation_data";
    runs = 1;
    sinkson = 1;
    seed = 12345678;
    sataptabound = 0;
    satdfabound = 50;
    mergesinks = 1;
    satmergesinks = 0;
    lower_bound = -1.0;
    satextra = 5;
    method = 1;
    heuristic = 1;
    extend = 1;
    symbol_count = 10;
    state_count = 25;
    correction = 0.0;
    extrapar = 0.5;
    satplus = 0;
    satfinalred = 0;
    symmetry = 1;
    forcing = 0;
    blueblue = 0;
    finalred = 0;
    largestblue = 0;
    testmerge = 0;
    shallowfirst = 0;
    debugging = 0;
    evalpar = "";
    klthreshold = 0.4;
    lsh_m = 1000;
    lsh_l = 1;
    lsh_d = 20;
    lsh_n = 100;
    lsh_c = 18;


};
