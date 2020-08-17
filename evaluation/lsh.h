//
// Created by rafail on 5/29/20.
//

#ifndef __COUNT__
#define __COUNT__

#include "evaluate.h"
#include <string>

#include <lsh/include/psd_lsh.h>

//Uncomment to use random hyperplanes LSH structure
//#include <lsh/include/rh_lsh.h>


/* The data contained in every node of the prefix tree or DFA */
class lsh_data : public evaluation_data {

protected:
    REGISTER_DEC_DATATYPE(lsh_data);

public:
    int num_accepting;
    int num_rejecting;
    int accepting_paths;
    int rejecting_paths;
    map<int, int> occcurences;
    vector<int> counter_1;
    vector<int> counter_2;
    vector<int> counter_3;
    int total_1;
    int total_2;
    int total_3;

//    Uncomment to use random hyperplanes LSH structure
//    lshbox::rhpLsh<int> *mylsh_1;
//    lshbox::rhpLsh<int> *mylsh_2;
//    lshbox::rhpLsh<int> *mylsh_3;
//    lshbox::rhpLsh<int>::Parameter param;

    lshbox::psdLsh<int> *mylsh_1;
    lshbox::psdLsh<int> *mylsh_2;
    lshbox::psdLsh<int> *mylsh_3;
    lshbox::psdLsh<int>::Parameter param;

    lsh_data();

    inline int pos(int i) {
        map<int, int>::iterator it = occcurences.find(i);
        if (it == occcurences.end()) return 0;
        return (*it).second;
    }

    virtual void read_from(int type, int index, int length, int symbol, string data);

//    virtual void read_to(int type, int index, int length, int symbol, string data);

    virtual void print_transition_label(iostream &output, int symbol, apta *aptacontext);

    virtual void print_state_label(iostream &output);

    virtual void update(evaluation_data *right);

    void update_lsh(list <vector<int>> all_labels);

    virtual void undo(evaluation_data *right);


};

class lsh : public evaluation_function {

protected:
    REGISTER_DEC_TYPE(lsh);

public:
    int num_merges;
    int kl_total_score;
    map<string, float> kl_score;
    bool first_merge;


//    virtual void update_score(state_merger *merger, apta_node *left, apta_node *right);

    virtual int compute_score(state_merger *, apta_node *left, apta_node *right);

    virtual void reset(state_merger *merger);

//    virtual bool consistency_check(evaluation_data *l, evaluation_data *r);

    virtual bool consistent(state_merger *merger, apta_node *left, apta_node *right);

};

#endif
