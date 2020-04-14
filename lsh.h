#ifndef __COUNT__
#define __COUNT__

#include "evaluate.h"
#include <string>
#include <lsh/include/rh_lsh.h>
//#include <lsh/include/rbs_lsh.h>


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
    vector<float> dist;
    lshbox::rhpLsh<int> *mylsh;
//    lshbox::rbsLsh<int> *mylsh;
    lshbox::rhpLsh<int>::Parameter param;
//    lshbox::rbsLsh<int>::Parameter param;

    lsh_data();

    inline int pos(int i){
        map<int, int>::iterator it = occcurences.find(i);
        if(it == occcurences.end()) return 0;
        return (*it).second;
    }

    virtual void read_from(int type, int index, int length, int symbol, string data);

    virtual void read_to(int type, int index, int length, int symbol, string data);

    virtual void print_transition_label(iostream &output, int symbol, apta *aptacontext);

    virtual void print_state_label(iostream &output);

    virtual void update(evaluation_data *right);

    void update_lsh(list <vector<int>> all_labels);

    virtual void undo(evaluation_data *right);

//    virtual int sink_type(apta_node *node);
//
//    virtual bool sink_consistent(apta_node *node, int type);
//
//    virtual int num_sink_types();
//
//    virtual bool is_accepting_sink(apta_node *node);
//
//    virtual bool is_rejecting_sink(apta_node *node);


};

class lsh : public evaluation_function {

protected:
    REGISTER_DEC_TYPE(lsh);

public:
    int num_merges;
    int kl_score;
//    int extra_parameters;
//    double perplexity;


    virtual void update_score(state_merger *merger, apta_node *left, apta_node *right);

    virtual int compute_score(state_merger *, apta_node *left, apta_node *right);

    virtual void reset(state_merger *merger);

    virtual bool consistency_check(evaluation_data *l, evaluation_data *r);

    virtual bool consistent(state_merger *merger, apta_node *left, apta_node *right);

//    virtual bool compute_consistency(state_merger *merger, apta_node *left, apta_node *right);


        //virtual void print_dot(iostream&, state_merger *);
};

#endif
