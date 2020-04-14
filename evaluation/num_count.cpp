#include "state_merger.h"
#include "evaluate.h"
#include "evaluation_factory.h"
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <map>
#include <stdio.h>
#include <gsl/gsl_cdf.h>

#include "parameters.h"
#include "num_count.h"

REGISTER_DEF_TYPE(count_driven);
REGISTER_DEF_DATATYPE(count_data);

count_data::count_data(){
    num_accepting = 0;
    num_rejecting = 0;
    accepting_paths = 0;
    rejecting_paths = 0;
};

void count_data::print_transition_label(iostream& output, int symbol){
    output << "count data"; 
};

void count_data::print_state_label(iostream& output){
    output << num_accepting;
};


void count_data::read_from(int type, int index, int length, int symbol, string data){
    if(type == 1){
        accepting_paths++;
    } else {
        rejecting_paths++;
    }
};

void count_data::read_to(int type, int index, int length, int symbol, string data){
    if(type == 1){
        if(length == index+1){
            num_accepting++;
        }
    } else {
        if(length == index+1){
            num_rejecting++;
        }
    }
};

void count_data::update(evaluation_data* right){
    count_data* other = reinterpret_cast<count_data*>(right);
    num_accepting += other->num_accepting;
    num_rejecting += other->num_rejecting;
    accepting_paths += other->accepting_paths;
    rejecting_paths += other->rejecting_paths;
};

void count_data::undo(evaluation_data* right){
    count_data* other = reinterpret_cast<count_data*>(right);
    num_accepting -= other->num_accepting;
    num_rejecting -= other->num_rejecting;
    accepting_paths -= other->accepting_paths;
    rejecting_paths -= other->rejecting_paths;
};

bool count_driven::consistency_check(evaluation_data* left, evaluation_data* right){
    count_data* l = reinterpret_cast<count_data*>(left);
    count_data* r = reinterpret_cast<count_data*>(right);
    
    if(l->num_accepting != 0 && r->num_rejecting != 0){ return false; }
    if(l->num_rejecting != 0 && r->num_accepting != 0){ return false; }
    
    return true;
};

/* default evaluation, count number of performed merges */
bool count_driven::consistent(state_merger *merger, apta_node* left, apta_node* right){
    if(inconsistency_found) return false;
  
    count_data* l = (count_data*)left->data;
    count_data* r = (count_data*)right->data;

    if(l->num_accepting != 0 && r->num_rejecting != 0){ inconsistency_found = true; return false; }
    if(l->num_rejecting != 0 && r->num_accepting != 0){ inconsistency_found = true; return false; }
    
    return true;
};

void count_driven::update_score(state_merger *merger, apta_node* left, apta_node* right){
  num_merges += 1;
};

int count_driven::compute_score(state_merger *merger, apta_node* left, apta_node* right){
  return num_merges;
};

void count_driven::reset(state_merger *merger){
  num_merges = 0;
  evaluation_function::reset(merger);
  compute_before_merge=false;
};


// sinks for evaluation data type
bool is_accepting_sink(apta_node* node){
    count_data* d = reinterpret_cast<count_data*>(node->data);

    node = node->find();
    return d->rejecting_paths == 0 && d->num_rejecting == 0;
};

bool is_rejecting_sink(apta_node* node){
    count_data* d = reinterpret_cast<count_data*>(node->data);

    node = node->find();
    return d->accepting_paths == 0 && d->num_accepting == 0;
};

int count_data::sink_type(apta_node* node){
    if(!USE_SINKS) return -1;

    if (is_rejecting_sink(node)) return 0;
    if (is_accepting_sink(node)) return 1;
    return -1;
};

bool count_data::sink_consistent(apta_node* node, int type){
    if(!USE_SINKS) return true;
    
    if(type == 0) return is_rejecting_sink(node);
    if(type == 1) return is_accepting_sink(node);
    
    return true;
};

int count_data::num_sink_types(){
    if(!USE_SINKS) return 0;
    
    // accepting or rejecting
    return 2;
};

