#include "state_merger.h"
#include "mse-error.h"
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <map>
#include <stdio.h>
#include <gsl/gsl_cdf.h>

REGISTER_DEF_DATATYPE(mse_data);
REGISTER_DEF_TYPE(mse_error);

mse_data::mse_data(){
    mean = 0;
    merge_point = -1;
};

void mse_data::read_from(int type, int index, int length, int symbol, string data){
    mean = ((mean * ((double)occs.size())) + occ) / ((double)(occs.size() + 1));
    occs.push_front(occ);
};

void mse_data::update(evaluation_data* right){
    mse_data* r = (mse_data*) right;
    
    if(occs.size() != 0 && r->occs.size() != 0)
        mean = ((mean * ((double)occs.size()) + (r->mean * ((double)r->occs.size())))) / ((double)occs.size() + (double)r->occs.size());

    r->merge_point = occs.end();
    --(r->merge_point);
    occs.splice(occs.end(), r->occs);
    ++(r->merge_point);
};

void mse_data::undo(evaluation_data* right){
    mse_data* r = (mse_data*) right;

    r->occs.splice(r->occs.begin(), occs, r->occ_merge_point, occs.end());
    
    if(occs.size() != 0 && r->occs.size() != 0)
        mean = ((mean * ((double)occs.size() + (double)r->occs.size())) - (r->mean * ((double)r->occs.size()))) / ((double)occs.size());
};

bool mse_error::consistent(state_merger *merger, apta_node* left, apta_node* right){
    if(evaluation_function::consistent(merger, left, right) == false){ inconsistency_found = true; return false; }
    mse_data* l = (mse_data*) left->data;
    mse_data* r = (mse_data*) right->data;

    if(left->size < STATE_COUNT || right->size < STATE_COUNT) return true;
    
    if(l->mean - r->mean > CHECK_PARAMETER){ inconsistency_found = true; return false; }
    if(r->mean - l->mean > CHECK_PARAMETER){ inconsistency_found = true; return false; }
    
    return true;
};

void mse_error::update_score(state_merger *merger, apta_node* left, apta_node* right){
    mse_data* l = (mse_data*) left->data;
    mse_data* r = (mse_data*) right->data;

    if(l->accepting_paths < STATE_COUNT || r->accepting_paths < STATE_COUNT) return;
    
    num_merges = num_merges + 1;
    num_points = num_points + l->occs.size() + r->occs.size();

    double error_left = 0.0;
    double error_right = 0.0;
    double error_total = 0.0;
    double mean_total = 0.0;
    
    mean_total = ((l->mean * ((double)l->occs.size()) + (r->mean * ((double)r->occs.size())))) / ((double)l->occs.size() + (double)r->occs.size());
    
    for(double_list::iterator it = l->occs.begin(); it != l->occs.end(); ++it){
        error_left  = error_left  + ((l->mean    - (double)*it)*(l->mean    - (double)*it));
        error_total = error_total + ((mean_total - (double)*it)*(mean_total - (double)*it));
    }
    for(double_list::iterator it = r->occs.begin(); it != r->occs.end(); ++it){
        error_right = error_right + ((r->mean    - (double)*it)*(r->mean    - (double)*it));
        error_total = error_total + ((mean_total - (double)*it)*(mean_total - (double)*it));
    }
    
    //error_right = error_right / ((double)r->occs.size());
    //error_left  = error_left / ((double)l->occs.size());
    //error_total = (error_total) / ((double)l->occs.size() + (double)r->occs.size());
    
    RSS_before += error_right+error_left;
    RSS_after += error_total;
};

int mse_error::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    return 2*num_merges + num_points*(log(RSS_before)/log(RSS_after));
};

void mse_error::reset(state_merger *merger ){
    inconsistency_found = false;
    num_merges = 0.0;
    num_points = 0.0;
    RSS_before = 0.0;
    RSS_after = 0.0;
};

void mse_error::print_dot(FILE* output, state_merger* merger){
    apta* aut = merger->aut;
    state_set s  = merger->red_states;
    
    cerr << "size: " << s.size() << endl;
    
    fprintf(output,"digraph DFA {\n");
    fprintf(output,"\t%i [label=\"root\" shape=box];\n", aut->root->find()->number);
    fprintf(output,"\t\tI -> %i;\n", aut->root->find()->number);
    for(state_set::iterator it = merger->red_states.begin(); it != merger->red_states.end(); ++it){
        apta_node* n = *it;
        
        double error = 0.0;
        double mean = n->mean;
        
        fprintf(output,"\t%i [shape=circle label=\"\n%.3f\n%.3f\n%i\"];\n", n->number, mean, error, (int)n->occs.size());
        
        state_set childnodes;
        set<int> sinks;
        for(int i = 0; i < alphabet_size; ++i){
            apta_node* child = n->get_child(i);
            if(child == 0){
                // no output
            } else {
                 if(sink_type(child) != -1){
                     sinks.insert(sink_type(child));
                 } else {
                     childnodes.insert(child);
                 }
            }
        }
        for(set<int>::iterator it2 = sinks.begin(); it2 != sinks.end(); ++it2){
            int stype = *it2;
            fprintf(output,"\tS%it%i [label=\"sink %i\" shape=box];\n", n->number, stype, stype);
            fprintf(output, "\t\t%i -> S%it%i [label=\"" ,n->number, n->number, stype);
            for(int i = 0; i < alphabet_size; ++i){
                if(n->get_child(i) != 0 && sink_type(n->get_child(i)) == stype){
                    fprintf(output, " %s [%i:%i]", aut->alph_str(i).c_str(), n->num_pos[i], n->num_neg[i]);
                }
            }
            fprintf(output, "\"];\n");
        }
        for(state_set::iterator it2 = childnodes.begin(); it2 != childnodes.end(); ++it2){
            apta_node* child = *it2;
            fprintf(output, "\t\t%i -> %i [label=\"" ,n->number, child->number);
            for(int i = 0; i < alphabet_size; ++i){
                if(n->get_child(i) != 0 && n->get_child(i) == child){
                    fprintf(output, " %s [%i:%i]", aut->alph_str(i).c_str(), n->num_pos[i], n->num_neg[i]);
                }
            }
            fprintf(output, "\"];\n");
        }
    }
    fprintf(output,"}\n");
};

