#include <stdlib.h>
#include <math.h>
#include <vector>
#include <map>
#include <stdio.h>
#include <gsl/gsl_cdf.h>

#include "state_merger.h"
#include "evaluate.h"
#include "likelihood.h"
#include "parameters.h"

REGISTER_DEF_DATATYPE(likelihood_data);
REGISTER_DEF_TYPE(likelihoodratio);

bool likelihoodratio::consistent(state_merger *merger, apta_node* left, apta_node* right){
    likelihood_data* l = (likelihood_data*) left->data;
    likelihood_data* r = (likelihood_data*) right->data;

    //if(left->depth != right->depth) {inconsistency_found = true; return false;};

    //if(l->num_accepting != 0 && r->num_accepting == 0) {inconsistency_found = true; return false;};
    //if(l->num_accepting == 0 && r->num_accepting != 0) {inconsistency_found = true; return false;};

    return count_driven::consistent(merger, left, right);
};

void likelihoodratio::update_likelihood(double left_count, double right_count, double left_divider, double right_divider){    
    if(left_count != 0.0)
        loglikelihood_orig += (left_count + CORRECTION)  * log((left_count + CORRECTION)  / left_divider);
    if(right_count != 0.0)
        loglikelihood_orig += (right_count + CORRECTION) * log((right_count + CORRECTION) / right_divider);
    if(right_count != 0.0 || left_count != 0.0)
        loglikelihood_merged += (left_count + right_count + 2*CORRECTION) * log((left_count + right_count + 2*CORRECTION) / (left_divider + right_divider));
    if(right_count != 0.0 && left_count != 0.0)
        extra_parameters = extra_parameters + 1;
};

/* Likelihood Ratio (LR), computes an LR-test (used in RTI) and uses the p-value as score and consistency */
void likelihoodratio::update_score(state_merger *merger, apta_node* left, apta_node* right){
    likelihood_data* l = (likelihood_data*) left->data;
    likelihood_data* r = (likelihood_data*) right->data;
    
    CORRECTION = 0.0;

    if(r->accepting_paths < STATE_COUNT || l->accepting_paths < STATE_COUNT) return;
    
    double left_divider = 1.0;
    double right_divider = 1.0;
    double left_count = 0.0;
    double right_count  = 0.0;

    for(int a = 0; a < alphabet_size; ++a){
        if(l->num_pos[a] >= SYMBOL_COUNT || r->num_pos[a] >= SYMBOL_COUNT){
            left_divider += CORRECTION;
            right_divider += CORRECTION;
        }
    }

    left_divider += (double)l->accepting_paths;// + (double)l->num_accepting;
    right_divider += (double)r->accepting_paths;// + (double)r->num_accepting;
    
    int l1_pool = 0;
    int r1_pool = 0;
    int l2_pool = 0;
    int r2_pool = 0;
    int matching_right = 0;

    for(num_map::iterator it = l->num_pos.begin(); it != l->num_pos.end(); ++it){
        left_count = (*it).second;
        right_count = r->pos((*it).first);
        matching_right += right_count;
        
        if(left_count >= SYMBOL_COUNT && right_count >= SYMBOL_COUNT)
            update_likelihood(left_count, right_count, left_divider, right_divider);

        if(right_count < SYMBOL_COUNT){
            l1_pool += left_count;
            r1_pool += right_count;
        }
        if(left_count < SYMBOL_COUNT) {
            l2_pool += left_count;
            r2_pool += right_count;
        }
    }
    
    r2_pool += r->accepting_paths - matching_right;
    
    left_count = l1_pool;
    right_count = r1_pool;
    
    if(right_count >= SYMBOL_COUNT || right_count >= SYMBOL_COUNT)
        update_likelihood(left_count, right_count, left_divider, right_divider);
    
    left_count = l2_pool;
    right_count = r2_pool;
    
    if(right_count >= SYMBOL_COUNT || right_count >= SYMBOL_COUNT)
        update_likelihood(left_count, right_count, left_divider, right_divider);

    //left_count = (double)l->num_accepting;
    //right_count = (double)r->num_accepting;
    
    //if(right_count >= SYMBOL_COUNT || right_count >= SYMBOL_COUNT)
    //    update_likelihood(left_count, right_count, left_divider, right_divider);
};

bool likelihoodratio::compute_consistency(state_merger *merger, apta_node* left, apta_node* right){
    if (inconsistency_found) return false;

    double test_statistic = 2.0 * (loglikelihood_orig - loglikelihood_merged);
    double p_value = gsl_cdf_chisq_Q (test_statistic, 1.0 + (double)extra_parameters);
    
    if (p_value < CHECK_PARAMETER) { return false; }

    return true;
};

int likelihoodratio::compute_score(state_merger *merger, apta_node* left, apta_node* right){
    if (inconsistency_found) return -1;

    double test_statistic = 2.0 * (loglikelihood_orig - loglikelihood_merged);
    double p_value = gsl_cdf_chisq_Q (test_statistic, (double)extra_parameters);

    return (int)(p_value * 10000000.0);
};

void likelihoodratio::reset(state_merger *merger){
    inconsistency_found = false;
    loglikelihood_orig = 0;
    loglikelihood_merged = 0;
    extra_parameters = 0;
};


/*void likelihoodratio::print_dot(iostream& output, state_merger* merger){
    alergia::print_dot(output, merger);
};*/
