//
// Created by rafail on 5/29/20.
//

#include "state_merger.h"
#include "evaluate.h"
#include "evaluation_factory.h"
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <map>
#include <stdio.h>
#include <gsl/gsl_cdf.h>
#include <string>
#include "parameters.h"
#include "lsh.h"
#include <lsh/include/psd_lsh.h>
//#include <lsh/include/rh_lsh.h>
//#include <lsh/include/rbs_lsh.h>
#include<cmath>
#include <tuple>

REGISTER_DEF_TYPE(lsh);
REGISTER_DEF_DATATYPE(lsh_data);

lsh_data::lsh_data() {
    occcurences = map<int, int>();

//    Uncomment to use random hyperplanes LSH structure
//    mylsh_1 = new lshbox::rhpLsh<int>();
//    mylsh_2 = new lshbox::rhpLsh<int>();
//    mylsh_3 = new lshbox::rhpLsh<int>();
//    lshbox::rhpLsh<int>::Parameter param;
//
    mylsh_1 = new lshbox::psdLsh<int>();
    mylsh_2 = new lshbox::psdLsh<int>();
    mylsh_3 = new lshbox::psdLsh<int>();
    lshbox::psdLsh<int>::Parameter param;

    int total_1 = 0;
    int total_2 = 0;
    int total_3 = 0;

    vector<int> counter_1(LSH_D);
    vector<int> counter_2(LSH_D);
    vector<int> counter_3(LSH_D);

    param.M = LSH_M; // Hash table size
    param.L = LSH_L; // Number of hash tables
    param.D = LSH_D; // Dimension of the vector
    param.T = LSH_T; // Index mode, you can choose 1(CAUCHY) or 2(GAUSSIAN)
    param.W = LSH_W; // Window size

//    Uncomment to use random hyperplanes LSH structure
//    param.N = LSH_N; // Binary code bytes
    mylsh_1->reset(param);
    mylsh_2->reset(param);
    mylsh_3->reset(param);
};

//normalize by adding  0.01 before dividing by sum
float kl_divergence(vector<float> dist1, vector<float> dist2) {
    float kl_score = 0;
    for (int i = 0; i < dist1.size(); ++i) {
        if (dist1[i] != 0 && dist2[i] != 0)
            kl_score += dist1[i] * log((float) dist1[i] / (float) dist2[i]);
        else if (dist1[i] != 0 && dist2[i] == 0)
            kl_score += dist1[i] * log((float) dist1[i] / 0.01);
    }
    return kl_score;
}


void lsh_data::print_transition_label(iostream &output, int symbol, apta *aptacontext) {
    output << occcurences[symbol];
    if (occcurences[symbol] == 0) {
        cout << "STOP" << endl;
    }
};

void lsh_data::print_state_label(iostream &output) {
    output << "count data";
};


void lsh_data::read_from(int type, int index, int length, int symbol, string data) {
    occcurences[symbol] = pos(symbol) + 1;
};

// Extracts discrete distribution from the LSH buckets (table)
tuple<vector<int>, int> get_distribution(std::vector < std::map < unsigned, std::vector < std::string >> >
                                                                              tables) {
    vector<int> dist(LSH_D);
    int total = 0;
    for (int t = 0; t != dist.size(); ++t) {
        for (auto it = tables.begin(); it != tables.end(); ++it) {
            std::map<unsigned, std::vector<std::string>> table = *it;
            if (table.find(t) != table.end()) {
                dist[t] = (table[t]).size() + 1;
                total += (table[t]).size() + 1;
            } else {
//              apply normalization
                dist[t] = 1;
                total += 1;
            }
        }
    }
    return make_tuple(dist, total);
};

// Used to update the LSH buckets while reading the APTA
void lsh_data::update_lsh(list <vector<int>> all_labels) {

    for (auto p = all_labels.begin(); p != all_labels.end(); ++p) {
        vector<int> &label_list = *p;
        if (label_list.size() <= 3) {
            mylsh_1->hash(label_list);
        } else if (label_list.size() <= 10) {
            mylsh_2->hash(label_list);
        } else if (label_list.size() >= LSH_D) {
            vector<int> b(label_list.begin(), label_list.begin() + LSH_D);
            mylsh_3->hash(b);
        } else {
            mylsh_3->hash(label_list);
        }
    }

    tie(counter_1, total_1) = get_distribution(mylsh_1->tables);
    tie(counter_2, total_2) = get_distribution(mylsh_2->tables);
    tie(counter_3, total_3) = get_distribution(mylsh_3->tables);

};

//Updates the data of a node when merged during the state-merging process
void lsh_data::update(evaluation_data *right) {
    lsh_data * other = reinterpret_cast<lsh_data *>(right);

    for (map<int, int>::iterator it = other->occcurences.begin(); it != other->occcurences.end(); ++it) {
        occcurences[(*it).first] = pos((*it).first) + (*it).second;
    }

    for (int i = 0; i != counter_1.size(); ++i) {
        counter_1[i] += other->counter_1[i];
        total_1 += other->counter_1[i];
    }
    for (int i = 0; i != counter_2.size(); ++i) {
        counter_2[i] += other->counter_2[i];
        total_2 += other->counter_2[i];
    }
    for (int i = 0; i != counter_3.size(); ++i) {
        counter_3[i] += other->counter_3[i];
        total_3 += other->counter_3[i];
    }
};

//Undo the changes in the data of a node when a merge is canceled
void lsh_data::undo(evaluation_data *right) {
    lsh_data * other = reinterpret_cast<lsh_data *>(right);
    for (map<int, int>::iterator it = other->occcurences.begin(); it != other->occcurences.end(); ++it) {
        occcurences[(*it).first] = pos((*it).first) - (*it).second;
    }

    for (int i = 0; i != counter_1.size(); ++i) {
        counter_1[i] -= other->counter_1[i];
        total_1 -= other->counter_1[i];
    }
    for (int i = 0; i != counter_2.size(); ++i) {
        counter_2[i] -= other->counter_2[i];
        total_2 -= other->counter_2[i];
    }
    for (int i = 0; i != counter_3.size(); ++i) {
        counter_3[i] -= other->counter_3[i];
        total_3 -= other->counter_3[i];
    }
};

// Checks if a merge is consistent by comparing the KL divergence of the 3 LSH
// structures with a user-defined threshold
bool lsh::consistent(state_merger *merger, apta_node *left, apta_node *right) {

    if (inconsistency_found) return false;
    if (!first_merge) return true;
    first_merge = false;

    lsh_data * l = (lsh_data *) left->data;
    lsh_data * r = (lsh_data *) right->data;


    vector<float> ldist_1;
    vector<float> rdist_1;
    for (int i = 0; i != (l->counter_1).size(); ++i) {
        ldist_1.push_back(((float) l->counter_1[i]) / ((float) l->total_1));
        rdist_1.push_back(((float) r->counter_1[i]) / ((float) r->total_1));
    }

    vector<float> ldist_2;
    vector<float> rdist_2;
    for (int i = 0; i != (l->counter_2).size(); ++i) {
        ldist_2.push_back(((float) l->counter_2[i]) / ((float) l->total_2));
        rdist_2.push_back(((float) r->counter_2[i]) / ((float) r->total_2));
    }


    vector<float> ldist_3;
    vector<float> rdist_3;
    for (int i = 0; i != (l->counter_3).size(); ++i) {
        ldist_3.push_back(((float) l->counter_3[i]) / ((float) l->total_3));
        rdist_3.push_back(((float) r->counter_3[i]) / ((float) r->total_3));
    }
    float kl_div_1 = kl_divergence(ldist_1, rdist_1);
    float kl_div_2 = kl_divergence(ldist_2, rdist_2);
    float kl_div_3 = kl_divergence(ldist_2, rdist_2);
    if (kl_div_1 > LSH_KL_THRESHOLD || kl_div_2 > LSH_KL_THRESHOLD || kl_div_3 > LSH_KL_THRESHOLD) return false;
    else return true;
};


// Computes the score of a merge
int lsh::compute_score(state_merger *merger, apta_node *left, apta_node *right) {
//    return num_merges;
    if (first_merge) {
        int l_total, r_total;
        lsh_data * l = (lsh_data *) left->data;
        lsh_data * r = (lsh_data *) right->data;

        vector<float> ldist_1;
        vector<float> rdist_1;
        for (int i = 0; i != (l->counter_1).size(); ++i) {
            ldist_1.push_back(((float) l->counter_1[i]) / ((float) l->total_1));
            rdist_1.push_back(((float) r->counter_1[i]) / ((float) r->total_1));
        }

        vector<float> ldist_2;
        vector<float> rdist_2;
        for (int i = 0; i != (l->counter_2).size(); ++i) {
            ldist_2.push_back(((float) l->counter_2[i]) / ((float) l->total_2));
            rdist_2.push_back(((float) r->counter_2[i]) / ((float) r->total_2));
        }


        vector<float> ldist_3;
        vector<float> rdist_3;
        for (int i = 0; i != (l->counter_3).size(); ++i) {
            ldist_3.push_back(((float) l->counter_3[i]) / ((float) l->total_3));
            rdist_3.push_back(((float) r->counter_3[i]) / ((float) r->total_3));
        }
        float kl_div_1 = kl_divergence(ldist_1, rdist_1);
        float kl_div_2 = kl_divergence(ldist_2, rdist_2);
        float kl_div_3 = kl_divergence(ldist_2, rdist_2);

        kl_total_score = (int) (100000 - (kl_div_1 * 100000.0) + 100000 - (kl_div_2 * 100000.0) + 100000 -
                                (kl_div_3 * 100000.0)) / 3;
    }
    return kl_total_score;

};


void lsh::reset(state_merger *merger) {
    num_merges = 0;
    kl_total_score = 0;
    kl_score = map<string, float>();
    evaluation_function::reset(merger);
    compute_before_merge = false;
    first_merge = true;
};


