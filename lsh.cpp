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
#include "lsh.h"
#include <lsh/include/rh_lsh.h>

REGISTER_DEF_TYPE(lsh);
REGISTER_DEF_DATATYPE(lsh_data);

lsh_data::lsh_data() {
    occcurences = map<int, int>();
    mylsh = new lshbox::rhpLsh<int>();
    lshbox::rhpLsh<int>::Parameter param;
    int kl_score;
    param.M = 10; // Hash table size
    param.L = 1; // Number of hash tables
    param.D = 10; // Dimension of the vector
    param.N = 100; // Binary code bytes
    mylsh->reset(param);
};

//normalize by adding  0.01 before dividing by sum
float kl_divergence(vector<float> dist1, vector<float> dist2) {
    float kl_score = 0;
    for (int i = 0; i < dist1.size(); ++i) {
        if (dist1[i] != 0 && dist2[i] != 0)
            kl_score += dist1[i] * log(dist1[i] / dist2[i]);
        else if (dist1[i] != 0 && dist2[i] == 0)
            kl_score += dist1[i] * log(dist1[i] / 0.01);
    }
    return kl_score;
}

void lsh_data::print_transition_label(iostream &output, int symbol, apta *aptacontext) {
//    output << "count data";
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

//    if (type == 1) {
//        accepting_paths++;
//    } else {
//        rejecting_paths++;
//    }
//    bool found = (std::find(symbol_list.begin(), symbol_list.end(), symbol) != symbol_list.end());
//    if (!found) {
//        symbol_list.push_back(symbol);
//    }
//    d = data;
};

void lsh_data::read_to(int type, int index, int length, int symbol, string data) {
//    if (type == 1) {
//        if (length == index + 1) {
//            num_accepting++;
//        }
//    } else {
//        if (length == index + 1) {
//            num_rejecting++;
//        }
//    }
//    bool found = (std::find(symbol_list.begin(), symbol_list.end(), symbol) != symbol_list.end());
//    if (!found) {
//        symbol_list.push_back(symbol);
//    }
//    d = data;
};

void lsh_data::update_lsh(list <vector<int>> all_labels) {
    // if (label_list.size() >= 10) {
    //     vector<int> b(label_list.begin(), label_list.begin() + 10);
    //     mylsh->hash(b);
    // }
    for (auto p = all_labels.begin(); p != all_labels.end(); ++p) {
        vector<int> &label_list = *p;
        if (label_list.size() >= 10) {
            vector<int> b(label_list.begin(), label_list.begin() + 10);
            mylsh->hash(b);
        } else {
            mylsh->hash(label_list);
        }
    }
};

void lsh_data::update(evaluation_data *right) {
    lsh_data * other = reinterpret_cast<lsh_data *>(right);
    for (int i = 0; i != (other->mylsh->tables[0]).size(); ++i) {
        for (auto j = 0; j != (other->mylsh->tables[0][i]).size(); ++j) {
            if (find((mylsh->tables[0][i]).begin(), (mylsh->tables[0][i]).end(), other->mylsh->tables[0][i][j]) ==
                (mylsh->tables[0][i]).end()) {
                (mylsh->tables[0][i]).push_back(other->mylsh->tables[0][i][j]); // TODO: remove duplicates
            }
        }
    }
};

void lsh_data::undo(evaluation_data *right) {
    lsh_data * other = reinterpret_cast<lsh_data *>(right);
    num_accepting -= other->num_accepting;
    num_rejecting -= other->num_rejecting;
    accepting_paths -= other->accepting_paths;
    rejecting_paths -= other->rejecting_paths;
    for (int i = 0; i != (other->mylsh->tables[0]).size(); ++i) {
        for (auto j = 0; j != (other->mylsh->tables[0][i]).size(); ++j) {
            if (find((mylsh->tables[0][i]).begin(), (mylsh->tables[0][i]).end(), other->mylsh->tables[0][i][j]) !=
                (mylsh->tables[0][i]).end()) {
                (mylsh->tables[0][i]).erase(remove((mylsh->tables[0][i]).begin(), (mylsh->tables[0][i]).end(),
                                                   other->mylsh->tables[0][i][j]), (mylsh->tables[0][i]).end());
            }
        }
    }
};

bool lsh::consistency_check(evaluation_data *left, evaluation_data *right) {
    lsh_data * l = reinterpret_cast<lsh_data *>(left);
    lsh_data * r = reinterpret_cast<lsh_data *>(right);

    if (l->num_accepting != 0 && r->num_rejecting != 0) { return false; }
    if (l->num_rejecting != 0 && r->num_accepting != 0) { return false; }

    return true;
};

vector<float> get_distribution(map<unsigned int, vector<string>> table) {
    vector<float> dist(10);
    int total = 0;
    for (int t = 0; t != dist.size(); ++t) {
        if (table.find(t) != table.end()) {
            dist[t] = (table[t]).size() + 1;
            total += (table[t]).size() + 1;
        } else {
            dist[t] = 1;
            total += 1;
        }
//
//
//        dist[t] = (table[t]).size() + 1;
//        total += (table[t]).size() + 1;

    }

    for (int i = 0; i != dist.size(); ++i) {
        dist[i] /= total;
    }
    return dist;
}

/* default evaluation, count number of performed merges */
bool lsh::consistent(state_merger *merger, apta_node *left, apta_node *right) {

    if (inconsistency_found) return false;

    lsh_data * l = (lsh_data *) left->data;
    lsh_data * r = (lsh_data *) right->data;

    vector<float> l_dist = get_distribution(l->mylsh->tables[0]);
    vector<float> r_dist = get_distribution(r->mylsh->tables[0]);

    if (kl_divergence(l_dist, r_dist) > LSH_KL_THRESHOLD) return false;
    else return true;
};

void lsh::update_score(state_merger *merger, apta_node *left, apta_node *right) {
//    num_merges += 1;
    kl_score += compute_score(merger, left, right);

};

int lsh::compute_score(state_merger *merger, apta_node *left, apta_node *right) {
//    return num_merges;
    lsh_data * l = (lsh_data *) left->data;
    lsh_data * r = (lsh_data *) right->data;

    vector<float> l_dist = get_distribution(l->mylsh->tables[0]);
    vector<float> r_dist = get_distribution(r->mylsh->tables[0]);
    return (int)(100 - kl_divergence(l_dist, r_dist) * 100.0);
};

void lsh::reset(state_merger *merger) {
    num_merges = 0;
    kl_score = 0;
    evaluation_function::reset(merger);
    compute_before_merge = false;
};


//// sinks for evaluation data type
//bool lsh_data::is_accepting_sink(apta_node *node) {
//    lsh_data * d = reinterpret_cast<lsh_data *>(node->data);
//
//    node = node->find();
//    return d->rejecting_paths == 0 && d->num_rejecting == 0;
//};
//
//bool lsh_data::is_rejecting_sink(apta_node *node) {
//    lsh_data * d = reinterpret_cast<lsh_data *>(node->data);
//
//    node = node->find();
//    return d->accepting_paths == 0 && d->num_accepting == 0;
//};
//
//int lsh_data::sink_type(apta_node *node) {
//    if (!USE_SINKS) return -1;
//
//    if (lsh_data::is_rejecting_sink(node)) return 0;
//    if (lsh_data::is_accepting_sink(node)) return 1;
//    return -1;
//};
//
//bool lsh_data::sink_consistent(apta_node *node, int type) {
////    std::cout << "HEYYYYY" << std::endl;
//    if (!USE_SINKS) return true;
//
//    if (type == 0) return lsh_data::is_rejecting_sink(node);
//    if (type == 1) return lsh_data::is_accepting_sink(node);
//    return true;
//};
//
//int lsh_data::num_sink_types() {
//    if (!USE_SINKS) return 0;
//
//    // accepting or rejecting
//    return 2;
//};

