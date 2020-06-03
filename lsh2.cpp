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
#include "lsh2.h"
//#include <lsh/include/psd_lsh.h>
#include <lsh/include/rh_lsh.h>
//#include <lsh/include/rbs_lsh.h>
#include<cmath>
#include <tuple>

REGISTER_DEF_TYPE(lsh2);
REGISTER_DEF_DATATYPE(lsh2_data);

lsh2_data::lsh2_data() {
//    TODO: delete type declaration
    occcurences = map<int, int>();
//    kl_score = map<string, float>();

//    mylsh = new lshbox::rbsLsh<int>();
//    lshbox::rbsLsh<int>::Parameter param;

    mylsh = new lshbox::rhpLsh<int>();
    lshbox::rhpLsh<int>::Parameter param;

//    mylsh = new lshbox::psdLsh<int>();
//    lshbox::psdLsh<int>::Parameter param;

    int total = 0;
    vector<int> counter(LSH_D);
    param.M = LSH_M; // Hash table size
    param.L = LSH_L; // Number of hash tables
    param.D = LSH_D; // Dimension of the vector
//    param.T = LSH_T; // Index mode, you can choose 1(CAUCHY) or 2(GAUSSIAN)
//    param.W = LSH_W; // Window size
    param.N = LSH_N; // Binary code bytes
//    param.C = LSH_C; // The Difference between upper and lower bound of each dimension
    mylsh->reset(param);
};

//normalize by adding  0.01 before dividing by sum
float kl_divergence2(vector<float> dist1, vector<float> dist2) {
    float kl_score = 0;
    for (int i = 0; i < dist1.size(); ++i) {
        if (dist1[i] != 0 && dist2[i] != 0)
            kl_score += dist1[i] * log((float) dist1[i] / (float) dist2[i]);
        else if (dist1[i] != 0 && dist2[i] == 0)
            kl_score += dist1[i] * log((float) dist1[i] / 0.01);
    }
    return kl_score;
}

float chi_squared_distance(vector<float> dist1, vector<float> dist2) {
    float cs_dist = 0;
    for (int i = 0; i < dist1.size(); ++i) {
        cs_dist += (float) (pow(dist1[i] - dist2[i], 2) / (float) (dist1[i] + dist2[i]));
    }
    return (float) cs_dist / 2.0;

}

float bhattacharyya_distance(vector<float> dist1, vector<float> dist2) {
    float b_dist = 0;
    for (int i = 0; i < dist1.size(); ++i) {
        b_dist += sqrt(dist1[i] * dist2[i]);
    }
    return -log(b_dist);
}

float hellinger_distance(vector<float> dist1, vector<float> dist2) {
    float h_dist = 0;
    for (int i = 0; i < dist1.size(); ++i) {
        float temp = sqrt(dist1[i]) - sqrt(dist2[i]);
        h_dist += pow(temp, 2);
    }
    return (float) (sqrt(h_dist)) / (float) (sqrt(2));

}

float bhattacharyya_coefficient(vector<float> dist1, vector<float> dist2) {
    float b_dist = 0;
    for (int i = 0; i < dist1.size(); ++i) {
        b_dist += sqrt(dist1[i] * dist2[i]);
    }
    return b_dist;
}

float KS_test(vector<float> dist1, vector<float> dist2) {
    vector<float> cum_dist1(dist1.size());
    vector<float> cum_dist2(dist2.size());
    float tot1 = 0.0;
    float tot2 = 0.0;

    for (int i = 0; i < dist1.size(); ++i) {
        tot1 += dist1[i];
        tot2 += dist2[i];
        cum_dist1[i] = tot1;
        cum_dist2[i] = tot2;
    }
    float max_dist = 0;
    for (int i = 0; i < cum_dist1.size(); ++i) {
        float temp = cum_dist1[i] - cum_dist2[i];
        if (abs(temp) > max_dist) max_dist = abs(temp);
    }
    return max_dist;

}


float cramer_test(vector<float> dist1, vector<float> dist2, vector<int> counter1, vector<int> counter2) {
    float Tcvm = 0;
    int total1 = 0, total2 = 0;
    for (int i = 0; i < counter1.size(); ++i) {
        total1 += counter1[i];
        total2 += counter2[i];
    }

    vector<float> cum_dist1(dist1.size());
    vector<float> cum_dist2(dist2.size());
    float tot1 = 0.0;
    float tot2 = 0.0;

    for (int i = 0; i < dist1.size(); ++i) {
        tot1 += dist1[i];
        tot2 += dist2[i];
        cum_dist1[i] = tot1;
        cum_dist2[i] = tot2;
    }

    for (int i = 0; i < dist1.size(); ++i) {
        Tcvm += (counter1[i] + counter2[i]) * pow((cum_dist1[i] - cum_dist2[i]), 2);
    }
    return ((float) (total1 * total2) / (float) (pow((total1 + total2), 2))) * Tcvm;
}

float likelihood_ratio(vector<float> dist1, vector<float> dist2, vector<int> counter1, vector<int> counter2) {
    float likelihood = 0;
    int total1 = 0, total2 = 0;
    for (int i = 0; i < counter1.size(); ++i) {
        total1 += counter1[i];
        total2 += counter2[i];
    }

    for (int i = 0; i < counter1.size(); ++i) {
        int ti = counter1[1] + counter2[i];
        float temp1 = log(
                (float) (1 + (float) counter1[i] / (float) counter2[i]) /
                (float) (1 + (float) total1 / (float) total2));
        float temp2 = log(
                (float) (total1 * counter1[i]) / (float) (total2 * counter2[i]));

        likelihood += ti * temp1 + counter1[i] * temp2;
    }
    return -2 * likelihood;
}


void lsh2_data::print_transition_label(iostream &output, int symbol, apta *aptacontext) {
//    output << "count data";
    output << occcurences[symbol];
    if (occcurences[symbol] == 0) {
        cout << "STOP" << endl;
    }
};

void lsh2_data::print_state_label(iostream &output) {
    output << "count data";
};


void lsh2_data::read_from(int type, int index, int length, int symbol, string data) {
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

void lsh2_data::read_to(int type, int index, int length, int symbol, string data) {
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
                dist[t] = 1;
                total += 1;
            }
//            dist[t] = (table[t]).size();
//            total += (table[t]).size();
        }
    }

//    for (int i = 0; i != dist.size(); ++i) {
//        dist[i] /= total;
//    }
    return make_tuple(dist, total);
};

void lsh2_data::update_lsh(list <vector<int>> all_labels) {
    for (auto p = all_labels.begin(); p != all_labels.end(); ++p) {
        vector<int> &label_list = *p;
        if (label_list.size() >= LSH_D) {
            vector<int> b(label_list.begin(), label_list.begin() + LSH_D);
            mylsh->hash(b);
        } else {
            mylsh->hash(label_list);
        }
    }

    tie(counter, total) = get_distribution(mylsh->tables);
//    if (dist.empty()) {
//        cout << "No distribution" << endl;
//    }
};

void lsh2_data::update(evaluation_data *right) {
    lsh2_data * other = reinterpret_cast<lsh2_data *>(right);
//    for (int i = 0; i != (other->mylsh->tables[0]).size(); ++i) {
//        for (auto j = 0; j != (other->mylsh->tables[0][i]).size(); ++j) {
//            (mylsh->tables[0][i]).push_back(other->mylsh->tables[0][i][j]); // TODO: remove duplicates
//
//        }
//    }

    for (map<int, int>::iterator it = other->occcurences.begin(); it != other->occcurences.end(); ++it) {
        occcurences[(*it).first] = pos((*it).first) + (*it).second;
    }

//    for (int i = 0; i != dist.size(); ++i) {
//        dist[i] = (dist[i] + other->dist[i]) / 2;
//    }
    for (int i = 0; i != counter.size(); ++i) {
        counter[i] += other->counter[i];
        total += other->counter[i];
    }
    if (counter.empty()) {
        cout << "No distribution" << endl;
    }
};

void lsh2_data::undo(evaluation_data *right) {
    lsh2_data * other = reinterpret_cast<lsh2_data *>(right);
    for (map<int, int>::iterator it = other->occcurences.begin(); it != other->occcurences.end(); ++it) {
        occcurences[(*it).first] = pos((*it).first) - (*it).second;
    }
//    num_accepting -= other->num_accepting;
//    num_rejecting -= other->num_rejecting;
//    accepting_paths -= other->accepting_paths;
//    rejecting_paths -= other->rejecting_paths;
//    for (int i = 0; i != (other->mylsh->tables[0]).size(); ++i) {
//        for (auto j = 0; j != (other->mylsh->tables[0][i]).size(); ++j) {
//            if (find((mylsh->tables[0][i]).begin(), (mylsh->tables[0][i]).end(), other->mylsh->tables[0][i][j]) !=
//                (mylsh->tables[0][i]).end()) {
//                (mylsh->tables[0][i]).erase(remove((mylsh->tables[0][i]).begin(), (mylsh->tables[0][i]).end(),
//                                                   other->mylsh->tables[0][i][j]), (mylsh->tables[0][i]).end());
//            }
//        }
//    }
//    for (int i = 0; i != dist.size(); ++i) {
//        dist[i] = 2 * dist[i] - other->dist[i];
//    }
    for (int i = 0; i != counter.size(); ++i) {
        counter[i] -= other->counter[i];
        total -= other->counter[i];

    }
    if (counter.empty()) {
        cout << "No distribution" << endl;
    }
};

bool lsh2::consistency_check(evaluation_data *left, evaluation_data *right) {
    lsh2_data * l = reinterpret_cast<lsh2_data *>(left);
    lsh2_data * r = reinterpret_cast<lsh2_data *>(right);

    if (l->num_accepting != 0 && r->num_rejecting != 0) { return false; }
    if (l->num_rejecting != 0 && r->num_accepting != 0) { return false; }

    return true;
};


/* default evaluation, count number of performed merges */
bool lsh2::consistent(state_merger *merger, apta_node *left, apta_node *right) {

    if (inconsistency_found) return false;

    lsh2_data * l = (lsh2_data *) left->data;
    lsh2_data * r = (lsh2_data *) right->data;

//    map<unsigned int, vector<string>> l_table = l->mylsh->tables[0];
//    map<unsigned int, vector<string>> r_table = l->mylsh->tables[0];
//
//    vector<float> l_dist = get_distribution(l->mylsh->tables);
//    vector<float> r_dist = get_distribution(r->mylsh->tables);

    vector<float> ldist(LSH_D);
    vector<float> rdist(LSH_D);
    for (int i = 0; i != (l->counter).size(); ++i) {
        ldist[i] = ((float) l->counter[i]) / ((float) l->total);
        rdist[i] = ((float) r->counter[i]) / ((float) r->total);
    }
    float kl_div = kl_divergence2(ldist, rdist);
//    float cs_dist = chi_squared_distance(ldist, rdist);
//    float b_dist = bhattacharyya_distance(ldist, rdist);
//    float b_coeff = bhattacharyya_coefficient(ldist, rdist);
//    float h_dist = hellinger_distance(ldist, rdist);
//    float ks_dist = KS_test(ldist, rdist);
//    float cramer = cramer_test(ldist, rdist, l->counter, r->counter);
//    float likelihood = likelihood_ratio(ldist, rdist, l->counter, r->counter);
//    kl_score[std::to_string(right->number) + "_" + std::to_string(left->number)] = kl_div;
    if (kl_div > LSH_KL_THRESHOLD) return false;
//    if (cs_dist > LSH_KL_THRESHOLD) return false;
//    if (b_dist > LSH_KL_THRESHOLD) return false;
//    if (h_dist > LSH_KL_THRESHOLD) return false;
//    if (ks_dist > LSH_KL_THRESHOLD) return false;
//    if (likelihood > LSH_KL_THRESHOLD) return false;
//    if (cramer > LSH_KL_THRESHOLD) return false;
//    if (b_coeff < LSH_KL_THRESHOLD) return false;
    else return true;
};

//void lsh2::update_score(state_merger *merger, apta_node *left, apta_node *right) {
//    num_merges += 1;
//};
//
//int lsh2::compute_score(state_merger *merger, apta_node *left, apta_node *right) {
//    return num_merges;
//};
void lsh2::update_score(state_merger *merger, apta_node *left, apta_node *right) {
    num_merges += 1;
//    kl_total_score += compute_score(merger, left, right);
//    kl_total_score = compute_score(merger, left, right);

};

int lsh2::compute_score(state_merger *merger, apta_node *left, apta_node *right) {
    return num_merges;

//        int l_total, r_total;
//        lsh2_data * l = (lsh2_data *) left->data;
//        lsh2_data * r = (lsh2_data *) right->data;
//
//    //    tie(l_counter,l_total) = get_distribution(l->mylsh->tables);
//    //    tie(r_counter,r_total) = get_distribution(r->mylsh->tables);
//
//        vector<float> ldist(LSH_D);
//        vector<float> rdist(LSH_D);
//        for (int i = 0; i != (l->counter).size(); ++i) {
//            ldist[i] = ((float) l->counter[i]) / ((float) l->total);
//            rdist[i] = ((float) r->counter[i]) / ((float) r->total);
//        }
//
//        return (int) (1000 * 1/kl_divergence2(ldist, rdist));
////        return 100000 - (int)(kl_divergence2(ldist, rdist) * 100000.0);
//    //    return (int)(1000 - kl_divergence2(l_dist, r_dist) * 1000.0);
//    //    return (int) (1000 - kl_score[std::to_string(right->number) + "_" + std::to_string(left->number)] * 1000.0);
//    //    return (int) (1000 - kl_score[std::to_string(right->number) + "_" + std::to_string(left->number)] * 1000.0);
};


void lsh2::reset(state_merger *merger) {
    num_merges = 0;
    kl_total_score = 0;
    kl_score = map<string, float>();
    evaluation_function::reset(merger);
    compute_before_merge = false;
};


// sinks for evaluation data type
//bool lsh2_data::is_accepting_sink(apta_node *node) {
//    lsh2_data * d = reinterpret_cast<lsh2_data *>(node->data);
//
//    node = node->find();
//    return d->rejecting_paths == 0 && d->num_rejecting == 0;
//};
//
//bool lsh2_data::is_rejecting_sink(apta_node *node) {
//    lsh2_data * d = reinterpret_cast<lsh2_data *>(node->data);
//
//    node = node->find();
//    return d->accepting_paths == 0 && d->num_accepting == 0;
//};
//
//int lsh2_data::sink_type(apta_node *node) {
//    if (!USE_SINKS) return -1;
//
//    if (lsh2_data::is_rejecting_sink(node)) return 0;
//    if (lsh2_data::is_accepting_sink(node)) return 1;
//    return -1;
//};
//
//bool lsh2_data::sink_consistent(apta_node *node, int type) {
////    std::cout << "HEYYYYY" << std::endl;
//    if (!USE_SINKS) return true;
//
//    if (type == 0) return lsh2_data::is_rejecting_sink(node);
//    if (type == 1) return lsh2_data::is_accepting_sink(node);
//    return true;
//};
//
//int lsh2_data::num_sink_types() {
//    if (!USE_SINKS) return 0;
//
//    // accepting or rejecting
//    return 2;
//};

