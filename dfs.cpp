//
// Created by rafail on 4/17/20.
//

#include <stdlib.h>

#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "dfs.h"


void DFSUtil(apta_node *v, apta_node *v_init, map<int, bool> visited, int path_index, int length, vector<int> symbols,
             list <vector<int>> &all_symbols, int depth) {
    // Mark the current node as visited and
    // print it
    bool found = false;
    if (length < depth) {
        visited[v->number] = true;
//        cout << v->label << " ";
        symbols[length] = v->label;
        length += 1;
        // Recur for all the vertices adjacent
        // to this vertex
        list < apta_node * > adj;
        for (auto it = v->guards.begin(); it != v->guards.end(); ++it) {
            apta_node *target = (*it).second->target;
            if (target != NULL) {
                adj.push_back(target);
            }
        }
        if (adj.empty()) {
            vector<int> b(symbols.begin() + 1, symbols.begin() + length);
            all_symbols.push_back(b);
//            cout << (v_init)->number << ": ";
//            for (int i = 0; i < length - 1; i++) {
//                cout << b[i] << " ";
//            }
//            cout << std::endl;
        } else {
            vector<int> b(symbols.begin() + 1, symbols.begin() + length);
            all_symbols.push_back(b);
            list<apta_node *>::iterator i;
            for (auto i = adj.begin(); i != adj.end(); ++i) {
                apta_node *n = *i;
                if (!visited[n->number]) {
                    DFSUtil(n, v_init, visited, path_index, length, symbols, all_symbols, depth);
                }
            }
        }
    } else if (length == depth) {
        vector<int> b(symbols.begin() + 1, symbols.end());
        all_symbols.push_back(b);
    }
}

// DFS traversal of the vertices reachable from v.
// It uses recursive DFSUtil()
void DFS(apta_node *v, list <vector<int>> &all_symbols, int depth) {
    // Mark all the vertices as not visited
    map<int, bool> visited;
    int V = depth;

    // Create an array to store paths
    vector<int> symbols(V);
    int path_index = 0; // Initialize path[] as empty
    int length = 0;
    // Call the recursive helper function
    // to print DFS traversal
    DFSUtil(v, v, visited, path_index, length, symbols, all_symbols, depth);
}
