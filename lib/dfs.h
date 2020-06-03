//
// Created by rafail on 4/17/20.
//

#ifndef FLEXFRINGE_DFS_H
#define FLEXFRINGE_DFS_H

#include <stdlib.h>

#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "apta.h"


void DFSUtil(apta_node *v, apta_node *v_init, map<int, bool> visited, int path_index, int length, vector<int> symbols,
             list <vector<int>> &all_symbols, int depth);

// DFS traversal of the vertices reachable from v.
// It uses recursive DFSUtil()
void DFS(apta_node *v, list <vector<int>> &all_symbols, int depth);


#endif //FLEXFRINGE_DFS_H



