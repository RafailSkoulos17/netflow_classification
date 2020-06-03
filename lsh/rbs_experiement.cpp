//template<typename DATATYPE>
//unsigned lshbox::rbsLsh<DATATYPE>::getHashVal(unsigned k, std::vector <DATATYPE> domin) {
////    std::cout << "GETTING ITS HASH VALUE" << std::endl;
//    unsigned sum(0);
//    int s[log2(param.M)];
//    for (unsigned i = 0; i != log2(param.M); ++i) {
////        std::cout << "ITERATE FOR THE " << i << " BINARY VALUE" << std::endl;
//        float flag(0);
//        int iters;
//
//        if (domin.size() < param.D) iters=domin.size();
//        else iters = param.D;
//
//        for (unsigned j = 0; j != iters; ++j) {
//            flag += uosArray[k][i][j] * domin[j];
//        }
//
//        if (flag >=0){
//            s[i] = 1;
//        }
//        else {
//            s[i] = 0;
//        }
//    }
//    int p = 1;
//    unsigned hashVal;
//    for(int & v : s){
//        hashVal += s*p;
//        p = p *2;
//    }
////    std::cout << "HASHVAL: " << hashVal << std::endl;
////    std::cout << std::endl;
//    return hashVal;