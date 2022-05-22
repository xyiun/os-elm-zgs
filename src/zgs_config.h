#ifndef OS_ELM_ZGS_SRC_CONFIG_H
#define OS_ELM_ZGS_SRC_CONFIG_H

#include <string>
#include <sstream>

namespace OsElmZgs {

struct ZgsConfig{
    // --- Agent Config ---
    size_t _InputDim; // dimension of input
    size_t _HiddenDim; // dimension of hidden layer 
    size_t _OutputDim; // dimension of output
    double _Beta;      // RBF function
    double _Lambda;    // ZGS param lambda


    // --- ZGS Config ---
    size_t _IterTime; // iterate time of ZGS
    double _GammaA;   // ZGS param GammaA, for output weight
    double _GammaB;   // ZGS param GammaB, for M

    std::string Display() {
        std::stringstream s;
        s << "---[ZGS Agent]---" << std::endl;
        s << "-输入的维度:" << _InputDim << std::endl;
        s << "-输出的维度:" << _OutputDim << std::endl;
        s << "-隐藏层维度:" << _HiddenDim << std::endl;
        s << "-RBF参数 Beta:" << _Beta <<std::endl;
        s << "-伪逆正则参数:" << _Lambda <<std::endl;
        s << "-----------------" << std::endl;
        s << std::endl;

        s << "---[ZGS Iteration]---" << std::endl;
        s << "-迭代次数为:" << _IterTime << std::endl;
        s << "-γ 1 的值为:" << _GammaA << std::endl;
        s << "-γ 2 的值为:" << _GammaB << std::endl;
        s << "---------------------" << std::endl;

        return s.str();
    }
};

};

#endif // OS_ELM_ZGS_SRC_CONFIG_H
