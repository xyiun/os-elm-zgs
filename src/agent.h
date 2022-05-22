#ifndef OS_ELM_ZGS_SRC_AGENT_H
#define OS_ELM_ZGS_SRC_AGENT_H

#include <src/common.h>
#include <src/log.h>
#include <src/zgs_config.h>

namespace OsElmZgs {

class ZgsAgent {
    friend class ZGS;
public:
    ZgsAgent(const ZgsConfig& config);

    // set RBF center param matrix
    void SetRBFCenter(const Eigen::MatrixXd& G);
    // tring RBFN Model
    void Fit(Eigen::MatrixXd& X, Eigen::MatrixXd& Y);
    // do predict
    Eigen::MatrixXd Predict(Eigen::MatrixXd& X);
private:
    // calculate the hidden layer output
    Eigen::MatrixXd CalculateHidden(Eigen::MatrixXd& X);

    // online ELM
    void OnLineElm(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, Eigen::MatrixXd NSumM);
private:
    // input dimension
    size_t _InputDim;
    // hidden layer dimension
    size_t _HiddenDim;
    // output dimension
    size_t _OutputDim;
    // RBF function param
    double _Beta;
    // ZGS param lambda
    double _Lambda;

    // RBF Center
    Eigen::MatrixXd _C;
    // Output layer weight
    Eigen::MatrixXd _W;

    // for ZGS iter
    Eigen::MatrixXd _H;
    Eigen::MatrixXd _HT;
    Eigen::MatrixXd _M;
    Eigen::MatrixXd _K;
    Eigen::MatrixXd _pinvK;

};

};

#endif // OS_ELM_ZGS_SRC_AGENT_H
