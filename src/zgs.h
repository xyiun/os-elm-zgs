#ifndef OS_ELM_ZGS_SRC_ZGS_H
#define OS_ELM_ZGS_SRC_ZGS_H

#include <src/common.h>
#include <src/agent.h>
#include <src/zgs_config.h>
#include <src/tools.h>

namespace OsElmZgs{

class ZGS {
public:
    // argc, argv is the param of main() funciton
    ZGS(int argc, char* argv[], ZgsConfig& config);
    ~ZGS();

    // Do Init
    bool Init();

    // init train
    bool InitFit(Eigen::MatrixXd& X, Eigen::MatrixXd& Y);

    // Do online Fit
    bool Update(Eigen::MatrixXd& X, Eigen::MatrixXd& Y);

    // Do Pridict
    Eigen::MatrixXd Predict(Eigen::MatrixXd& X);

    int GetId() const {return _Id;}
    int GetAgentCount() const {return _AgentCount;}

    // for debug
    void Check();
private:
    // Get average matrix of all Agent
    Eigen::MatrixXd GetAgentAvg(Eigen::MatrixXd& X);
private:
    // Single Zgs Agent
    ZgsAgent _Agent;
    // MPI ID
    int _Id;
    // MPI machine name
    std::string _MachineName;
    // MPI Agent count
    int _AgentCount;
    // config
    ZgsConfig _Config;

    // Sum of history M
    Eigen::MatrixXd _SumM;
};

};
#endif // OS_ELM_ZGS_SRC_ZGS_H
