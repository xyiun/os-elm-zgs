#include <src/zgs.h>
#include <src/log.h>

namespace OsElmZgs {

ZGS::ZGS(int argc, char* argv[], ZgsConfig& config)
    :_Agent(config),
     _Id(-1),
     _AgentCount(0),
     _Config(config)
{
     MPI_Init(&argc, &argv);
     MPI_Comm_rank(MPI_COMM_WORLD, &_Id);
     MPI_Comm_size(MPI_COMM_WORLD, &_AgentCount);
     char machineName[MPI_MAX_PROCESSOR_NAME];
     int namelen;
     MPI_Get_processor_name(machineName, &namelen);
     ZGS_LOG(INFO, "current id is [%d], "
                   "machine name is [%s], "
                   "total agent count [%d]",
                   _Id, machineName, _AgentCount);
     _MachineName = machineName;
}

ZGS::~ZGS() {
    MPI_Finalize();
}

bool ZGS::Init() {
    if(_Id == 0) {
        ZGS_LOG(INFO, "[%d]Random generate RBF Center", _Id);
        Eigen::MatrixXd C = Eigen::MatrixXd::Random(_Config._HiddenDim, _Config._InputDim);
        _Agent.SetRBFCenter(C);
        
        int mpiRet;
        mpiRet = MPI_Bcast(C.data(), // data
                  C.rows() * C.cols(), // elemCount
                  MPI_DOUBLE, //data type
                  _Id, // who send this message
                  MPI_COMM_WORLD);
        if(mpiRet != 0) {
            ZGS_LOG(ERROR, "do init failed, bcast failed, error code is [%d]", mpiRet);
            return false;
        }
    } else {
        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(_Config._HiddenDim, _Config._InputDim);
        ZGS_LOG(INFO, "[%d]Waiting for RBF Center", _Id);
        int mpiRet;
        mpiRet = MPI_Bcast(C.data(), // data
                  C.rows() * C.cols(), // elemCount
                  MPI_DOUBLE, //data type
                  0, // who send this message
                  MPI_COMM_WORLD);
        if(mpiRet != 0) {
            ZGS_LOG(ERROR, "do init failed, recv bcast failed, error code is [%d]", mpiRet);
            return false;
        }
        _Agent.SetRBFCenter(C);
    }
    return true;
}

Eigen::MatrixXd ZGS::GetAgentAvg(Eigen::MatrixXd& X) {
    int cols = X.cols() / _AgentCount;
    int rows = X.rows();
    Eigen::MatrixXd ret = Eigen::MatrixXd::Zero(rows, cols);
    for(int i = 0 ; i < _AgentCount; i++) {
        ret += X.block(0, i * cols, rows, cols);
    }
    return ret / _AgentCount;
}


bool ZGS::InitFit(Eigen::MatrixXd& X, Eigen::MatrixXd& Y) {
    _Agent.Fit(X, Y);
    if(_AgentCount == 1) {
        ZGS_LOG(INFO, "%s", "Only one agent");
        _SumM = _Agent._M;
        return true;
    }
    // Do ZGS
    Eigen::MatrixXd Ws = Eigen::MatrixXd::Zero(_Config._HiddenDim,
            _Config._OutputDim * _AgentCount);
    Eigen::MatrixXd Ms = Eigen::MatrixXd::Zero(_Config._HiddenDim,
            _Config._HiddenDim * _AgentCount);
    for(unsigned long i = 0; i < _Config._IterTime; i++) {
        // send W to other agents
        // also get W from other agents
        int mpiRet = 0;
        mpiRet = MPI_Allgather(_Agent._W.data(),
                               _Config._HiddenDim * _Config._OutputDim,
                               MPI_DOUBLE,
                               Ws.data(),
                               _Config._HiddenDim * _Config._OutputDim,
                               MPI_DOUBLE,
                               MPI_COMM_WORLD);
        if(mpiRet != 0) {
            ZGS_LOG(ERROR, "InitFit failed, mpi Allgather failed, error code is [%d]", mpiRet);
            return false;
        }

        mpiRet = MPI_Allgather(_Agent._M.data(),
                               _Config._HiddenDim * _Config._HiddenDim,
                               MPI_DOUBLE,
                               Ms.data(),
                               _Config._HiddenDim * _Config._HiddenDim,
                               MPI_DOUBLE,
                               MPI_COMM_WORLD);
        if(mpiRet != 0) {
            ZGS_LOG(ERROR, "InitFit failed, mpi Allgather failed, error code is [%d]", mpiRet);
            return false;
        }
        if(i + 1 == _Config._IterTime) {
            _Agent._W = GetAgentAvg(Ws);
            _Agent._M = GetAgentAvg(Ms);
            break;
        } 
        Eigen::MatrixXd tmpW = _Agent._W * (-1 * _AgentCount + 1);
        for(int i = 0; i < _AgentCount; i++) {
            if(i == _Id) {
                continue;
            }
            tmpW = tmpW + Ws.block(0, _Config._OutputDim * i, _Config._HiddenDim, _Config._OutputDim);
        }
        _Agent._W = _Agent._W + _Agent._pinvK * tmpW * _Config._GammaA;
       
        Eigen::MatrixXd tmpM = _Agent._M * (-1 * _AgentCount + 1);
        for(int i = 0; i < _AgentCount; i++) {
            if(i == _Id) {
                continue;
            }
            tmpM = tmpM + Ms.block(0, _Config._HiddenDim * i, _Config._HiddenDim, _Config._HiddenDim);
        }
        _Agent._M = _Agent._M + tmpM * _Config._GammaB;
    }

    _SumM = _Agent._M;
    return true;
}

bool ZGS::Update(Eigen::MatrixXd& X, Eigen::MatrixXd& Y) {
    // do online ELM
    _Agent.OnLineElm(X, Y, _SumM * _AgentCount);
    // do zgs
    if(_AgentCount == 1) {
        _SumM += _Agent._M;
        return true;
    }
    
    Eigen::MatrixXd E = Eigen::MatrixXd::Identity(_Config._HiddenDim, _Config._HiddenDim)
                            * _Config._Lambda;
    if(_Id != 0) {
        Eigen::MatrixXd W = (_Agent._M + E).completeOrthogonalDecomposition().pseudoInverse();
        _Agent._W = W * _Agent._HT * Y;
    }


    // Do ZGS
    Eigen::MatrixXd Ws = Eigen::MatrixXd::Zero(_Config._HiddenDim,
            _Config._OutputDim * _AgentCount);
    Eigen::MatrixXd Ms = Eigen::MatrixXd::Zero(_Config._HiddenDim,
            _Config._HiddenDim * _AgentCount);

    Eigen::MatrixXd pinvM = (_Agent._M + E).completeOrthogonalDecomposition().pseudoInverse();
    for(unsigned long i = 0; i < _Config._IterTime; i++) {
        // send W to other agents
        // also get W from other agents
        int mpiRet = 0;
        mpiRet = MPI_Allgather(_Agent._W.data(),
                               _Config._HiddenDim * _Config._OutputDim,
                               MPI_DOUBLE,
                               Ws.data(),
                               _Config._HiddenDim * _Config._OutputDim,
                               MPI_DOUBLE,
                               MPI_COMM_WORLD);
        if(mpiRet != 0) {
            ZGS_LOG(ERROR, "InitFit failed, mpi Allgather failed, error code is [%d]", mpiRet);
            return false;
        }

        mpiRet = MPI_Allgather(_Agent._M.data(),
                               _Config._HiddenDim * _Config._HiddenDim,
                               MPI_DOUBLE,
                               Ms.data(),
                               _Config._HiddenDim * _Config._HiddenDim,
                               MPI_DOUBLE,
                               MPI_COMM_WORLD);
        if(mpiRet != 0) {
            ZGS_LOG(ERROR, "InitFit failed, mpi Allgather failed, error code is [%d]", mpiRet);
            return false;
        }
    
        if(i + 1 == _Config._IterTime) {
            _Agent._W = GetAgentAvg(Ws);
            _Agent._M = GetAgentAvg(Ms);
            break;
        } 

        Eigen::MatrixXd tmpW = _Agent._W * (-1 * _AgentCount + 1);
        for(int i = 0; i < _AgentCount; i++) {
            if(i == _Id) {
                continue;
            }
            tmpW = tmpW + Ws.block(0, _Config._OutputDim * i, _Config._HiddenDim, _Config._OutputDim);
        }
        if(_Id == 0) {
            _Agent._W = _Agent._W + _Agent._pinvK * tmpW * _Config._GammaA;
        } else {
            _Agent._W = _Agent._W + pinvM * tmpW * _Config._GammaA;
        }

        Eigen::MatrixXd tmpM = _Agent._M * (-1 * _AgentCount + 1);
        for(int i = 0; i < _AgentCount; i++) {
            if(i == _Id) {
                continue;
            }
            tmpM = tmpM + Ms.block(0, _Config._HiddenDim * i, _Config._HiddenDim, _Config._HiddenDim);
        }
        _Agent._M = _Agent._M + tmpM * _Config._GammaB;
    }

    _SumM += _Agent._M;
    return true;
}

Eigen::MatrixXd ZGS::Predict(Eigen::MatrixXd& X) {
    return _Agent.Predict(X);
}

void ZGS::Check() {
    ZGS_LOG(INFO, "W_0_0[%f] M_0_0[%f]", _Agent._W(0,0), _Agent._M(0, 0));
}

};
