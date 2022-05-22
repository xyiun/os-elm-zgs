#include <src/agent.h>
#include <src/Eigen/QR>

namespace OsElmZgs{

ZgsAgent::ZgsAgent(const ZgsConfig& config)
    :_InputDim(config._InputDim),
     _HiddenDim(config._HiddenDim),
     _OutputDim(config._OutputDim),
     _Beta(config._Beta),
     _Lambda(config._Lambda)
{}

void ZgsAgent::SetRBFCenter(const Eigen::MatrixXd& C) {
    size_t rows = C.rows();
    size_t cols = C.cols();
    assert(_HiddenDim == rows);
    assert(_InputDim == cols);
    _C = C;
}

void ZgsAgent::Fit(Eigen::MatrixXd& X, Eigen::MatrixXd& Y) {
    assert(X.cols() == (long)_InputDim);
    assert(X.rows() == Y.rows());
    assert(Y.cols() == (long)_OutputDim);
    ZGS_LOG(DEBUG, "%s", "start tring single agent");
    Eigen::MatrixXd H = CalculateHidden(X);
    Eigen::MatrixXd HT = H.transpose();

    ZGS_LOG(DEBUG, "HT shape [%d, %d]", HT.rows(), HT.cols());

    _H = H;
    _HT = HT;
    _M = HT * H;
    _K = _M;

    // calculate pinv of K
    Eigen::MatrixXd E = Eigen::MatrixXd::Identity(_HiddenDim, _HiddenDim) * _Lambda;
    Eigen::MatrixXd pinvK = (_K + E).completeOrthogonalDecomposition().pseudoInverse();
    // calculate output layer weight
    _W = pinvK * HT * Y;
    
    _pinvK = pinvK;

    ZGS_LOG(DEBUG, "%s", "singel agent fit done");
}

Eigen::MatrixXd ZgsAgent::CalculateHidden(Eigen::MatrixXd& X) {
    ZGS_LOG(DEBUG, "%s", "start calculate hidden");
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(X.rows(), _HiddenDim);
    for(long i = 0; i < X.rows(); i++) {
        for(size_t j = 0; j < _HiddenDim; j++) {
            double tmp;
            ZGS_LOG(DEBUG, "%d, %d", _C.row(j).cols(), X.row(i).cols());
            tmp = (X.row(i) - _C.row(j)).norm();
            tmp = exp(-1.0 * _Beta * tmp);
            H(i, j) = tmp;
        }
    }
    ZGS_LOG(DEBUG, "%s", "end calculate hidden");
    return H;
}

Eigen::MatrixXd ZgsAgent::Predict(Eigen::MatrixXd& X) {
    ZGS_LOG(DEBUG, "%s", "start do predict");
    Eigen::MatrixXd H = CalculateHidden(X);
    Eigen::MatrixXd ret = H * _W;
    ZGS_LOG(DEBUG, "%s", "predict done");
    return ret;
}

void ZgsAgent::OnLineElm(Eigen::MatrixXd& X, Eigen::MatrixXd& Y, Eigen::MatrixXd NSumM) {
    _H = CalculateHidden(X);
    _HT = _H.transpose();
    _M = _HT * _H;
    _K = _K + NSumM;
    Eigen::MatrixXd E = Eigen::MatrixXd::Identity(_HiddenDim, _HiddenDim) * _Lambda;
    _pinvK = (_K + E).completeOrthogonalDecomposition().pseudoInverse();
    _W = _W + _pinvK * _HT * (Y - _H * _W); 
}
};
