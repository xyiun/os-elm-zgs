#include <iostream>
#include <src/zgs.h>
#include <src/Eigen/Dense>
#include <src/log.h>

using namespace std;
using namespace Eigen;
using namespace OsElmZgs;

void Process(int argc, char* argv[]) {
    ZgsConfig config = {
        ._InputDim = 2,
        ._HiddenDim = 100,
        ._OutputDim = 1,
        ._Beta = 0.1,
        ._Lambda = 0.001,

        ._IterTime = 1000,
        ._GammaA = 0.001,
        ._GammaB = 0.01
    };

    ZGS zgs(argc, argv, config);
    zgs.Init();

    MatrixXd X = MatrixXd::Random(50, 2);
    MatrixXd Y = MatrixXd::Zero(50, 1);
    for(size_t i = 0; i < (size_t)X.rows(); i++) {
        Y(i, 0) = sin(X(i, 0)) + cos(X(i, 1));
    }

    MatrixXd testX = MatrixXd::Zero(50, 2);
    MatrixXd testY = MatrixXd::Zero(50, 1);
    for(size_t i = 0; i < (size_t)testX.rows(); i++) {
        testX(i, 0) = i * 0.02 - 0.5;
        testX(i, 1) = i * 0.02 - 0.5;
        testY(i, 0) = sin(testX(i, 0)) + cos(testX(i, 1));
    }

    zgs.InitFit(X, Y);
    MatrixXd P = zgs.Predict(testX);
    cout << "MSE " << CalculateMSE(P, testY) << endl;
   
    for(size_t i = 0; i < 10; i++) {
        X = MatrixXd::Random(10, 2);
        Y = MatrixXd::Zero(10, 1);
        for(size_t i = 0; i < (size_t)X.rows(); i++) {
            Y(i, 0) = sin(X(i, 0)) + cos(X(i, 1));
        }
        zgs.Update(X, Y);
    }

    P = zgs.Predict(testX);
    cout << "MSE " << CalculateMSE(P, testY) << endl;
}

int main(int argc, char* argv[]) {
    Process(argc, argv);
    return 0;
}
