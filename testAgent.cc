#include <iostream>
#include <src/zgs.h>
#include <src/Eigen/Dense>
#include <src/log.h>

using namespace std;
using namespace Eigen;
using namespace OsElmZgs;

void TestZgsAgent() {
    {
        ZGS_LOG(INFO, "%s", "开始测试单个Zgs的回归能力"); 
        ZgsConfig config = {
            ._InputDim = 2,
            ._HiddenDim = 100,
            ._OutputDim = 1,
            ._Beta = 0.1
        };
        ZGS_LOG(INFO, "\n%s", config.Display().c_str());
        ZgsAgent agent(config);
        MatrixXd C = MatrixXd::Random(config._HiddenDim, config._InputDim);
        agent.SetRBFCenter(C);

        MatrixXd X = MatrixXd::Random(1000, 2);
        MatrixXd Y = MatrixXd::Zero(1000, 1);
        for(long i = 0; i < X.rows(); i++) {
            Y(i, 0) = sin(X(i, 0) * 2) + cos(X(i, 1)) * 10;
        }
        agent.Fit(X, Y);


        MatrixXd testX = MatrixXd::Random(1000, 2);
        MatrixXd testY = MatrixXd::Zero(1000, 1);
        for(long i = 0; i < X.rows(); i++) {
            testY(i, 0) = sin(testX(i, 0) * 2) + cos(testX(i, 1)) * 10;
        }
        MatrixXd P = agent.Predict(testX);
        for(size_t i = 0; i < 3; i++) {
            cout << "X is [" << testX.row(i) << "]" << endl;
            cout << "Y is [" << testY.row(i) << "]" << endl;
            cout << "P is [" << P.row(i) << "]" << endl;
            cout << endl;
        }

        cout << "MSE is [" << CalculateMSE(P, testY) << "]" << endl; 
    }

    {
        ZGS_LOG(INFO, "%s", "开始测试单个Zgs的分类能力"); 
        ZgsConfig config = {
            ._InputDim = 2,
            ._HiddenDim = 100,
            ._OutputDim = 4,
            ._Beta = 0.01
        };

        ZGS_LOG(INFO, "\n%s", config.Display().c_str());
        ZgsAgent agent(config);
        MatrixXd C = MatrixXd::Random(config._HiddenDim, config._InputDim);
        agent.SetRBFCenter(C);

        MatrixXd X = MatrixXd::Random(1000, 2);
        MatrixXd Y = MatrixXd::Zero(1000, 1);
        for(long i = 0; i < X.rows(); i++) {
            if(X(i, 0) > 0 && X(i, 1) > 0) {
                Y(i, 0) = 0;
            } else if(X(i, 0) > 0 && X(i, 1) <= 0) {
                Y(i, 0) = 1;
            } else if(X(i, 0) <= 0 && X(i, 1) > 0) {
                Y(i, 0) = 2;
            } else {
                Y(i, 0) = 3;
            }
        }
        MatrixXd onehotY = Tag2Onehot(Y);
        agent.Fit(X, onehotY);


        MatrixXd testX = MatrixXd::Random(1000, 2);
        MatrixXd testY = MatrixXd::Zero(1000, 1);
        for(long i = 0; i < X.rows(); i++) {
            if(testX(i, 0) > 0 && testX(i, 1) > 0) {
                testY(i, 0) = 0;
            } else if(testX(i, 0) > 0 && testX(i, 1) <= 0) {
                testY(i, 0) = 1;
            } else if(testX(i, 0) <= 0 && testX(i, 1) > 0) {
                testY(i, 0) = 2;
            } else {
                testY(i, 0) = 3;
            }
        }
        MatrixXd onehotTestY = Tag2Onehot(testY);
        MatrixXd predict = agent.Predict(testX);
        MatrixXd P = Onehot2Tag(predict);
        for(size_t i = 0; i < 3; i++) {
            cout << "X is [" << testX.row(i) << "]" << endl;
            cout << "Y is [" << testY.row(i) << "]" << endl;
            cout << "P is [" << P.row(i) << "]" << endl;
            cout << "origin P is [" << predict.row(i) << "]" << endl; 
            cout << "onehot Y is [" << onehotTestY.row(i) << "]" << endl; 
            cout << endl;
        }

        cout << "ACC is [" << CalculateAcc(predict, testY) << "]" << endl; 
    }
}

int main() {
    TestZgsAgent();
    return 0;
}
