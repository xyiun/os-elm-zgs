#include <iostream>
#include <src/zgs.h>
#include <src/Eigen/Dense>
#include <src/log.h>
#include <fstream>
using namespace std;
using namespace Eigen;
using namespace OsElmZgs;

vector<double> splitString(string& s) {
    size_t last = 0;
    size_t next = 0;
    string token;
    vector<double> ret;
    while((next = s.find(",", last)) != string::npos) {
        token = s.substr(last, next - last);
        ret.push_back(stod(token));
        last = next + 1;
    }
    if(last < s.size()) {
        token = s.substr(last);
        ret.push_back(stod(token));
    }
    return ret;
}

double sigmod(double x) {
    return 1.0 / (1.0 + exp(-1.0 * x)); 
}

double relu(double x) {
    return x > 0.0 ? x : 0.0;
}

class BaseNN {
public:
    BaseNN(string modelFile) {
        ifstream infile(modelFile);
        string line;
        W1 = MatrixXd::Zero(1024, 2049);
        B1 = VectorXd::Zero(2049);
        W2 = MatrixXd::Zero(2049, 10);
        infile >> line;
        for(size_t i = 0; i < 1024; i++) {
            infile >> line;
            auto v = splitString(line);
            W1.row(i) = Map<ArrayXd>(v.data(), 2049);
        }
        infile >> line;
        for(size_t i = 0; i < 2049; i++) {
            infile >> line;
            B1(i) = stod(line);
        }
        infile >> line;
        for(size_t i = 0; i < 2049; i++) {
            infile >> line;
            auto v = splitString(line);
            W2.row(i) = Map<ArrayXd>(v.data(), 10);
        }
    }

    MatrixXd ExtractFeature(MatrixXd X) {
        MatrixXd feature = X * W1;
        feature.rowwise() += B1.transpose();
        return Relu(feature);
        //return Sigmod(feature);
    }

    MatrixXd Predict(MatrixXd X) {
        MatrixXd hidden = ExtractFeature(X);
        return hidden * W2;
    }

    MatrixXd Relu(MatrixXd& X) {
        return X.unaryExpr(&relu);
    }

    MatrixXd Sigmod(MatrixXd& X) {
        return X.unaryExpr(&sigmod);
    }
private:
    MatrixXd W1;
    VectorXd B1;
    MatrixXd W2;
};

void Process(int argc, char* argv[]) {
    ZgsConfig config = {
        ._InputDim = 1024,
        ._HiddenDim = 100,
        ._OutputDim = 10,
        ._Beta = 0.1,
        ._Lambda = 0.001,

        ._IterTime = 1000,
        ._GammaA = 0.0001,
        ._GammaB = 0.01
    };

    ZGS zgs(argc, argv, config);
    zgs.Init();

    int id = zgs.GetId();
    int agentCount = zgs.GetAgentCount();
    
    ZGS_LOG(INFO, "%s", "loading data...");
    MatrixXd X = MatrixXd(4800, 1024);
    MatrixXd Y = MatrixXd(4800, 1);
    {
        ifstream infile("./data/test_data/rtraindataFFT10.txt");
        string line;
        for(int i = 0 ; i < 4800 ; i++) {
            infile >> line;
            auto v = splitString(line);
            X.row(i) = Map<ArrayXd>(v.data(), 1024); 
            Y(i, 0) = v[1024] - 1;
        }
    }
    MatrixXd testX = MatrixXd(1200, 1024);
    MatrixXd testY = MatrixXd(1200, 1);
    {
        ifstream infile("./data/test_data/testdataFFT10.txt");
        string line;
        for(int i = 0 ; i < 1200 ; i++) {
            infile >> line;
            auto v = splitString(line);
            testX.row(i) = Map<ArrayXd>(v.data(), 1024); 
            testY(i, 0) = v[1024] - 1;
        }
    }
    // split data for test os-elm-zgs
    long total = X.rows();
    long singleAgentDataCount = total / agentCount;
    {
        MatrixXd localX = X.block(id * singleAgentDataCount, 0, singleAgentDataCount, 1024);
        MatrixXd localY = Y.block(id * singleAgentDataCount, 0, singleAgentDataCount, 1);
        X = localX;
        Y = localY;
    }
    ZGS_LOG(INFO, "local data count is [%d]", X.rows());

    MatrixXd onehotY = Tag2Onehot(Y, 10);
    MatrixXd onehotTestY = Tag2Onehot(testY);

    int times = 50;
    int singleCount = X.rows() / times;

    MatrixXd X1 = X.block(0, 0, singleCount * 10, 1024);
    MatrixXd Y1 = onehotY.block(0, 0, singleCount * 10, 10);
    ZGS_LOG(INFO, "Init Fit X shape [%d, %d] Y shape [%d, %d]", X1.rows(), X1.cols(), Y1.rows(), Y1.cols());
    zgs.InitFit(X1, Y1);
    MatrixXd P = zgs.Predict(testX);
    ZGS_LOG(INFO, "[Agent %d][%d/%d][Acc : %f]", id, 0, times - 1, CalculateAcc(P, testY));
    zgs.Check();
    //for(int i = 0; i < 10; i++) {
    //    cout << Y1.row(i) << endl;
    //    cout << P.row(i) << endl;
    //    cout << endl;
    //}
    
    for(int i = 10; i < times; i++) {
        X1 = X.block(0, 0, singleCount, 1024);
        Y1 = onehotY.block(0, 0, singleCount, 10);
        ZGS_LOG(INFO, "[%d/%d]", i, times - 1);
        zgs.Update(X1, Y1);
        if(i % 5 == 0 || i + 1 == times) {
            P = zgs.Predict(testX);
            ZGS_LOG(INFO, "[Agent %d][%d/%d][Acc : %f]", id, i, times - 1, CalculateAcc(P, testY));
        }
    }
}

void ProcessWithBaseModel(int argc, char* argv[]) {
    ZgsConfig config = {
        ._InputDim = 2049,
        ._HiddenDim = 500,
        ._OutputDim = 10,
        ._Beta = 0.05,
        ._Lambda = 0.01,

        ._IterTime = 1000,
        ._GammaA = 0.0001,
        ._GammaB = 0.01
    };

    ZGS zgs(argc, argv, config);
    zgs.Init();

    int id = zgs.GetId();
    int agentCount = zgs.GetAgentCount();
    
    ZGS_LOG(INFO, "%s", "loading data...");
    BaseNN nn("./data/model_2.txt");
    MatrixXd X = MatrixXd(4800, 1024);
    MatrixXd Y = MatrixXd(4800, 1);
    {
        ifstream infile("./data/test_data/rtraindataFFT10.txt");
        string line;
        for(int i = 0 ; i < 4800 ; i++) {
            infile >> line;
            auto v = splitString(line);
            X.row(i) = Map<ArrayXd>(v.data(), 1024); 
            Y(i, 0) = v[1024] - 1;
        }
    }
    MatrixXd testX = MatrixXd(1200, 1024);
    MatrixXd testFeature = nn.ExtractFeature(testX);
    MatrixXd testY = MatrixXd(1200, 1);
    {
        ifstream infile("./data/test_data/testdataFFT10.txt");
        string line;
        for(int i = 0 ; i < 1200 ; i++) {
            infile >> line;
            auto v = splitString(line);
            testX.row(i) = Map<ArrayXd>(v.data(), 1024); 
            testY(i, 0) = v[1024] - 1;
        }
    }
    // split data for test os-elm-zgs
    long total = X.rows();
    long singleAgentDataCount = total / agentCount;
    {
        MatrixXd localX = X.block(id * singleAgentDataCount, 0, singleAgentDataCount, 1024);
        MatrixXd localY = Y.block(id * singleAgentDataCount, 0, singleAgentDataCount, 1);
        X = localX;
        Y = localY;
    }
    ZGS_LOG(INFO, "local data count is [%d]", X.rows());

    MatrixXd onehotY = Tag2Onehot(Y, 10);
    MatrixXd onehotTestY = Tag2Onehot(testY);

    int times = 50;
    int singleCount = X.rows() / times;
    MatrixXd feature = nn.ExtractFeature(X);

    MatrixXd X1 = feature.block(0, 0, singleCount * 10, 2049);
    MatrixXd Y1 = onehotY.block(0, 0, singleCount * 10, 10);
    ZGS_LOG(INFO, "Init Fit X shape [%d, %d] Y shape [%d, %d]", X1.rows(), X1.cols(), Y1.rows(), Y1.cols());
    zgs.InitFit(X1, Y1);
    MatrixXd P = zgs.Predict(testFeature);
    ZGS_LOG(INFO, "[Agent %d][%d/%d][Acc : %f]", id, 0, times - 1, CalculateAcc(P, testY));
    zgs.Check();
    
    return;
    for(int i = 10; i < times; i++) {
        X1 = feature.block(0, 0, singleCount, 2049);
        Y1 = onehotY.block(0, 0, singleCount, 10);
        ZGS_LOG(INFO, "[%d/%d]", i, times - 1);
        zgs.Update(X1, Y1);
        if(i % 5 == 0 || i + 1 == times) {
            P = zgs.Predict(testFeature);
            ZGS_LOG(INFO, "[Agent %d][%d/%d][Acc : %f]", id, i, times - 1, CalculateAcc(P, testY));
        }
    }
}

void test() {
    ZGS_LOG(INFO, "%s", "loading data...");
    BaseNN nn("./data/model_2.txt");
    MatrixXd X = MatrixXd(1200, 1024);
    MatrixXd Y = MatrixXd(1200, 1);
    ifstream infile("./data/test_data/testdataFFT10.txt");
    string line;
    for(int i = 0 ; i < 1200 ; i++) {
        infile >> line;
        auto v = splitString(line);
        X.row(i) = Map<ArrayXd>(v.data(), 1024); 
        Y(i, 0) = v[1024] - 1;
    }
    ZGS_LOG(INFO, "%s", "predicting...");
    MatrixXd P = nn.Predict(X);
    MatrixXd tagP = Onehot2Tag(P);
    cout << "ACC " << CalculateAcc(P, Y) << endl; 
    for(int i = 0; i < 3; i ++) {
        cout << "P : [" << tagP.row(i) << endl; 
        cout << "P : [" << P.row(i) << endl; 
        cout << "Y : [" << Y.row(i) << endl; 
        cout << endl;
    }
}

int main(int argc, char* argv[]) {
    //test();
    ProcessWithBaseModel(argc, argv);
    return 0;
}
