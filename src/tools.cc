#include <src/tools.h>

namespace OsElmZgs {

double CalculateMSE(Eigen::MatrixXd& predict, Eigen::MatrixXd& tag) {
    assert (predict.rows() != 0);
    assert (predict.rows() == tag.rows());
    assert (predict.cols() == tag.cols());

    double total = 0.0;
    for(long i = 0; i < predict.rows(); i++) {
        total += (predict.row(i) - tag.row(i)).norm();
    }
    return total / predict.rows();
}

size_t ArgMaxOfRow(Eigen::MatrixXd row) {
    size_t ret = 0;
    double m = row(0, 0);
    for(long i = 1; i < row.cols(); i++) {
        if(row(0, i) > m) {
            m = row(0, i);
            ret = i;
        }
    }
    return ret;
}

double CalculateAcc(Eigen::MatrixXd& predict, Eigen::MatrixXd& tag) {
    assert (predict.rows() != 0);
    assert (tag.cols() == 1);
    assert (predict.rows() == tag.rows());

    size_t accCount = 0;
    for(long i = 0; i < predict.rows(); i++) {
        int category = (int)tag(i, 0);
        assert (category >= 0 && category < predict.cols());
        size_t p = ArgMaxOfRow(predict.row(i));

        if(p == (size_t)category) {
            accCount += 1;
        }
    }
    return accCount * 1.0 / predict.rows();
}

double MaxOfRow(Eigen::MatrixXd row) {
    assert (row.cols() != 0);
    double ret = row(0, 0);
    for(long i = 1; i < row.cols(); i++) {
        if(row(0, i) > ret) {
            ret = row(0, i);
        }
    }
    return ret;
}

Eigen::MatrixXd Tag2Onehot(Eigen::MatrixXd& tag, int categoryCount) {
    assert (tag.rows() != 0);
    assert (tag.cols() == 1);
    double maxTag = tag(0, 0);
    assert((int)maxTag >= 0);
    for(long i = 1; i < tag.rows(); i++) {
        assert ((int)tag(i, 0) >= 0); 
        if(tag(i, 0) > maxTag) {
            maxTag = tag(i, 0);
        }
    }
    if(categoryCount > 0) {
        assert ((int)maxTag < categoryCount);
    } else {
        categoryCount = (int)maxTag + 1;
    }
    Eigen::MatrixXd oneHot = Eigen::MatrixXd::Zero(tag.rows(), categoryCount);
    for(long i = 0; i < tag.rows(); i++) {
        size_t index = (size_t)tag(i, 0);
        oneHot(i, index) = 1.0;
    }
    return oneHot;
}

Eigen::MatrixXd Onehot2Tag(Eigen::MatrixXd& onehot, int categoryCount) {
    assert (onehot.rows() > 0);
    if(categoryCount > 0) {
        assert (onehot.cols() == categoryCount);
    }
    Eigen::MatrixXd tag = Eigen::MatrixXd::Zero(onehot.rows(), 1);
    for(long i = 0; i < onehot.rows(); i++) {
        size_t t = ArgMaxOfRow(onehot.row(i));
        tag(i, 0) = (double)t;
    }
    return tag;
}


};
