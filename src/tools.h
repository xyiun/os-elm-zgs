#ifndef OS_ELM_ZGS_SRC_TOOLS_H
#define OS_ELM_ZGS_SRC_TOOLS_H

#include <src/common.h>

namespace OsElmZgs {

// calculate MSE
// MSE = sum( ||predict[i] - tag[i]||_2 ) / N
double CalculateMSE(Eigen::MatrixXd& predict, Eigen::MatrixXd& tag);

// caldulate ACC
// ACC = sum( argmax(predict[i]) == tag[i] ? 1 : 0 ) / N
double CalculateAcc(Eigen::MatrixXd& predict, Eigen::MatrixXd& tag);

// convert tag vector into onehot encoding format
Eigen::MatrixXd Tag2Onehot(Eigen::MatrixXd& tag, int categoryCount=-1);

// conveert onehot encoding tag into vector
Eigen::MatrixXd Onehot2Tag(Eigen::MatrixXd& onehot, int categoryCount=-1);

// get the index of max element in a row
size_t ArgMaxOfRow(Eigen::MatrixXd row);

// get max element in a row
double MaxOfRow(Eigen::MatrixXd row);

};
#endif // OS_ELM_ZGS_SRC_TOOLS_H
