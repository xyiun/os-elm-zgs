#include "mpi.h"
#include <src/log.h>
#include <iostream>
#include <src/common.h>
using namespace std;

void TestMpi(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int processCount; // 目前连入集群的节点数量
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    int myId; // 当前机器的ID
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    int totalCount; // 总节点数
    MPI_Comm_rank(MPI_COMM_WORLD, &totalCount);
    char machineName[MPI_MAX_PROCESSOR_NAME];
    int namelen;
    MPI_Get_processor_name(machineName, &namelen);
    ZGS_LOG(INFO, "共创建[%d]个节点", processCount);
    ZGS_LOG(INFO, "当前节点的ID是[%d] 在机器[%s]上", myId, machineName);



    Eigen::MatrixXd A = Eigen::MatrixXd::Random(2,2);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(2,10);
    MPI_Allgather(A.data(), A.rows() * A.cols(), MPI_DOUBLE,
            B.data(), A.rows() * A.cols(), MPI_DOUBLE, MPI_COMM_WORLD);

    if(myId == 0) {
        cout << A << endl << endl;
        cout << B << endl;
    }
}

int main(int argc, char* argv[]) {
    TestMpi(argc, argv); 
    return 0;
}
