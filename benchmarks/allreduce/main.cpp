#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstring>

static bool check_correctness(const std::vector<double>& buf,
                              const std::vector<double>& ref,
                              double eps = 1e-9) {
    for (size_t i = 0; i < buf.size(); ++i)
        if (std::fabs(buf[i] - ref[i]) > eps)
            return false;
    return true;
}

// -----------------------------------------------------------------------------
// MPICH internal Allreduce variants
// -----------------------------------------------------------------------------
int MPICH_Allreduce_ring(const char* sendbuf, char* recvbuf, int count,
                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int MPICH_Allreduce_reduce_scatter_allgather(const char* sendbuf, char* recvbuf, int count,
                                             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int MPICH_Allreduce_recursive_multiplying(const char* sendbuf, char* recvbuf, int count,
                                         MPI_Datatype datatype, MPI_Op op, MPI_Comm comm, int k);

int MPICH_Allreduce_recursive_doubling(const char* sendbuf, char* recvbuf, int count,
                                       MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);



// Baseline MPI_Allreduce wrapper
static int MPICH_Allreduce_block(const char* sendbuf, char* recvbuf, int count,
                                 MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    return MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

// -----------------------------------------------------------------------------
// Harness runners
// CSV columns: algorithm_name,k,nprocs,count,time,is_correct
// -----------------------------------------------------------------------------
template<typename Func>
static void run_no_k(const std::string& name, int count, Func func,
                     MPI_Comm comm, std::ofstream& csv,
                     int rank, int nprocs) {

    std::vector<double> sendbuf(count), refbuf(count), recvbuf(count);

    for (int i = 0; i < count; ++i)
        sendbuf[i] = rank * static_cast<double>(count) + i;

    MPI_Allreduce(sendbuf.data(), refbuf.data(),
                  count, MPI_DOUBLE, MPI_SUM, comm);

    for (int rep = 0; rep < 5; ++rep) {
        std::fill(recvbuf.begin(), recvbuf.end(), 0.0);

        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        int err = func(reinterpret_cast<const char*>(sendbuf.data()),
                       reinterpret_cast<char*>(recvbuf.data()),
                       count, MPI_DOUBLE, MPI_SUM, comm);

        MPI_Barrier(comm);
        double t1 = MPI_Wtime();

        bool correct = (err == MPI_SUCCESS) &&
                       check_correctness(recvbuf, refbuf);

        if (rank == 0) {
            csv << name << ",0," << nprocs << "," << count/nprocs << ","
                << (t1 - t0) << "," << (correct ? 1 : 0) << "\n";
            csv.flush();
        }
    }
}

template<typename Func>
static void run_k_only(const std::string& name, int k, int count, Func func,
                       MPI_Comm comm, std::ofstream& csv,
                       int rank, int nprocs) {

    std::vector<double> sendbuf(count), refbuf(count), recvbuf(count);

    for (int i = 0; i < count; ++i)
        sendbuf[i] = rank * static_cast<double>(count) + i;

    MPI_Allreduce(sendbuf.data(), refbuf.data(),
                  count, MPI_DOUBLE, MPI_SUM, comm);

    for (int rep = 0; rep < 2; ++rep) {
        std::fill(recvbuf.begin(), recvbuf.end(), 0.0);

        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        int err = func(reinterpret_cast<const char*>(sendbuf.data()),
                       reinterpret_cast<char*>(recvbuf.data()),
                       count, MPI_DOUBLE, MPI_SUM, comm, k);

        MPI_Barrier(comm);
        double t1 = MPI_Wtime();

        bool correct = (err == MPI_SUCCESS) &&
                       check_correctness(recvbuf, refbuf);

        if (rank == 0) {
            csv << name << "," << k << "," << nprocs << "," << count/nprocs << ","
                << (t1 - t0) << "," << (correct ? 1 : 0) << "\n";
            csv.flush();
        }
    }
}

// -----------------------------------------------------------------------------
// main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // program <n_iter> [--overwrite]
    if (argc < 2 || argc > 3) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0]
                      << " <n_iter> [--overwrite]\n";
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int  n_iter    = std::atoi(argv[1]);
    const bool overwrite = (argc == 3 && std::strcmp(argv[2], "--overwrite") == 0);

    std::ofstream csv;
    if (rank == 0) {
        std::string fname =
            "allreduce_results" + std::to_string(nprocs / 32) + ".csv";

        bool exists = std::ifstream(fname).good();
        if (overwrite || !exists) {
            csv.open(fname, std::ios::out | std::ios::trunc);
            csv << "algorithm_name,k,nprocs,send_count,time,is_correct\n";
        } else {
            csv.open(fname, std::ios::out | std::ios::app);
        }
    }

    const int base = 8*nprocs;
    for (int i = 0; i < n_iter; ++i) {
        int count = base << i;

        // ---- k-parameterized
        for (int k = 2; k <= nprocs; k <<= 1) {
            run_k_only("MPICH_Allreduce_recursive_multiplying",
                       k, count,
                       MPICH_Allreduce_recursive_multiplying,
                       MPI_COMM_WORLD, csv, rank, nprocs);
        }

        // ---- no-k algorithms
        run_no_k("MPICH_Allreduce_ring",
                 count, MPICH_Allreduce_ring,
                 MPI_COMM_WORLD, csv, rank, nprocs);

        run_no_k("MPICH_Allreduce_reduce_scatter_allgather",
                 count, MPICH_Allreduce_reduce_scatter_allgather,
                 MPI_COMM_WORLD, csv, rank, nprocs);

        run_no_k("MPICH_Allreduce_recursive_doubling",
                 count, MPICH_Allreduce_recursive_doubling,
                 MPI_COMM_WORLD, csv, rank, nprocs);

        run_no_k("MPI_Allreduce",
                 count, MPICH_Allreduce_block,
                 MPI_COMM_WORLD, csv, rank, nprocs);
    }

    if (rank == 0) csv.close();
    MPI_Finalize();
    return EXIT_SUCCESS;
}