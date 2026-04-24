#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstring>
#include <cstddef>

template <typename T>
bool check_correctness(const std::vector<T>& buf,
                       const std::vector<T>& ref,
                       double eps = 1e-5) {
    if (buf.size() != ref.size())
        return false;

    for (size_t i = 0; i < buf.size(); ++i) {
        if (std::fabs(static_cast<double>(buf[i]) - static_cast<double>(ref[i])) > eps)
            return false;
    }
    return true;
}



int reduce_scatter_radix_batch(const void *sendbuf, void *recvbuf,
                               MPI_Aint recvcount, MPI_Datatype datatype,
                               MPI_Op op, MPI_Comm comm, int k, int b);




template<typename Func>
void run_r_b(const std::string& name, int r, int b, int count, Func func,
             MPI_Comm comm, std::ofstream& csv, int rank, int nprocs,
             MPI_Datatype datatype, MPI_Op op)
{
    // For MPI_Reduce_scatter_block:
    // - sendbuf has count * nprocs elements
    // - recvbuf has count elements (per rank)
    std::vector<int> sendbuf(count * nprocs), ref_recv(count), recvbuf(count);

    // Simple deterministic init (distinct per rank)
    for (int i = 0; i < count * nprocs; ++i)
        sendbuf[i] = rank * (count * nprocs) + i;

    // Reference using the MPI collective
    // (this computes the ground truth for each rank)
    {
        std::vector<int> tmp_ref(count); // receives this rank's block
        MPI_Reduce_scatter_block(sendbuf.data(), tmp_ref.data(),
                                 count, datatype, op, comm);
        ref_recv = std::move(tmp_ref);
    }

    for (int rep = 0; rep < 20; ++rep) {
        std::fill(recvbuf.begin(), recvbuf.end(), 0);

        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        int err = func(
            reinterpret_cast<char*>(sendbuf.data()),   // char* sendbuf
            reinterpret_cast<char*>(recvbuf.data()),   // char* recvbuf
            count,                                     // recvcount per rank
            datatype,
            op,
            comm,
            r,                                         // radix
            b                                          // batch / local group size
        );

        MPI_Barrier(comm);
        double t1 = MPI_Wtime();

        bool local_correct = (err == MPI_SUCCESS) && check_correctness(recvbuf, ref_recv);

        // Aggregate correctness across ranks (logical AND) and take max time
        int local_ok = local_correct ? 1 : 0, all_ok = 0;
        MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, comm); // 1 only if all 1
        double elapsed = t1 - t0, max_elapsed = 0.0;
        MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (rank == 0) {
            csv << name << "," << r << "," << b << "," << nprocs << "," << count << ","
                << max_elapsed << "," << (all_ok ? 1 : 0) << "\n";
            csv.flush();
        }
    }
}



template<typename Func>
void run_no_params(const std::string& name, int count, Func func,
                      MPI_Comm comm, std::ofstream& csv, int rank, int nprocs,
                      MPI_Datatype datatype, MPI_Op op)
{
    // MPI_Reduce_scatter_block layout:
    // - sendbuf: count * nprocs elements (per rank)
    // - recvbuf: count elements (per rank)
    std::vector<int> sendbuf(count * nprocs), ref_recv(count), recvbuf(count);

    // Deterministic per-rank initialization
    for (int i = 0; i < count * nprocs; ++i)
        sendbuf[i] = rank * (count * nprocs) + i;

    // Reference via MPI_Reduce_scatter_block (this rank's expected block)
    {
        std::vector<int> tmp_ref(count);
        MPI_Reduce_scatter_block(sendbuf.data(), tmp_ref.data(),
                                 count, datatype, op, comm);
        ref_recv = std::move(tmp_ref);
    }

    for (int rep = 0; rep < 20; ++rep) {
        std::fill(recvbuf.begin(), recvbuf.end(), 0);

        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        int err = func(
            reinterpret_cast<char*>(sendbuf.data()),   // char* sendbuf
            reinterpret_cast<char*>(recvbuf.data()),   // char* recvbuf
            count,                                     // recvcount per rank
            datatype,
            op,
            comm
        );

        MPI_Barrier(comm);
        double t1 = MPI_Wtime();

        bool local_correct = (err == MPI_SUCCESS) && check_correctness(recvbuf, ref_recv);

        // Global correctness (AND across ranks) and max time across ranks
        int ok_local = local_correct ? 1 : 0, ok_global = 0;
        MPI_Allreduce(&ok_local, &ok_global, 1, MPI_INT, MPI_MIN, comm);

        double elapsed = t1 - t0, elapsed_max = 0.0;
        MPI_Reduce(&elapsed, &elapsed_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (rank == 0) {
            // Keep CSV schema: algorithm_name,k,b,nprocs,send_count,time,is_correct
            csv << name << "," << 0 << "," << 0 << "," << nprocs << ","
                << count << "," << elapsed_max << "," << (ok_global ? 1 : 0) << "\n";
            csv.flush();
        }
    }
}



int reduce_scatter_standard(char* sendbuf, char* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm){
    return MPI_Reduce_scatter_block(sendbuf, recvbuf, count, datatype, op, comm);
}

long long closestDivisorToSqrt(long long x) {
    long long root = std::sqrt(x);
    for (long long d = root; d >= 1; --d) {
        if (x % d == 0)
            return d; // first divisor <= sqrt(x)
    }
    return 1; // fallback, though this will never happen for x >= 1
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    //
    // Parse arguments:
    //   program <n_iter> [--overwrite] [b=<value>] [base=<value>] [num_nodes=<value>] [radix_increment=<value>]
    //
    int n_iter;
    bool overwrite = false;

    int b = 16;
    int base = 1;
    int num_nodes = 1;
    int radix_increment = 1;

    if (b > 32) b = 32;

    if (argc < 2 || argc > 7) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " <n_iter> [--overwrite] [b=<value>] [base=<value>] "
                         "[num_nodes=<value>] [radix_increment=<value>]\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    n_iter = std::atoi(argv[1]);

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--overwrite") == 0) {
            overwrite = true;
        } else if (std::strncmp(argv[i], "b=", 2) == 0) {
            b = std::atoi(argv[i] + 2);
        } else if (std::strncmp(argv[i], "base=", 5) == 0) {
            base = std::atoi(argv[i] + 5);
        } else if (std::strncmp(argv[i], "num_nodes=", 10) == 0) {
            num_nodes = std::atoi(argv[i] + 10);
        } else if (std::strncmp(argv[i], "radix_increment=", 16) == 0) {
            radix_increment = std::atoi(argv[i] + 16);
        } else {
            if (rank == 0) {
                std::cerr << "Unknown parameter: " << argv[i] << "\n";
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    //
    // Rank 0 manages the CSV file.
    //
    std::ofstream csv;
    if (rank == 0) {
        std::string filename = "results" + std::to_string(nprocs/num_nodes) +"_"+ std::to_string(num_nodes)+ "_" + std::to_string(b) + ".csv";
        bool exists = std::ifstream(filename).good();

        if (overwrite || !exists) {
            csv.open(filename, std::ios::out | std::ios::trunc);
            csv << "algorithm_name,k,b,nprocs,send_count,time,is_correct\n";
        } else {
            csv.open(filename, std::ios::out | std::ios::app);
        }
    }

    // Choose the reduction op (adjust if you benchmark different ops)
    MPI_Op op = MPI_SUM;

    for (int i = 0; i < n_iter; ++i) {
        int count = base << i;

        // --- Algorithms with radix r and batch b (reduce-scatter, block variant)
        for (int r = 2; r < b; r += radix_increment) {
            run_r_b("reduce_scatter_radix_batch",
                    r, b, count,
                    reduce_scatter_radix_batch,
                    MPI_COMM_WORLD, csv, rank, nprocs,
                    MPI_INT, op);
        }

        // --- Algorithm without r/b (standard reduce-scatter)
        run_no_params("reduce_scatter_standard",
                         count,
                         reduce_scatter_standard,
                         MPI_COMM_WORLD, csv, rank, nprocs,
                         MPI_INT, op);
    }

    if (rank == 0) csv.close();
    MPI_Finalize();
    return EXIT_SUCCESS;
}