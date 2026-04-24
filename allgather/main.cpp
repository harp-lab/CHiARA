#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <string>
#include <cstring>


int allgather_radix_batch(char* sendbuf, int sendcount, MPI_Datatype datatype, char* recvbuf, MPI_Comm comm, int k, int b);



// Compare output buffer to reference buffer within tolerance
bool check_correctness(const std::vector<int>& buf,
                       const std::vector<int>& ref,
                       double eps = 1e-9) {
    for (size_t i = 0; i < buf.size(); ++i) {
        if ((buf[i] != ref[i]))
            return false;
    }
    return true;
}



template<typename Func>
void run_k_b(const std::string& name, int k, int b, int count, Func func,
            MPI_Comm comm, std::ofstream& csv, int rank, int nprocs, MPI_Datatype datatype) {
    std::vector<int> sendbuf(count), refbuf(count*nprocs), recvbuf(count*nprocs);
    // Unique initialization
    for (int i = 0; i < count; ++i)
        sendbuf[i] = rank * static_cast<int>(count) + i;

    // Reference result via MPI_Allgather
    MPI_Allgather(sendbuf.data(), count, datatype,
                  refbuf.data(), count, datatype, comm);

    for (int rep = 0; rep < 50; ++rep) {
        std::fill(recvbuf.begin(), recvbuf.end(), 0);
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        int err = func(reinterpret_cast<char*>(sendbuf.data()),
                       count, datatype,
                       reinterpret_cast<char*>(recvbuf.data()),
                    comm,
                       k, b);

        MPI_Barrier(comm);
        double t1 = MPI_Wtime();

        bool correct = (err == MPI_SUCCESS)
                       && check_correctness(recvbuf, refbuf);
        if (rank == 0) {
            csv << name << "," << k << ","<<b<<","<<nprocs<<"," << count << ","
                << (t1 - t0) << "," << (correct?1:0) << "\n";
            csv.flush();
        }
    }
}

template<typename Func>
void run_no_k(const std::string& name, int count, Func func,
            MPI_Comm comm, std::ofstream& csv, int rank, int nprocs, MPI_Datatype datatype) {
    std::vector<int> sendbuf(count), refbuf(count*nprocs), recvbuf(count*nprocs);
    // Unique initialization
    for (int i = 0; i < count; ++i)
        sendbuf[i] = rank * static_cast<int>(count) + i;

    // Reference result via MPI_Allgather
    MPI_Allgather(sendbuf.data(), count, datatype,
                  refbuf.data(), count, datatype, comm);

    for (int rep = 0; rep < 50; ++rep) {
        std::fill(recvbuf.begin(), recvbuf.end(), 0);
        MPI_Barrier(comm);
        double t0 = MPI_Wtime();

        int err = func(reinterpret_cast<char*>(sendbuf.data()),
                       count, datatype,
                       reinterpret_cast<char*>(recvbuf.data()), count,
                       comm);

        MPI_Barrier(comm);
        double t1 = MPI_Wtime();

        bool correct = (err == MPI_SUCCESS)
                       && check_correctness(recvbuf, refbuf);
        if (rank == 0) {
            csv << name << "," << "0" << ","<< "0" << ","<<nprocs<<"," << count << ","
                << (t1 - t0) << "," << (correct?1:0) << "\n";
            csv.flush();
        }
    }
}

int allgather_standard(char* sendbuf, int sendcount, MPI_Datatype datatype, char* recvbuf, int recvcount, MPI_Comm comm){
    return MPI_Allgather(sendbuf, sendcount, datatype, recvbuf, recvcount, datatype, comm);
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
    // Only rank‐0 manages the CSV file.
    //
    std::ofstream csv;
    if (rank == 0) {
        // check whether results.csv already exists
        std::string filename = "results" + std::to_string(nprocs/num_nodes) +"_"+ std::to_string(num_nodes)+ "_" + std::to_string(b) + ".csv";
        bool exists = std::ifstream(filename).good();

        if (overwrite || !exists) {
            // truncate (or create) + write header
            csv.open(filename, std::ios::out | std::ios::trunc);
            csv << "algorithm_name,k,b,nprocs,send_count,time,is_correct\n";
        }
        else {
            // append, no header
            csv.open(filename, std::ios::out | std::ios::app);
        }
    }

    for (int i = 0; i < n_iter; ++i) {
        int count = base << i;

        // Algorithms with k + single_phase_recv
        for (int k = 2; k < b; k+=radix_increment) {
        run_k_b("allgather_radix_batch", k, b, count, allgather_radix_batch,
                MPI_COMM_WORLD, csv, rank, nprocs, MPI_INT);

        }

        // Algorithms without k
        ///rund the standard allgather implementation
        run_no_k("allgather_standard", count, allgather_standard,
                MPI_COMM_WORLD, csv, rank, nprocs, MPI_INT);
    }

    if (rank == 0) csv.close();
    MPI_Finalize();
    return EXIT_SUCCESS;
}