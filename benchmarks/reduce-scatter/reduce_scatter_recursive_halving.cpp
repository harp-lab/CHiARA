#include "reduce_scatter.hpp"

#define MAX_RADIX 8

enum { RS_TAG = 798423 };

int MPICH_reduce_scatter_rec_halving(const char* sendbuf, char* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

    int rank, comm_size, i;
    int extent;
    int *disps;
    void *tmp_recvbuf, *tmp_results;
    int mpi_errno = MPI_SUCCESS;
    int total_count, dst;
    int mask;
    int rem, newdst, send_idx, recv_idx, last_idx, send_cnt, recv_cnt;
    int pof2, old_i, newrank;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);
    MPI_Type_size(datatype, &extent);

    disps = (int*) malloc(sizeof(int) * comm_size);

    total_count = comm_size * count;
    for (i = 0; i < comm_size; i++) {
        disps[i] = i * count;
    }


    tmp_recvbuf = (char*) malloc(extent * total_count);
    tmp_results = (char*) malloc(extent * total_count);

    if(sendbuf != MPI_IN_PLACE) {
        memcpy((char*)tmp_results, sendbuf, extent * total_count);
    } else {
        memcpy((char*)tmp_results, recvbuf, extent * total_count);
    }

    pof2 = 1;
    while (pof2 <= comm_size) {
        pof2 *= 2;
    
    }

    pof2 = pof2 == comm_size ? pof2 : pof2 / 2;

    rem = comm_size - pof2;

    if (rank < 2 * rem) {
        if (rank % 2 == 0) { 
            mpi_errno = MPI_Send(tmp_results, total_count, datatype, rank + 1, RS_TAG, comm);
            newrank = -1;

        }else{
            mpi_errno = MPI_Recv(tmp_recvbuf, total_count, datatype, rank - 1, RS_TAG, comm, MPI_STATUS_IGNORE);


            mpi_errno = MPI_Reduce_local(tmp_recvbuf, tmp_results, total_count, datatype, op);
            newrank = rank / 2;

        }

    }else{
        newrank = rank - rem;
    }

    if (newrank != -1){

        int *newcnts, *newdisps;
        newcnts = (int*) malloc(sizeof(int) * pof2);
        newdisps = (int*) malloc(sizeof(int) * pof2);

        for (i = 0; i < pof2; i++) {
            old_i = (i < rem) ? i * 2 + 1 : i + rem;
            if (old_i < 2 * rem) {
                newcnts[i] = count * 2;
            } else {
                newcnts[i] = count;
            }
        }
        if(pof2)
            newdisps[0] = 0;
        for (i = 1; i < pof2; i++) {
            newdisps[i] = newdisps[i - 1] + newcnts[i - 1];
        }

        mask = pof2 >> 1;
        send_idx = recv_idx = 0;
        last_idx = pof2;
        while (mask > 0) {
            newdst = newrank ^ mask;
            /* find real rank of dest */
            dst = (newdst < rem) ? newdst * 2 + 1 : newdst + rem;

            send_cnt = recv_cnt = 0;
            if (newrank < newdst) {
                send_idx = recv_idx + mask;
                for (i = send_idx; i < last_idx; i++)
                    send_cnt += newcnts[i];
                for (i = recv_idx; i < send_idx; i++)
                    recv_cnt += newcnts[i];
            } else {
                recv_idx = send_idx + mask;
                for (i = send_idx; i < recv_idx; i++)
                    send_cnt += newcnts[i];
                for (i = recv_idx; i < last_idx; i++)
                    recv_cnt += newcnts[i];
            }

            if((send_cnt != 0) && (recv_cnt != 0)){
                mpi_errno = MPI_Sendrecv((char*)tmp_results + newdisps[send_idx] * extent, send_cnt, datatype, dst, RS_TAG,
                                    (char*)tmp_recvbuf + newdisps[recv_idx] * extent, recv_cnt, datatype, dst, RS_TAG, comm, MPI_STATUS_IGNORE);
            }else if ((send_cnt == 0) && (recv_cnt != 0)){
                mpi_errno = MPI_Recv((char*)tmp_recvbuf + newdisps[recv_idx] * extent, recv_cnt, datatype, dst, RS_TAG, comm, MPI_STATUS_IGNORE);
            }else if ((send_cnt != 0) && (recv_cnt == 0)){
                mpi_errno = MPI_Send((char*)tmp_results + newdisps[send_idx] * extent, send_cnt, datatype, dst, RS_TAG, comm);
            }

            if(recv_cnt){
                mpi_errno = MPI_Reduce_local((char*)tmp_recvbuf + newdisps[recv_idx] * extent,
                                        (char*)tmp_results + newdisps[recv_idx] * extent, recv_cnt, datatype, op);
            }

            send_idx = recv_idx;
            last_idx = recv_idx + mask;
            mask >>=1;
        }

        memcpy(recvbuf, (char*)tmp_results + disps[rank] * extent, count * extent);

        free(newcnts);
        free(newdisps);

    }

    if(rank < 2 * rem){
        if (rank % 2){
            mpi_errno = MPI_Send((char *)tmp_results + disps[rank-1] * extent, count, datatype, rank - 1, RS_TAG, comm);
        }else{
            mpi_errno = MPI_Recv(recvbuf, count, datatype, rank + 1, RS_TAG, comm, MPI_STATUS_IGNORE);
        }
    }

    free(tmp_recvbuf);
    free(tmp_results);
    free(disps);

    return mpi_errno;



}



#ifdef DEBUG_MODE

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <r> <b>\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Parse r and b
    int r = std::atoi(argv[1]);
    int b = std::atoi(argv[2]);

    // Each rank will receive `count` ints.
    const int count =4;
    // Total elements in sendbuf = count * size
    const int total_send_elems = count * size;

    // Allocate sendbuf and recvbuf with vectors
    std::vector<int> sendbuf(total_send_elems);
    std::vector<int> recvbuf(total_send_elems);

    // Fill sendbuf so that element (j*count + k) = rank*1000 + (j*count + k)
    for (int j = 0; j < size; ++j) {
        for (int k = 0; k < count; ++k) {
            int idx = j * count + k;
            sendbuf[idx] = rank * 1000 + idx;
        }
    }

    //
    // === Gather all initial send-buffers at rank 0 ===
    //
    // We want to write each rank’s initial sendbuf into the file.
    std::vector<int> all_sendbuf; 
    if (rank == 0) {
        // root will hold size * total_send_elems ints
        all_sendbuf.resize(size * total_send_elems);
    }
    MPI_Gather(
        sendbuf.data(),                 // send buffer
        total_send_elems,               // sendcount
        MPI_INT,
        (rank == 0 ? all_sendbuf.data() : nullptr),
        total_send_elems,               // recvcount at root
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    //
    // === Now call your reduce_scatter_radix_block ===
    //
    int mpi_err = MPICH_reduce_scatter_rec_halving(
        reinterpret_cast<char*>(sendbuf.data()),
        reinterpret_cast<char*>(recvbuf.data()),
        count,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD
    );
    // Create recvcounts array - each process gets 'count' elements
    // int* recvcounts = new int[size];
    // for (int i = 0; i < size; i++) {
    //     recvcounts[i] = count;
    // }
    
    // int mpi_err = MPI_Reduce_scatter(sendbuf.data(), recvbuf.data(), recvcounts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // // Don't forget to free the memory
    // delete[] recvcounts;
    if (mpi_err != MPI_SUCCESS) {
        std::cerr << "Rank " << rank << ": reduce_scatter_radix_block failed\n";
        MPI_Abort(MPI_COMM_WORLD, mpi_err);
    }

    //
    // === Gather all final recv-buffers at rank 0 ===
    //
    std::vector<int> all_recvbuf;
    if (rank == 0) {
        all_recvbuf.resize(size * total_send_elems);
    }
    MPI_Gather(
        recvbuf.data(),   // send buffer
        count,
        MPI_INT,
        (rank == 0 ? all_recvbuf.data() : nullptr),
        count,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    //
    // === Rank 0 writes everything to "all_buffers.txt" ===
    //
    if (rank == 0) {
        std::ofstream ofs("all_buffers.txt");
        if (!ofs.is_open()) {
            std::cerr << "Could not open all_buffers.txt for writing\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // 1) Write initial send-buffers
        ofs << "## initial send buffers (r=" << r << ", b=" << b << ")\n\n";
        for (int src = 0; src < size; ++src) {
            ofs << "SendRank " 
                << (src < 10 ? " " : "") << src << ":";

            // For rank=src, its sendbuf occupies:
            //   all_sendbuf[ src * total_send_elems + 0 ... src * total_send_elems + (total_send_elems-1) ]
            int base = src * total_send_elems;
            for (int idx = 0; idx < total_send_elems; ++idx) {
                ofs << " " << all_sendbuf[base + idx];
            }
            ofs << "\n";
        }
        ofs << "\n";

        // 2) Write final recv-buffers
        ofs << "## reduce_scatter_radix_block results (r=" << r << ", b=" << b << ")\n"
            << "## Each row = MPI rank, columns = recvbuf[0.." << (count - 1) << "]\n\n";

        for (int src = 0; src < size; ++src) {
            ofs << "Rank " << (src < 10 ? " " : "") << src << ":";
            for (int k = 0; k < count; ++k) {
                int val = all_recvbuf[src * count + k];
                ofs << " " << val;
            }
            ofs << "\n";
        }

        ofs.close();
        std::cout << "All send & recv buffers written to all_buffers.txt\n";
    }

    MPI_Finalize();
    return 0;
}

#endif