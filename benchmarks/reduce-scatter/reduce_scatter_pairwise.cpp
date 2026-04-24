#include "reduce_scatter.hpp"


int MPICH_reduce_scatter_pairwise(const void *sendbuf, void *recvbuf,
                               MPI_Aint recvcount, MPI_Datatype datatype,
                               MPI_Op op, MPI_Comm comm){

    int rank, comm_size, i;
    int extent;
    void *tmp_recvbuf;
    int mpi_errno = MPI_SUCCESS;
    int src, dst;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);
    MPI_Type_size(datatype, &extent);

    int is_commutative;
    MPI_Op_commutative(op, &is_commutative);
    

    int *disps;

    disps = (int*) malloc(sizeof(int) * comm_size);

    for(i = 0; i < comm_size; i++) {
        disps[i] = i * recvcount;
    }


    if(sendbuf != MPI_IN_PLACE){
        memcpy((char*)recvbuf, (char*)sendbuf + disps[rank] * extent, recvcount * extent);
    }

    tmp_recvbuf = (char*) malloc((extent+1) * recvcount);

    for (i = 1; i < comm_size; i++) {
        src = (rank - i + comm_size) % comm_size;
        dst = (rank + i) % comm_size;

        if(sendbuf != MPI_IN_PLACE){
            mpi_errno = MPI_Sendrecv((char*)sendbuf + disps[dst] * extent, recvcount, datatype, dst, 0,
                                     tmp_recvbuf, recvcount, datatype, src, 0, comm, MPI_STATUS_IGNORE);
        }else{
            mpi_errno = MPI_Sendrecv((char*)recvbuf + disps[dst] * extent, recvcount, datatype, dst, 0,
                                     tmp_recvbuf, recvcount, datatype, src, 0, comm, MPI_STATUS_IGNORE); 
        }
        if(mpi_errno != MPI_SUCCESS) {
            std::cout<<"Error in MPI_Sendrecv"<<std::endl;
            goto fn_fail;
        }

        if(sendbuf != MPI_IN_PLACE){
            mpi_errno = MPI_Reduce_local(tmp_recvbuf, recvbuf, recvcount, datatype, op);
        }else{
            mpi_errno = MPI_Reduce_local(tmp_recvbuf, (char*)recvbuf + disps[rank] * extent, recvcount, datatype, op);
        }
        if(mpi_errno != MPI_SUCCESS) {
            std::cout<<"Error in MPI_Reduce_local"<<std::endl;
            goto fn_fail;
        }
    }

    if((sendbuf == MPI_IN_PLACE)&& (rank != 0)){
        memcpy(recvbuf, (char*)recvbuf + disps[rank] * extent, recvcount * extent);
    }

    fn_exit:
        free(tmp_recvbuf);
        free(disps);
        return mpi_errno;
    fn_fail:
        goto fn_exit;
}


#ifdef DEBUG_MODE
static int almost_equal(double a, double b) {
    double diff = std::fabs(a - b);
    double tol  = 1e-10 * std::max({1.0, std::fabs(a), std::fabs(b)});
    return diff <= tol;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    const int counts[] = {1, 3, 16, 31};
    int failures = 0;

    for (int recvcount : counts) {
        const int total = recvcount * P;

        // INT / SUM (regular)
        {
            std::vector<int> send(total), ref(recvcount), got(recvcount);
            for (int j = 0; j < total; ++j) send[j] = rank * 1000 + j;

            MPI_Reduce_scatter_block(send.data(), ref.data(), recvcount, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            int rc = MPICH_reduce_scatter_pairwise(send.data(), got.data(), recvcount, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (rc != MPI_SUCCESS) { if (rank==0) std::fprintf(stderr, "[INT SUM regular] error %d\n", rc); ++failures; }
            else {
                for (int i = 0; i < recvcount; ++i)
                    if (got[i] != ref[i]) { std::fprintf(stderr, "Rank %d: INT regular mismatch idx %d\n", rank, i); ++failures; break; }
            }
        }

        // INT / SUM (in-place)
        {
            std::vector<int> inout(total), ref(recvcount);
            for (int j = 0; j < total; ++j) inout[j] = rank * 1000 + j;

            MPI_Reduce_scatter_block(inout.data(), ref.data(), recvcount, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            int rc = MPICH_reduce_scatter_pairwise(MPI_IN_PLACE, inout.data(), recvcount, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (rc != MPI_SUCCESS) { if (rank==0) std::fprintf(stderr, "[INT SUM in-place] error %d\n", rc); ++failures; }
            else {
                for (int i = 0; i < recvcount; ++i)
                    if (inout[i] != ref[i]) { std::fprintf(stderr, "Rank %d: INT in-place mismatch idx %d\n", rank, i); ++failures; break; }
            }
        }

        // DOUBLE / MAX (regular)
        {
            std::vector<double> send(total), ref(recvcount), got(recvcount);
            for (int j = 0; j < total; ++j) send[j] = (double)rank + 0.001 * j;

            MPI_Reduce_scatter_block(send.data(), ref.data(), recvcount, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            int rc = MPICH_reduce_scatter_pairwise(send.data(), got.data(), recvcount, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rc != MPI_SUCCESS) { if (rank==0) std::fprintf(stderr, "[DOUBLE MAX regular] error %d\n", rc); ++failures; }
            else {
                for (int i = 0; i < recvcount; ++i)
                    if (!almost_equal(got[i], ref[i])) { std::fprintf(stderr, "Rank %d: DOUBLE regular mismatch idx %d\n", rank, i); ++failures; break; }
            }
        }

        // DOUBLE / MAX (in-place)
        {
            std::vector<double> inout(total), ref(recvcount);
            for (int j = 0; j < total; ++j) inout[j] = (double)rank + 0.001 * j;

            MPI_Reduce_scatter_block(inout.data(), ref.data(), recvcount, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            int rc = MPICH_reduce_scatter_pairwise(MPI_IN_PLACE, inout.data(), recvcount, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rc != MPI_SUCCESS) { if (rank==0) std::fprintf(stderr, "[DOUBLE MAX in-place] error %d\n", rc); ++failures; }
            else {
                for (int i = 0; i < recvcount; ++i)
                    if (!almost_equal(inout[i], ref[i])) { std::fprintf(stderr, "Rank %d: DOUBLE in-place mismatch idx %d\n", rank, i); ++failures; break; }
            }
        }

        if (rank == 0) std::printf("[count=%d] tests done.\n", recvcount);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        if (failures == 0) std::printf("✅ All tests passed.\n");
        else               std::printf("❌ %d failure(s) detected.\n", failures);
    }

    MPI_Finalize();
    return failures ? 1 : 0;
}

#endif