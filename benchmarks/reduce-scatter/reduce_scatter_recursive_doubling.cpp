#include "reduce_scatter.hpp"

static inline MPI_Aint aint_max(MPI_Aint a, MPI_Aint b) { return (a > b) ? a : b; }

struct TypeGuard {
    MPI_Datatype t = MPI_DATATYPE_NULL;
    ~TypeGuard() { if (t != MPI_DATATYPE_NULL) MPI_Type_free(&t); }
};

int MPICH_reduce_scatter_rec_doubling(const void *sendbuf, void *recvbuf,
                            MPI_Aint recvcount, MPI_Datatype datatype,
                            MPI_Op op, MPI_Comm comm)
{
    int rank = 0, P = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    MPI_Aint lb_std, extent_std, true_lb, true_extent;
    MPI_Type_get_extent(datatype, &lb_std, &extent_std);
    MPI_Type_get_true_extent(datatype, &true_lb, &true_extent);
    MPI_Aint extent = aint_max(extent_std, true_extent);

    const MPI_Aint total_count = recvcount * (MPI_Aint)P;

    // allocate raw bytes; we will "shift" by -true_lb to honor negative lower bounds
    std::vector<char> tmp_recvbuf_raw((size_t)(total_count * extent + true_extent));
    std::vector<char> tmp_results_raw((size_t)(total_count * extent + true_extent));
    void *tmp_recvbuf = (void*)(tmp_recvbuf_raw.data()   - true_lb);
    void *tmp_results = (void*)(tmp_results_raw.data()   - true_lb);

    // initialize tmp_results with input (pure local copy, no comm)
    {
        const char *src_base = nullptr;
        if (sendbuf != MPI_IN_PLACE) src_base = (const char*)sendbuf;
        else                         src_base = (const char*)recvbuf;

        // adjust by -true_lb to match how tmp_* pointers are adjusted
        const char *src = src_base - true_lb;
        std::memcpy(tmp_results_raw.data(), src, (size_t)(total_count * extent));
    }

    // displacements (in elements) for the final copy-out
    std::vector<MPI_Aint> disps((size_t)P);
    for (int i = 0; i < P; ++i) disps[i] = (MPI_Aint)i * recvcount;

    int is_commutative = 0;
    MPI_Op_commutative(op, &is_commutative);

    int mask = 1;
    int stage = 0;

    while (mask < P) {
        const int dst = rank ^ mask;

        const int dst_tree_root = (dst  >> stage) << stage;
        const int my_tree_root  = (rank >> stage) << stage;

        // ---- Build send/recv hindexed datatypes (two blocks each) ----
        MPI_Aint bl_send[2], dis_send_elems[2];
        bl_send[0] = (MPI_Aint)my_tree_root * recvcount;
        bl_send[1] = ((MPI_Aint)P - (MPI_Aint)(my_tree_root + mask)) * recvcount;
        if (bl_send[1] < 0) bl_send[1] = 0;

        dis_send_elems[0] = 0;
        {
            MPI_Aint gap_ranks = (MPI_Aint)(((my_tree_root + mask) < P ? (my_tree_root + mask) : P) - my_tree_root);
            dis_send_elems[1]  = bl_send[0] + recvcount * gap_ranks;
        }
        MPI_Aint dis_send_bytes[2] = { dis_send_elems[0] * extent, dis_send_elems[1] * extent };

        MPI_Aint bl_recv[2], dis_recv_elems[2];
        bl_recv[0] = recvcount * (MPI_Aint)((dst_tree_root < P) ? dst_tree_root : P);
        bl_recv[1] = recvcount * (MPI_Aint)(P - (dst_tree_root + mask));
        if (bl_recv[1] < 0) bl_recv[1] = 0;

        dis_recv_elems[0] = 0;
        {
            MPI_Aint gap_ranks = (MPI_Aint)((((dst_tree_root + mask) < P) ? (dst_tree_root + mask) : P) - dst_tree_root);
            dis_recv_elems[1]  = bl_recv[0] + recvcount * gap_ranks;
        }
        MPI_Aint dis_recv_bytes[2] = { dis_recv_elems[0] * extent, dis_recv_elems[1] * extent };

        // hindexed requires int blocklengths; assume your test sizes fit
        int bl_send_i[2] = { (int)bl_send[0], (int)bl_send[1] };
        int bl_recv_i[2] = { (int)bl_recv[0], (int)bl_recv[1] };

        TypeGuard sendtype, recvtype;
        int e = MPI_Type_create_hindexed(2, bl_send_i, dis_send_bytes, datatype, &sendtype.t);
        if (e != MPI_SUCCESS) return e;
        e = MPI_Type_commit(&sendtype.t); if (e != MPI_SUCCESS) return e;

        e = MPI_Type_create_hindexed(2, bl_recv_i, dis_recv_bytes, datatype, &recvtype.t);
        if (e != MPI_SUCCESS) return e;
        e = MPI_Type_commit(&recvtype.t); if (e != MPI_SUCCESS) return e;

        int received = 0;

        if (dst < P) {
            e = MPI_Sendrecv(tmp_results, 1, sendtype.t, dst, 123,
                             tmp_recvbuf, 1, recvtype.t, dst, 123,
                             comm, MPI_STATUS_IGNORE);
            if (e != MPI_SUCCESS) return e;
            received = 1;
        }

        // --- Handle non power-of-two tails (forwarding) ---
        if (dst_tree_root + mask > P) {
            int nprocs_completed = P - my_tree_root - mask;

            int j = mask, k = 0; while (j) { j >>= 1; ++k; } --k;
            int tmp_mask = mask >> 1;

            while (tmp_mask) {
                int sd = rank ^ tmp_mask;
                int tree_root = (rank >> k) << k;

                if ((sd > rank) && (rank < tree_root + nprocs_completed) &&
                    (sd   >= tree_root + nprocs_completed)) {
                    e = MPI_Send(tmp_recvbuf, 1, recvtype.t, sd, 456, comm);
                    if (e != MPI_SUCCESS) return e;
                } else if ((sd < rank) &&
                           (sd   < tree_root + nprocs_completed) &&
                           (rank >= tree_root + nprocs_completed)) {
                    e = MPI_Recv(tmp_recvbuf, 1, recvtype.t, sd, 456, comm, MPI_STATUS_IGNORE);
                    if (e != MPI_SUCCESS) return e;
                    received = 1;
                }
                tmp_mask >>= 1; --k;
            }
        }

        // --- Reduce received data into tmp_results (two blocks) ---
        if (received) {
            const bool reduce_recv_into_res = is_commutative || (dst_tree_root < my_tree_root);

            // block 0
            if (bl_recv[0] > 0) {
                void *dst0 = tmp_results;
                void *src0 = tmp_recvbuf;
                if (reduce_recv_into_res)
                    e = MPI_Reduce_local(src0, dst0, bl_recv_i[0], datatype, op);
                else
                    e = MPI_Reduce_local(dst0, src0, bl_recv_i[0], datatype, op);
                if (e != MPI_SUCCESS) return e;
            }
            // block 1
            if (bl_recv[1] > 0) {
                char *dst1 = ((char*)tmp_results) + dis_recv_bytes[1];
                char *src1 = ((char*)tmp_recvbuf) + dis_recv_bytes[1];
                if (reduce_recv_into_res)
                    e = MPI_Reduce_local(src1, dst1, bl_recv_i[1], datatype, op);
                else
                    e = MPI_Reduce_local(dst1, src1, bl_recv_i[1], datatype, op);
                if (e != MPI_SUCCESS) return e;
            }

            // if we reduced into tmp_recvbuf (non-commutative “else”), copy that region back
            if (!is_commutative && !(dst_tree_root < my_tree_root)) {
                e = MPI_Sendrecv(tmp_recvbuf, 1, recvtype.t, rank, 789,
                                 tmp_results,  1, recvtype.t, rank, 789,
                                 comm, MPI_STATUS_IGNORE);
                if (e != MPI_SUCCESS) return e;
            }
        }

        mask <<= 1;
        ++stage;
    }

    // Copy this rank's final block to recvbuf
    if (recvcount > 0) {
        char *src = ((char*)tmp_results) + (disps[(size_t)rank] * extent);
        std::memcpy(recvbuf, src, (size_t)(recvcount * extent));
    }

    return MPI_SUCCESS;
}

// ---------------------- TEST DRIVER ----------------------
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

            int rc = MPICH_reduce_scatter_rec_halving(send.data(), got.data(), recvcount, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
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

            int rc = MPICH_reduce_scatter_rec_halving(MPI_IN_PLACE, inout.data(), recvcount, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
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

            int rc = MPICH_reduce_scatter_rec_halving(send.data(), got.data(), recvcount, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
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

            int rc = MPICH_reduce_scatter_rec_halving(MPI_IN_PLACE, inout.data(), recvcount, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
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