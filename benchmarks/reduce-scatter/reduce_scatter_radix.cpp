#include "reduce_scatter.hpp"

//#define DEBUG_MODE



static int MPICH_Recexchalgo_get_neighbors(int rank, int nranks, int *k_, int *step1_sendto, int **step1_recvfrom_, int *step1_nrecvs, int ***step2_nbrs_, int *step2_nphases, int *p_of_k_, int *T_){
    int mpi_errno = MPI_SUCCESS;
    int i, j, k;
    int p_of_k = 1, log_p_of_k = 0, rem, T, newrank;
    int **step2_nbrs;
    int *step1_recvfrom;

    k = *k_;
    if (nranks < k)     /* If size of the communicator is less than k, reduce the value of k */
        k = (nranks > 2) ? nranks : 2;
    *k_ = k;

    while (p_of_k <= nranks) {
        p_of_k *= k;
        log_p_of_k++;
    }
    p_of_k /= k;
    log_p_of_k--;

    step1_recvfrom = *step1_recvfrom_ = (int *) malloc(sizeof(int) * (k - 1));
    step2_nbrs = *step2_nbrs_ = (int **) malloc(sizeof(int *) * log_p_of_k);



    if(step1_recvfrom == NULL || step2_nbrs == NULL) {
        std::cout<<"Error in malloc"<<std::endl;
        return MPI_ERR_OTHER;
    }

    for (i = 0; i < log_p_of_k; i++) {
        (*step2_nbrs_)[i] = (int *) malloc(sizeof(int) * (k - 1));
    }


    *step2_nphases = log_p_of_k;

    rem = nranks - p_of_k;

    T = (rem * k) / (k - 1);
    *T_ = T;
    *p_of_k_ = p_of_k;

    *step1_nrecvs = 0;
    *step1_sendto = -1;

    if (rank < T) {
        if (rank % k != (k - 1)) {      /* I am a non-participating rank */
            *step1_sendto = rank + (k - 1 - rank % k);  /* partipating rank to send the data to */
            /* if the corresponding participating rank is not in T,
             * then send to the Tth rank to preserve non-commutativity */
            if (*step1_sendto > T - 1)
                *step1_sendto = T;
            newrank = -1;       /* tag this rank as non-participating */
        } else {        /* participating rank */
            for (i = 0; i < k - 1; i++) {
                step1_recvfrom[i] = rank - i - 1;
            }
            *step1_nrecvs = k - 1;
            newrank = rank / k; /* this is the new rank amongst the set of participating ranks */
        }
    } else {    /* rank >= T */
        newrank = rank - rem;

        if (rank == T && (T - 1) % k != k - 1 && T >= 1) {
            int nsenders = (T - 1) % k + 1;     /* number of ranks sending their data to me in Step 1 */

            for (j = nsenders - 1; j >= 0; j--) {
                step1_recvfrom[nsenders - 1 - j] = T - nsenders + j;
            }
            *step1_nrecvs = nsenders;
        }
    }

    if (*step1_sendto == -1) {  /* calculate step2_nbrs only for participating ranks */
        int *digit = (int *) malloc(sizeof(int) * log_p_of_k);
        if(digit == NULL) {
            std::cout<<"Error in malloc"<<std::endl;
            return MPI_ERR_OTHER;
        }
        int temprank = newrank;
        int mask = 0x1;
        int phase = 0, cbit, cnt, nbr, power;

        /* calculate the digits in base k representation of newrank */
        for (i = 0; i < log_p_of_k; i++)
            digit[i] = 0;

        int remainder, i_digit = 0;
        while (temprank != 0) {
            remainder = temprank % k;
            temprank = temprank / k;
            digit[i_digit] = remainder;
            i_digit++;
        }

        while (mask < p_of_k) {
            cbit = digit[phase];        /* phase_th digit changes in this phase, obtain its original value */
            cnt = 0;
            for (i = 0; i < k; i++) {   /* there are k-1 neighbors */
                if (i != cbit) {        /* do not generate yourself as your nieighbor */
                    digit[phase] = i;   /* this gets us the base k representation of the neighbor */

                    /* calculate the base 10 value of the neighbor rank */
                    nbr = 0;
                    power = 1;
                    for (j = 0; j < log_p_of_k; j++) {
                        nbr += digit[j] * power;
                        power *= k;
                    }

                    /* calculate its real rank and store it */
                    step2_nbrs[phase][cnt] =
                        (nbr < rem / (k - 1)) ? (nbr * k) + (k - 1) : nbr + rem;
                    cnt++;
                }
            }

            digit[phase] = cbit;        /* reset the digit to original value */
            phase++;
            mask *= k;
        }

        free(digit);
    }

    return mpi_errno;

}

static int MPICH_Recexchalgo_origrank_to_step2rank(int rank, int rem, int T, int k)
{
    int step2rank;


    step2rank = (rank < T) ? rank / k : rank - rem;


    return step2rank;
}


static int MPICH_Recexchalgo_step2rank_to_origrank(int rank, int rem, int T, int k)
{
    int orig_rank;


    orig_rank = (rank < rem / (k - 1)) ? (rank * k) + (k - 1) : rank + rem;


    return orig_rank;
}


static int MPICH_Recexchalgo_get_count_and_offset(int rank, int phase, int k, int nranks, int *count,
                                          int *offset)
{
    int mpi_errno = MPI_SUCCESS;
    int step2rank, min, max, orig_max, orig_min;
    int k_power_phase = 1;
    int p_of_k = 1, rem, T;


    /* p_of_k is the largest power of k that is less than nranks */
    while (p_of_k <= nranks) {
        p_of_k *= k;
    }
    p_of_k /= k;

    rem = nranks - p_of_k;
    T = (rem * k) / (k - 1);

    /* k_power_phase is k^phase */
    while (phase > 0) {
        k_power_phase *= k;
        phase--;
    }
    /* Calculate rank in step2 */
    step2rank = MPICH_Recexchalgo_origrank_to_step2rank(rank, rem, T, k);
    /* Calculate min and max ranks of the range of ranks that 'rank'
     * represents in phase 'phase' */
    min = ((step2rank / k_power_phase) * k_power_phase) - 1;
    max = min + k_power_phase;
    /* convert (min,max] to their original ranks */
    orig_min = (min >= 0) ? MPICH_Recexchalgo_step2rank_to_origrank(min, rem, T, k) : min;
    orig_max = MPICH_Recexchalgo_step2rank_to_origrank(max, rem, T, k);
    *count = orig_max - orig_min;
    *offset = orig_min + 1;


    return mpi_errno;
}


// ----------------- implementation -----------------

// Stand-alone, overlapped radix-k Reduce_scatter_block (recexch-style).
// Assumes 'op' is commutative (like MPICH's intra_recexch path).
int MPICH_reduce_scatter_radix(const void *sendbuf, void *recvbuf,
                               MPI_Aint recvcount, MPI_Datatype datatype,
                               MPI_Op op, MPI_Comm comm, int k)
{

    int mpi_errno = MPI_SUCCESS;
    int is_inplace;
    int extent;
    int step1_sendto = -1, step2_nphases = 0, step1_nrecvs = 0;
    int in_step2;
    int *step1_recvfrom = NULL;
    int **step2_nbrs = NULL;
    int nranks, rank, p_of_k, T, dst;
    int i, phase, offset;
    void *tmp_recvbuf = NULL, *tmp_results = NULL;
    int num_reqs = 0;
   

    is_inplace = (sendbuf == MPI_IN_PLACE);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    MPI_Type_size(datatype, &extent);

    int total_count = recvcount * nranks;

    MPICH_Recexchalgo_get_neighbors(rank, nranks, &k, &step1_sendto, &step1_recvfrom, &step1_nrecvs, &step2_nbrs, &step2_nphases, &p_of_k, &T);
    MPI_Request* reqs = (MPI_Request*)malloc(sizeof(MPI_Request)* (step1_nrecvs + 1));

    in_step2 = (step1_sendto == -1) ? 1 : 0;

    tmp_results = malloc(extent * total_count);
    tmp_recvbuf = malloc(extent * total_count);

    if(in_step2){
        if(!is_inplace){
            memcpy(tmp_results, sendbuf, extent * total_count);
        }else{
            memcpy(tmp_results, recvbuf, extent * total_count);
        }
    }


    if(!in_step2){
        void *buf_to_send;
        if (is_inplace)
            buf_to_send = recvbuf;
        else
            buf_to_send = (void *) sendbuf;
        mpi_errno = MPI_Send(buf_to_send, (int)total_count, datatype, step1_sendto, 0, comm);
    }else{
        for(i = 0; i < step1_nrecvs; i++){
            num_reqs = 0;
            mpi_errno = MPI_Irecv((char*)tmp_recvbuf, total_count, datatype, step1_recvfrom[i], 0, comm , &reqs[num_reqs++]);
            
            if(mpi_errno != MPI_SUCCESS) {
                std::cout<<"Error in MPI_Recv"<<std::endl;
                goto fn_fail;
            }
            
            mpi_errno = MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
            
            mpi_errno = MPI_Reduce_local(tmp_recvbuf, tmp_results, total_count, datatype, op);
           
            if(mpi_errno != MPI_SUCCESS) {
                std::cout<<"Error in MPI_Reduce_local"<<std::endl;
                goto fn_fail;
            }
        }
    }




    /* Step 2 */
    for (phase = step2_nphases - 1; phase >= 0 && step1_sendto == -1; phase--) {
        for (i = 0; i < k - 1; i++) {
            dst = step2_nbrs[phase][i];
            int send_cnt = 0, recv_cnt = 0;
            num_reqs = 0;
            /* Both send and recv have similar dependencies */
            MPICH_Recexchalgo_get_count_and_offset(dst, phase, k, nranks, &send_cnt, &offset);

            int send_offset = offset * extent * recvcount;

            mpi_errno = MPI_Isend((char*) tmp_results + send_offset, send_cnt * recvcount, datatype, dst, 0, comm, &reqs[num_reqs++]);

            if(mpi_errno != MPI_SUCCESS) {
                std::cout<<"Error in MPI_Send"<<std::endl;
                goto fn_fail;
            }

            MPICH_Recexchalgo_get_count_and_offset(rank, phase, k, nranks, &recv_cnt, &offset);

            int recv_offset = offset * extent * recvcount;
            mpi_errno = MPI_Irecv(tmp_recvbuf, recv_cnt * recvcount, datatype, dst, 0, comm, &reqs[num_reqs++]);

            if(mpi_errno != MPI_SUCCESS) {
                std::cout<<"Error in MPI_Recv"<<std::endl;
                goto fn_fail;
            }



            mpi_errno = MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);

            mpi_errno = MPI_Reduce_local(tmp_recvbuf, (char*)tmp_results + recv_offset, recv_cnt * recvcount, datatype, op);

            if(mpi_errno != MPI_SUCCESS) {
                std::cout<<"Error in MPI_Reduce_local"<<std::endl;
                goto fn_fail;
            }

        }
    }


    if(in_step2){
        memcpy(recvbuf, (char*)tmp_results + rank * recvcount * extent, recvcount * extent);
    }

    num_reqs = 0;





    if(step1_sendto != -1){
        mpi_errno = MPI_Irecv(recvbuf, recvcount, datatype, step1_sendto, 0, comm, &reqs[num_reqs++]);
        if(mpi_errno != MPI_SUCCESS) {
            std::cout<<"Error in MPI_Recv"<<std::endl;
            goto fn_fail;
        }
    }


    for(i = 0; i < step1_nrecvs; i++) {
        mpi_errno = MPI_Isend((char*) tmp_results + recvcount * step1_recvfrom[i] * extent, recvcount, datatype, step1_recvfrom[i], 0, comm, &reqs[num_reqs++]);
    }




    mpi_errno = MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);




fn_exit:

    if(step2_nbrs != NULL){
        for(i = 0; i < step2_nphases && step2_nbrs[i] != NULL; i++)
            free(step2_nbrs[i]);
        
        free(step2_nbrs);
    }

    if(step1_recvfrom != NULL)
         free(step1_recvfrom);

    free(reqs);
    free(tmp_results);
    free(tmp_recvbuf);
        
        
    return mpi_errno;

fn_fail:
    goto fn_exit;



    
}
// ----------------- Tiny test driver (compares vs MPI_Reduce_scatter_block) -----------------
#ifdef DEBUG_MODE
static int almost_equal(double a, double b) {
    double diff = std::fabs(a - b);
    double tol  = 1e-10 * std::max({1.0, std::fabs(a), std::fabs(b)});
    return diff <= tol;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank=0, P=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int k = (argc >= 2) ? std::atoi(argv[1]) : 3; // default radix=3
    if (k < 2) k = 2;

    const int counts[] = {1, 3, 16, 31,64, 127, 256};
    int failures = 0;

    for (int recvcount : counts) {
        const int total = recvcount * P;

        // INT / SUM
        {
            std::vector<int> send(total), ref(recvcount), got(recvcount);
            for (int j = 0; j < total; ++j) send[j] = rank * 1000 + j;

            MPI_Reduce_scatter_block(send.data(), ref.data(), recvcount, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            int rc = MPICH_reduce_scatter_radix(send.data(), got.data(), recvcount, MPI_INT, MPI_SUM, MPI_COMM_WORLD, k);
            if (rc != MPI_SUCCESS) { if (rank==0) std::fprintf(stderr, "[k=%d INT] err %d\n", k, rc); ++failures; }
            else {
                for (int i = 0; i < recvcount; ++i)
                    if (got[i] != ref[i]) { std::fprintf(stderr, "Rank %d: [k=%d INT] mismatch @%d\n", rank, k, i); ++failures; break; }
            }
        }

        // In-place INT / SUM
        {
            std::vector<int> inout(total), ref(recvcount);
            for (int j = 0; j < total; ++j) inout[j] = rank * 1000 + j;
            MPI_Reduce_scatter_block(inout.data(), ref.data(), recvcount, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            int rc = MPICH_reduce_scatter_radix(MPI_IN_PLACE, inout.data(), recvcount, MPI_INT, MPI_SUM, MPI_COMM_WORLD, k);
            if (rc != MPI_SUCCESS) { if (rank==0) std::fprintf(stderr, "[k=%d INT in-place] err %d\n", k, rc); ++failures; }
            else {
                for (int i = 0; i < recvcount; ++i)
                    if (inout[i] != ref[i]) { std::fprintf(stderr, "Rank %d: [k=%d INT in-place] mismatch @%d\n", rank, k, i); ++failures; break; }
            }
        }

        // DOUBLE / MAX
        {
            std::vector<double> send(total), ref(recvcount), got(recvcount);
            for (int j = 0; j < total; ++j) send[j] = (double)rank + 0.001 * j;

            MPI_Reduce_scatter_block(send.data(), ref.data(), recvcount, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

            int rc = MPICH_reduce_scatter_radix(send.data(), got.data(), recvcount, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, k);
            if (rc != MPI_SUCCESS) { if (rank==0) std::fprintf(stderr, "[k=%d DBL] err %d\n", k, rc); ++failures; }
            else {
                for (int i = 0; i < recvcount; ++i)
                    if (!almost_equal(got[i], ref[i])) { std::fprintf(stderr, "Rank %d: [k=%d DBL] mismatch @%d\n", rank, k, i); ++failures; break; }
            }
        }

        if (rank == 0) std::printf("[k=%d count=%d] tests done.\n", k, recvcount);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        if (failures == 0) std::printf("✅ All tests passed (k-nomial, nonblocking overlap).\n");
        else               std::printf("❌ %d failure(s) detected.\n", failures);
    }

    MPI_Finalize();
    return failures ? 1 : 0;
}

#endif