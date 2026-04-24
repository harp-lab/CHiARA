#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cstring>   // for std::memcpy / memcpy
#include <climits>   // for INT_MAX
// #include <cmath>
// #define DEBUG_MODE  

static inline int MPICH_Recexchalgo_get_neighbors(int rank, int nranks, int *k_, int *step1_sendto, int **step1_recvfrom_, int *step1_nrecvs, int ***step2_nbrs_, int *step2_nphases, int *p_of_k_, int *T_){
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

static inline int MPICH_Recexchalgo_origrank_to_step2rank(int rank, int rem, int T, int k)
{
    int step2rank;


    step2rank = (rank < T) ? rank / k : rank - rem;


    return step2rank;
}


static inline int MPICH_Recexchalgo_step2rank_to_origrank(int rank, int rem, int T, int k)
{
    int orig_rank;


    orig_rank = (rank < rem / (k - 1)) ? (rank * k) + (k - 1) : rank + rem;


    return orig_rank;
}

static inline void Recexchalgo_get_all_count_and_offset(int nranks, int max_phases, int k, int *count, int *offset){
    int step2rank, min, max, orig_max, orig_min;
    int k_power_phase = 1;
    int p_of_k = 1, rem, T;
    
    
    while (p_of_k <= nranks) {
        p_of_k *= k;
    }
    p_of_k /= k; //THIS IS ALWAYS THE SAME, DOES NOT MAKE SENSE TO TALCULATE IT EVERY TIME

    rem = nranks - p_of_k;
    T = (rem * k) / (k - 1); //ALWAYS THE SAME

    for(int phase = 0; phase < max_phases; phase++){
        
        for(int rank = 0; rank < nranks; rank++){
            
            step2rank = MPICH_Recexchalgo_origrank_to_step2rank(rank, rem, T, k);
            /* Calculate min and max ranks of the range of ranks that 'rank'
            * represents in phase 'phase' */
            min = ((step2rank / k_power_phase) * k_power_phase) - 1;
            max = min + k_power_phase;
            /* convert (min,max] to their original ranks */
            orig_min = (min >= 0) ? MPICH_Recexchalgo_step2rank_to_origrank(min, rem, T, k) : min;
            orig_max = MPICH_Recexchalgo_step2rank_to_origrank(max, rem, T, k);

            count[phase * nranks + rank] = orig_max - orig_min;
            offset[phase * nranks + rank] = orig_min + 1;
        }
        k_power_phase *= k; //EVERY PHASE ONE K LESS
            
    
    }

}



int all_reduce_radix_batch(char *sendbuf, char *recvbuf, int aCount,
                                MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                                int k, int b){

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
    int stage;
    int nIters,delta;
    int src;
    bool rootCond;


    int node_id;
    int nnodes;
    int node_rank;
    int nstages;

#ifdef DEBUG_MODE
    double t0, t1;
    MPI_Barrier(comm);
    t0 = MPI_Wtime();
#endif

    is_inplace = (sendbuf == MPI_IN_PLACE);

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    MPI_Type_size(datatype, &extent);
    int recvcount = aCount / nranks;

    node_id = rank / b;
    nnodes = nranks / b;
    node_rank = rank % b;
    nstages = nnodes / b;

    int root_node, root_rank, src_rank;


    int intra_recvcount = recvcount * b;


    int intra_total_count = intra_recvcount * b;

    int total_count = recvcount * nranks;

    int total_bytes = total_count * extent;

    int nu_count = nnodes%b;

    int max_sendcount = 0;
    int max_recvcount = 0;

    int idst;

    const int intra_recvbytes = intra_recvcount * extent;

    const int max_offset = nu_count * intra_recvbytes;

    int send_cnt, recv_cnt, send_offset, recv_offset;




    int nphases = 0, tmpb = b - 1;
    
    
    while (tmpb > 0) { ++nphases; tmpb /= k; }
   

    int j;


    MPICH_Recexchalgo_get_neighbors(node_rank, b, &k, &step1_sendto, &step1_recvfrom, &step1_nrecvs, &step2_nbrs, &step2_nphases, &p_of_k, &T);

    int *count = (int *)malloc(sizeof(int) * step2_nphases * b);
    int *offsets = (int *)malloc(sizeof(int) * step2_nphases * b);

    Recexchalgo_get_all_count_and_offset(b, step2_nphases, k, count, offsets);

    
    MPI_Request* reqs = (MPI_Request*)malloc(sizeof(MPI_Request)* (nnodes + 2*k));

    in_step2 = (step1_sendto == -1) ? 1 : 0;


    tmp_results = malloc(total_bytes);//extend to total_count
    tmp_recvbuf = malloc(total_bytes);

    char* phase_buf = (char*)tmp_recvbuf;//(char*) malloc(extent*total_recv_elems);

    void *tmp_results_start = tmp_results;
    void *tmp_recvbuf_start = tmp_recvbuf;



    if(in_step2){
        if(!is_inplace){
            memcpy(tmp_results, sendbuf, total_bytes); //here should be total_count
        }else{
            memcpy(tmp_results, recvbuf, total_bytes); //here should be total_count
        }
    }


    if(!in_step2){
        void *buf_to_send;

        if (is_inplace)
            buf_to_send = recvbuf;
        else
            buf_to_send = (void *) sendbuf;
        
            MPI_Isend(buf_to_send, (int)total_count, datatype, step1_sendto + b*node_id, 0, comm, &reqs[num_reqs++]);
    }else{
        for(i = 0; i < step1_nrecvs; i++){
            num_reqs = 0;

            MPI_Irecv((char*)tmp_recvbuf, total_count, datatype, step1_recvfrom[i] + b*node_id, 0, comm , &reqs[num_reqs++]);//here should be total_count
            
            MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
            
            MPI_Reduce_local(tmp_recvbuf, tmp_results, total_count, datatype, op);//here should be total_count
           
        }
    }


    //Here we iterate for stages
    for(stage = 0; stage < nstages; stage++){


        for (phase = step2_nphases - 1; phase >= 0 && step1_sendto == -1; phase--) {
            for (i = 0; i < k - 1; i++) {
                dst = step2_nbrs[phase][i];
                send_cnt = 0, recv_cnt = 0;
                num_reqs = 0;

                send_cnt = count[phase * b + dst];
                offset = offsets[phase * b + dst];

                send_offset = offset * intra_recvbytes;

                MPI_Isend((char*) tmp_results + send_offset, send_cnt * intra_recvcount, datatype, dst + b*node_id, 0, comm, &reqs[num_reqs++]);


                recv_cnt = count[phase * b + node_rank];
                offset = offsets[phase * b + node_rank];

                recv_offset = offset * intra_recvbytes;
                MPI_Irecv(tmp_recvbuf, recv_cnt * intra_recvcount, datatype, dst + b*node_id, 0, comm, &reqs[num_reqs++]);

                MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);

                MPI_Reduce_local(tmp_recvbuf, (char*)tmp_results + recv_offset, recv_cnt * intra_recvcount, datatype, op);


            }
        }

        num_reqs = 0;

        if(in_step2){
            memcpy(phase_buf, (char*)tmp_results + node_rank * intra_recvbytes, intra_recvbytes);
        }

        

        if(step1_sendto != -1){
            MPI_Irecv(phase_buf, intra_recvcount, datatype, step1_sendto + b*node_id, 0, comm, &reqs[num_reqs++]);
        }


        for(i = 0; i < step1_nrecvs; i++) {
            MPI_Isend((char*) tmp_results + intra_recvcount * step1_recvfrom[i] * extent, intra_recvcount, datatype, step1_recvfrom[i] + b*node_id, 0, comm, &reqs[num_reqs++]);
        }




        

        tmp_results = (char*)tmp_results + (intra_total_count * extent);
        tmp_recvbuf = (char*)tmp_recvbuf + (intra_total_count * extent);

        phase_buf = (char*)phase_buf + (intra_recvbytes);

        MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
        

    }



    if(nu_count != 0){


        
        for (phase = step2_nphases - 1; phase >= 0 && step1_sendto == -1; phase--) {
            num_reqs = 0;
            for (i = 0; i < k - 1; i++) {
                idst = step2_nbrs[phase][i];

                dst = idst + b*node_id;

                send_cnt = 0, recv_cnt = 0;
                
                send_cnt = count[phase * b + idst];
                offset = offsets[phase * b + idst];

                send_offset = offset * intra_recvbytes;

                max_sendcount = std::min(send_cnt, (nu_count - offset));

                if((send_offset < max_offset) && (max_sendcount > 0)){
                    MPI_Isend((char*) tmp_results + send_offset, max_sendcount * intra_recvcount, datatype, dst, 0, comm, &reqs[num_reqs++]);
                }




                //MPICH_Recexchalgo_get_count_and_offset(node_rank, phase, k, b, &recv_cnt, &offset);
                recv_cnt = count[phase * b + node_rank];
                offset = offsets[phase * b + node_rank];

                recv_offset = offset * intra_recvbytes;

                max_recvcount = std::min(recv_cnt, (nu_count - offset));


                if((recv_offset < max_offset) && (max_recvcount > 0)){

                    MPI_Irecv(tmp_recvbuf, max_recvcount * intra_recvcount, datatype, dst, 0, comm, &reqs[num_reqs++]);

                    MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
                    
                    MPI_Reduce_local(tmp_recvbuf, (char*)tmp_results + recv_offset, max_recvcount * intra_recvcount, datatype, op);
                    
                    num_reqs = 0;

                }



            }
            
            MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);
        }

         if(in_step2 && node_rank < nu_count){
            memcpy(phase_buf, (char*)tmp_results + node_rank * intra_recvbytes, intra_recvbytes);
        }

        num_reqs = 0;

        if(step1_sendto != -1 && node_rank < nu_count){
            MPI_Irecv(phase_buf, intra_recvcount, datatype, step1_sendto + b*node_id, 0, comm, &reqs[num_reqs++]);
        }


        for(i = 0; i < step1_nrecvs; i++) {
            if(intra_recvbytes * step1_recvfrom[i] < max_offset)
                MPI_Isend((char*) tmp_results + intra_recvbytes * step1_recvfrom[i], intra_recvcount, datatype, step1_recvfrom[i] + b*node_id, 0, comm, &reqs[num_reqs++]);
        }

         MPI_Waitall(num_reqs, reqs, MPI_STATUSES_IGNORE);


    }

    #ifdef DEBUG_MODE


    t1 = MPI_Wtime();
    MPI_Barrier(comm);
    if(rank == 0){
        printf("Red-Scatter Phase 1 time: %f\n", t1 - t0);
    }

    #endif



    tmp_results = tmp_results_start;
    tmp_recvbuf = tmp_recvbuf_start;

    

    nIters = (nu_count == 0) ? (nstages) : 1 + (nstages); // number of rotating roots (i)


    for (i = 0; i < nIters; ++i) {
        root_node = i * b + node_rank;        // 0..nnodes-1 (lane root for this i)

        if (root_node >= nnodes) break;        // guard (shouldn’t happen if nnodes % b == 0)

        root_rank = root_node * b + node_rank;

        if (node_id == root_node) {

            tmp_recvbuf = (char*)tmp_recvbuf + i * intra_recvbytes;
            num_reqs = 0;
            for (stage = 0, src_rank = node_rank; stage < nnodes; ++stage, src_rank += b) {

                if (stage == node_id) continue;


                MPI_Irecv((char*)tmp_results + intra_recvbytes * stage, intra_recvcount, datatype, src_rank, i, comm, &reqs[num_reqs++]);
  
            }

            int jdx = 0;

            for(stage = 0 ; stage < nnodes; stage++){

                if (stage == node_id) continue;

                MPI_Wait(&reqs[jdx++], MPI_STATUS_IGNORE);

                MPI_Reduce_local((char*)tmp_results + intra_recvbytes * stage, tmp_recvbuf, intra_recvcount, datatype, op);
            }

        } else {
            // Non-root sends its lane-chunk to the root of this iteration
            MPI_Isend((char*)tmp_recvbuf_start + i * intra_recvbytes, intra_recvcount, datatype, root_rank, i, comm, &reqs[0]);
            MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
        }

    
    }


    #ifdef DEBUG_MODE
    t0 = MPI_Wtime();
    MPI_Barrier(comm);
    if(rank == 0){
        printf("Red-Scatter Phase 2 time: %f\n", t0 - t1);
    }
    #endif

    //ALLGATHER PHASE

     for(i = 0; i<nnodes; i+=b){
        num_reqs = 0;
        rootCond = ((node_id - i) == node_rank);
        if(!rootCond && (i+node_rank < nnodes)){
            MPI_Irecv((char*)tmp_results+intra_recvbytes*(i/b), intra_recvcount, datatype, (i+node_rank)*b + node_rank, i, comm, &reqs[num_reqs++]);
        }else if(rootCond){

            memcpy((char*)tmp_results+intra_recvbytes*(i/b), tmp_recvbuf, intra_recvbytes);

            for(j=0; j<nnodes; j++){
                if(j==node_id )
                    continue;
                dst = j*b + node_rank;
                MPI_Isend(tmp_recvbuf, intra_recvcount, datatype, dst, i, comm, &reqs[num_reqs++]);
            }
        }
        MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);//i think i can move this wait outside the for loop
    }


    #ifdef DEBUG_MODE
    t1 = MPI_Wtime();
    MPI_Barrier(comm);
    if(rank == 0){
        printf("Allgather Phase 1 time: %f\n", t1 - t0);
    }
    #endif




    if(node_rank == 0) {
        tmp_recvbuf = recvbuf;
    } else {
        tmp_recvbuf = tmp_recvbuf_start;
    }

    delta = 1;

    for(stage = 0; stage < nstages; stage++){

        memcpy(tmp_recvbuf, tmp_results, intra_recvcount * extent);
        
        for(i = 0; i < nphases; i++) {
            num_reqs = 0;
            for(j = 1; j < k; j++) {
                if(delta * j >= b)  // Fixed 'size' to 'nranks'
                    break;
                    
                dst = (b + (node_rank - delta * j)) % b + node_id*b;
                src = (node_rank + delta * j) % b + node_id*b;  // Fixed 'size' to 'nranks' and 'k' to 'j'

                int tmp_count;
                if ((i == nphases - 1) && (!(p_of_k==b))) {
                    tmp_count = intra_recvcount * delta;
                    int left_count = intra_recvcount * (b - delta * j);
                    if (j == k - 1) {
                        tmp_count = left_count;
                    } else {
                        tmp_count = std::min(tmp_count, left_count);
                    }
                } else {
                    tmp_count = intra_recvcount * delta;
                }

                MPI_Irecv((char*)tmp_recvbuf + j * intra_recvcount * delta * extent, 
                            tmp_count, datatype, src, 1, comm, &reqs[num_reqs++]);

                MPI_Isend(tmp_recvbuf, tmp_count, datatype, 
                            dst, 1, comm, &reqs[num_reqs++]);
            }
            
            MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
            delta *= k;
        }

        if(node_rank != 0) {
            // Rearrange data in the correct order
            memcpy(recvbuf, 
                    (char*)tmp_recvbuf + (b - node_rank) * intra_recvcount * extent, 
                    node_rank * intra_recvcount * extent);

            memcpy(recvbuf + node_rank * intra_recvcount * extent, 
                    tmp_recvbuf, 
                    (b - node_rank) * intra_recvcount * extent);
                    

        }

        delta = 1;

        tmp_recvbuf = (char*)tmp_recvbuf + b * intra_recvcount * extent;
        recvbuf = (char*)recvbuf + b*intra_recvcount*extent;
        tmp_results = (char*)tmp_results + intra_recvcount*extent;
    }


    if(nu_count != 0) {

        if(node_rank < nu_count) {
            memcpy(tmp_recvbuf, tmp_results, intra_recvcount * extent);
        } else {
            tmp_recvbuf = recvbuf;
        }

        int active[b];
        int send_sizes[nphases+1][b];
        memset(send_sizes, 0, (nphases+1)*b*sizeof(int));
        //sizes array

        for(i =0; i< b; i++){
            active[i] = i < nu_count ? 0 : -1;
            send_sizes[0][i] = i < nu_count ? intra_recvcount : 0;

        }



        int isrc;
        int received;

        for(i = 0; i < nphases; i++) {
            num_reqs = 0;

            received = send_sizes[i][node_rank];
            
            for(j = 1; j < k; j++) {
                if(delta * j >= b){ 

                    for(int l = 0; l < b; l++){
                        if(active[l] == i){
                                active[l] = i+1;

                        }
                        send_sizes[i+1][l] += send_sizes[i][l];
                    }


                    break;

                }  

                

                

                isrc = (node_rank + delta * j) % b;
                idst = (b + (node_rank - delta * j)) % b;
                    
                dst = idst + node_id*b;
                src = isrc + node_id*b;  // Fixed 'size' to 'nranks' and 'k' to 'j'

                
                int ssssize;
                ssssize = std::min(send_sizes[i][isrc], (nu_count) * intra_recvcount - received);
                if(active[isrc] == i && ssssize > 0){
                    MPI_Irecv((char*)tmp_recvbuf + received * extent, 
                            ssssize, datatype, src, 1, comm, &reqs[num_reqs++]);
                    
                    received += ssssize;
                }
                ssssize = std::min(send_sizes[i][node_rank], (nu_count) * intra_recvcount - (send_sizes[i][idst] + send_sizes[i+1][idst]));

                if(active[node_rank] == i && ssssize > 0){
                    MPI_Isend(tmp_recvbuf, ssssize , datatype, 
                            dst, 1, comm, &reqs[num_reqs++]);
                            
                }

                for(int l = 0; l < b; l++){
                    if(active[l] == i){
                        active[( b+ (l - delta * j)) % b] = active[(b+ (l - delta * j)) % b] != i ? i+1 : i;

                        
                        send_sizes[i+1][(b + (l - delta * j)) % b] += send_sizes[i][l];
                        
                        if(j == k-1){
                            active[l] = i+1;
                        }
                    }
                    if(j == k-1){
                        send_sizes[i+1][l] += send_sizes[i][l];
                    }

                }

            }
            MPI_Waitall(num_reqs, reqs, MPI_STATUS_IGNORE);
            delta *= k;
            
        }

        if(node_rank != 0 && node_rank < nu_count) {
            // Rearrange data in the correct order
            memcpy(recvbuf, 
                    (char*)tmp_recvbuf + ((nu_count) - node_rank) * intra_recvcount * extent, 
                    node_rank * intra_recvcount * extent);

            memcpy(recvbuf + node_rank * intra_recvcount * extent, 
                    tmp_recvbuf, 
                    ((nu_count) - node_rank) * intra_recvcount * extent);

        }

    }

    #ifdef DEBUG_MODE
    t0 = MPI_Wtime();
    MPI_Barrier(comm);
    if(rank == 0){
        printf("Allgather Phase 2 time: %f\n", t0 - t1);
    }
    #endif




    for(i = 0; i < step2_nphases && step2_nbrs[i] != NULL; i++)
        free(step2_nbrs[i]);
        
    free(step2_nbrs);
    
    free(count);
    free(offsets);

    free(step1_recvfrom);

    free(reqs);
    free(tmp_results_start);
    free(tmp_recvbuf_start);
        
        
    return mpi_errno;



}


#ifdef DEBUG_MODE


bool check_correctness(const std::vector<int>& buf,
                       const std::vector<int>& ref,
                       double eps = 1e-9) {
    for (size_t i = 0; i < buf.size(); ++i) {
        if (std::fabs(buf[i] - ref[i]) > eps)
            return false;
    }
    return true;
}


template<typename Func>
void run_k2(const std::string& name, int k, int count, Func func,
            MPI_Comm comm, std::ofstream& csv, int rank, int nprocs, int b) {
    std::vector<int> sendbuf(count), refbuf(count), recvbuf(count);
    // Unique initialization
    for (int i = 0; i < count; ++i)
        sendbuf[i] = rank * static_cast<int>(count) + i;

    // Reference result via MPI_Allreduce
    MPI_Allreduce(sendbuf.data(), refbuf.data(), count,
                  MPI_INT, MPI_SUM, comm);

    for (int rep = 0; rep < 2; ++rep) {
        std::fill(recvbuf.begin(), recvbuf.end(), 0.0);
        MPI_Barrier(comm);
        if(rank == 0)
            std::cout<<"Running "<<name<<" with k="<<k<<", b="<<b<<", count="<<count<<", nprocs="<<nprocs<<"\n";
        double t0 = MPI_Wtime();

        int err = func(reinterpret_cast<char*>(sendbuf.data()),
                       reinterpret_cast<char*>(recvbuf.data()),
                       count, MPI_INT, MPI_SUM, comm,
                       k, b);

        MPI_Barrier(comm);
        double t1 = MPI_Wtime();

        bool correct = (err == MPI_SUCCESS)
                       && check_correctness(recvbuf, refbuf);
        if (rank == 0) {
            csv << name << "," << k << ","<<b<<","<<nprocs<<"," << count/nprocs << ","
                << (t1 - t0) << "," << (correct?1:0) << "\n";
            csv.flush();
        }
    }
}


int main(int argc, char** argv) {
       MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    //
    // Parse arguments:  program <n_iter> [--overwrite] [b=<value>] [base=<value>]
    //
    int n_iter;
    bool overwrite = false;

    int b = 16;

    if(b > 32) b = 32;

    if (argc < 2 || argc > 5) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " <n_iter> [--overwrite] [b=<value>] [base=<value>]\n";
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int base = 1;

    n_iter = std::atoi(argv[1]);

    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--overwrite") == 0) {
            overwrite = true;
        } else if (std::strncmp(argv[i], "b=", 2) == 0) {
            b = std::atoi(argv[i] + 2);
        } else if (std::strncmp(argv[i], "base=", 5) == 0) {
            base = std::atoi(argv[i] + 5);
        } else {
            if (rank == 0) {
                std::cerr << "Unknown parameter: " << argv[i] << "\n";
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }



    std::ofstream csv;
    if (rank == 0) {
        // check whether results.csv already exists

        std::string filename = "results" + std::to_string(nprocs / 32) + ".csv";

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

    base *= nprocs;
    for (int i = 0; i < n_iter; ++i) {
        int count = base << i;

        // Algorithms with k + single_phase_recv

        run_k2("all_reduce_radix_batch", 2, count,
                all_reduce_radix_batch,
                MPI_COMM_WORLD, csv, rank, nprocs, b);

        


    }

    if (rank == 0) csv.close();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

#endif