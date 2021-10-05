subroutine mdim_gls(t, y, err, freqs, t_ref, coeffs, A)

    !implicit Real*4 (a-h,o-z)
    Real*4, intent(in) :: t(:), y(:), err(:)
    Real*4, intent(in) :: freqs(:), t_ref
    Real*4, intent(out):: coeffs(2*size(freqs)+1)
    Real*4 :: F(size(t),2*size(freqs)+1), Ft(2*size(freqs)+1, size(t)), b(2*size(freqs)+1), y_err(size(t))
    Real*4 , intent(out):: A(2*size(freqs)+1,2*size(freqs)+1)
    Real*4 ::  pi=acos(-1.0)
    integer :: i, j, k, n_dim, LS_INFO, pivot(2*size(freqs)+1)

    n_dim = size(freqs)
    n_data = size(t)

    do k = 1, 2*n_dim+1
        if (k.EQ.1) then
            do j = 1,n_data
                F(j,k) = 1.0/err(j)
            enddo
        else if ((k.GT.1).AND.(k.LE.(n_dim+1))) then
             do j = 1,n_data
                 F(j,k) = cos(2.0*pi*(t(j)-t_ref)*freqs(k-1))/err(j)
             enddo
        else if ((k.GT.(n_dim+1)).AND.(k.LT.(2*n_dim+2))) then
             do j = 1,n_data
                 F(j,k) = sin(2.0*pi*(t(j)-t_ref)*freqs(k-(n_dim+1)))/err(j)
             enddo
         !else if (k.EQ.(2*n_dim+2)) then
         !    do j = 1,n_data
         !        F(j,k) = (t(j)-t_ref)/err(j)
         !    enddo
        endif
     enddo
    
    do i =1,n_data
        y_err(i) = y(i)/err(i)
    enddo
    !arrange linear system F_t*F*theta = F_t*y_e --> A*theta = b
    Ft = transpose(F)
    A = matmul(Ft,F)
    b = matmul(Ft,y_err)
    !solves a system of linear equations A*X = B 
    call SGESV(2*n_dim+1, 1, A, 2*n_dim+1, pivot, b, 2*n_dim+1, LS_INFO)
    !fill output vector: fitting coeffs.
    do i = 1,2*n_dim+1
        coeffs(i) = b(i)
    enddo

end subroutine mdim_gls

subroutine mdim_gls_multiset(times, rvs, errs, freqs, jitters, len_sets, coeffs, A)

    !implicit Real*4 (a-h,o-z)
    real, intent(in) :: times(:), rvs(:), errs(:)
    real, intent(in) :: freqs(:), jitters(:)
    integer,  intent(in) :: len_sets(:)
    real, intent(out):: coeffs(2*size(freqs)+size(len_sets))
    real :: F(size(times),2*size(freqs)+size(len_sets)), Ft(2*size(freqs)+size(len_sets), size(times))
    real :: b(2*size(freqs)+size(len_sets)), y_err(size(times))
    real, intent(out):: A(2*size(freqs)+size(len_sets),2*size(freqs)+size(len_sets))
    real ::  pi=acos(-1.0), t_ref = 0.0, errs_jitter(size(times))
    integer :: i, j, k, final, n_dim, n_sets, LS_INFO, pivot(2*size(freqs)+size(len_sets))
    integer :: init_times(size(len_sets)) 
     
    !number of datasets
    n_sets = size(len_sets)
    !total length of data (list of all datasets concatenated)
    n_data = size(times)
    !num of dimensions
    n_dim = size(freqs)
    
    !init times for each set (store array index)
    ind = 1
    do i = 1, n_sets
        init_times(i) = ind
        ind = ind + len_sets(i)
    enddo
    
    !init matrix
    do k = 1, 2*n_dim + n_sets 
        do j = 1, n_data
            F(j,k) = 0.0
        enddo
    enddo
    
    !fill offsets + jitters
    do k = 1, n_sets
        if (k.EQ.n_sets) then
            final = n_data
        else
            final = init_times(k+1) - 1 
        endif
        
        do j = init_times(k), final
            errs_jitter(j) = SQRT(errs(j)**2.0 + jitters(k)**2.0)  !redefinition of errs variable to include jitters
            F(j,k) = 1.0/(errs_jitter(j))
        enddo
    enddo
    
    t_ref = times(1)
    !fill {cos,sin} elements
    do 11 k = n_sets, 2*n_dim + n_sets
        if ((k.GT.n_sets).AND.(k.LE.(n_dim+n_sets))) then
            do j = 1,n_data
                F(j,k) = cos(2.0*pi*(times(j)-t_ref)*freqs(k-n_sets))/errs_jitter(j)
            enddo    
        
        else if ((k.GT.(n_dim+n_sets)).AND.(k.LE.(2*n_dim+n_sets))) then
            do j = 1,n_data
                F(j,k) = sin(2.0*pi*(times(j)-t_ref)*freqs(k-(n_dim+n_sets)))/errs_jitter(j)
              enddo
        
        endif    
        
11  enddo
    ! b array
    do i = 1, n_data
         y_err(i) = rvs(i)/errs_jitter(i)
    enddo

    !arrange linear system F_t*F*theta = F_t*y_e --> A*theta = b
    Ft = transpose(F)
    A = matmul(Ft,F)
    b = matmul(Ft,y_err)
    !call sgemm('t','n',2*n_dim+n_sets,2*n_dim+n_sets,n_data,1.0,F,n_data,F,n_data,0.0,A,2*n_dim+n_sets)
    !call sgemm('t','n',2*n_dim+n_sets,1,n_data,1.0,F,n_data,y_err,n_data,0,b,2*n_dim+n_sets)
    
    !solves a system of linear equations A*X = B 
    !call SPOSV('U', 2*n_dim+n_sets, 1, A, 2*n_dim+n_sets, b, 2*n_dim+n_sets, LS_INFO)
    call SGESV(2*n_dim+n_sets, 1, A, 2*n_dim+n_sets, pivot, b, 2*n_dim+n_sets, LS_INFO)
    
    !fill output vector: fitting coeffs.
    do i = 1, 2*n_dim + n_sets 
        coeffs(i) = b(i)
    enddo
    

end subroutine mdim_gls_multiset

subroutine mdim_gls_multiset_trend(times, rvs, errs, freqs, jitters, len_sets, coeffs, A)

    !implicit Real*4 (a-h,o-z)
    real, intent(in) :: times(:), rvs(:), errs(:)
    real, intent(in) :: freqs(:), jitters(:)
    integer,  intent(in) :: len_sets(:)
    real, intent(out):: coeffs(2*size(freqs)+size(len_sets)+1)
    real :: F(size(times),2*size(freqs)+size(len_sets)+1), Ft(2*size(freqs)+size(len_sets)+1, size(times))
    real :: b(2*size(freqs)+size(len_sets)+1), y_err(size(times))
    real, intent(out):: A(2*size(freqs)+size(len_sets)+1,2*size(freqs)+size(len_sets)+1)
    real ::  pi=acos(-1.0), t_ref = 0.0, errs_jitter(size(times))
    integer :: i, j, k, final, n_dim, n_sets, LS_INFO, pivot(2*size(freqs)+size(len_sets)+1)
    integer :: init_times(size(len_sets)) 
  
    !number of datasets
    n_sets = size(len_sets)
    !total length of data (list of all datasets concatenated)
    n_data = size(times)
    !num of dimensions
    n_dim = size(freqs)
    
    !init times for each set (store array index)
    ind = 1
    do i = 1, n_sets
        init_times(i) = ind
        ind = ind + len_sets(i)
    enddo
    
    !init matrix
    do k = 1, 2*n_dim + n_sets + 1
        do j = 1, n_data
            F(j,k) = 0.0
        enddo
    enddo
    
    !fill offsets + jitters
    do k = 1, n_sets
        if (k.EQ.n_sets) then
            final = n_data
        else
            final = init_times(k+1) - 1 
        endif
        
        do j = init_times(k), final
            errs_jitter(j) = SQRT(errs(j)**2.0 + jitters(k)**2.0)  !redefinition of errs variable to include jitters
            F(j,k) = 1.0/(errs_jitter(j))
        enddo
    enddo
    
    t_ref = times(1)
    !fill {cos,sin} elements
    do 11 k = n_sets, 2*n_dim + n_sets + 1
        if ((k.GT.n_sets).AND.(k.LE.(n_dim+n_sets))) then
            do j = 1,n_data
                F(j,k) = cos(2.0*pi*(times(j)-t_ref)*freqs(k-n_sets))/errs_jitter(j)
            enddo    
        
        else if ((k.GT.(n_dim+n_sets)).AND.(k.LE.(2*n_dim+n_sets))) then
            do j = 1,n_data
                F(j,k) = sin(2.0*pi*(times(j)-t_ref)*freqs(k-(n_dim+n_sets)))/errs_jitter(j)
              enddo
        
        else if (k.GT.(2*n_dim+n_sets)) then
            do j = 1,n_data
                F(j,k) = (times(j)-t_ref)/errs_jitter(j)
            enddo
        
        endif    
        
11  enddo
    ! b array
    do i = 1, n_data
         y_err(i) = rvs(i)/errs_jitter(i)
    enddo

    !arrange linear system F_t*F*theta = F_t*y_e --> A*theta = b
    Ft = transpose(F)
    A = matmul(Ft,F)
    b = matmul(Ft,y_err)
    !call sgemm('t','n',2*n_dim+n_sets+1,2*n_dim+n_sets+1,n_data,1.0,F,n_data,F,n_data,0.0,A,2*n_dim+n_sets+1)
    !call sgemm('t','n',2*n_dim+n_sets+1,1,n_data,1.0,F,n_data,y_err,n_data,0,b,2*n_dim+n_sets+1)
    !solves a system of linear equations A*X = B 
    !call SPOSV('U', 2*n_dim+n_sets+1, 1, A, 2*n_dim+n_sets+1, b, 2*n_dim+n_sets+1, LS_INFO)
    call SGESV(2*n_dim+n_sets+1, 1, A, 2*n_dim+n_sets+1, pivot, b, 2*n_dim+n_sets+1, LS_INFO)
    
    !fill output vector: fitting coeffs.
    do i = 1, 2*n_dim + n_sets + 1
        coeffs(i) = b(i)
    enddo
    

end subroutine mdim_gls_multiset_trend

subroutine model(t_p, t_ref, freqs, fitting_coeffs, y_model)
    !computes model according to fitting coefficients, frequencies
    double precision, intent(in) :: t_p, t_ref, fitting_coeffs(:), freqs(:)
    double precision, intent(out) :: y_model
    double precision :: pi=dacos(-1.0d0)
    integer :: k, n_dim
    
    n_dim = size(freqs)
    y_model = 0.0
    do k = 1,size(fitting_coeffs)
        if (k.EQ.1) then
            y_model = y_model + fitting_coeffs(k)
        else if ((k.GT.1).AND.(k.LE.(n_dim+1))) then
            y_model = y_model + &
                      fitting_coeffs(k)*cos(2.0*pi*(t_p-t_ref)*freqs(k-1))
        else if ((k.GT.(n_dim+1)).AND.(k.LT.(2*n_dim+2))) then
            y_model = y_model + &
                fitting_coeffs(k)*sin(2.0*pi*(t_p-t_ref)*freqs((k-(n_dim+1))))
        !else if (k.EQ.(2*n_dim+2)) then
        !    y_model = y_model + fitting_coeffs(k)*(t_p-t_ref)
        endif
    enddo

end subroutine model

subroutine model_series(t, t_ref, freqs, fitting_coeffs, n_points, y_model_s)
    !computes model according to fitting coefficients, frequencies
    double precision, intent(in) :: t(:), t_ref, fitting_coeffs(:), freqs(:)
    integer, intent(in) :: n_points
    double precision, intent(out) :: y_model_s(n_points)
    double precision :: pi=dacos(-1.0d0), y_model
    integer :: k, n_dim
    
    n_dim = size(freqs)

    do i = 1,n_points
        y_model = 0.0
        do k = 1,size(fitting_coeffs)
            if (k.EQ.1) then
                y_model = y_model + fitting_coeffs(k)
            else if ((k.GT.1).AND.(k.LE.(n_dim+1))) then
                y_model = y_model + &
                        fitting_coeffs(k)*cos(2.0*pi*(t(i)-t_ref)*freqs(k-1))
            else if ((k.GT.(n_dim+1)).AND.(k.LT.(2*n_dim+2))) then
                y_model = y_model + &
                    fitting_coeffs(k)*sin(2.0*pi*(t(i)-t_ref)*freqs((k-(n_dim+1))))
            else if (k.EQ.(2*n_dim+2)) then  !if exists a linear trend slope
                y_model = y_model + fitting_coeffs(k)*(t(i)-t_ref)
            endif
        enddo
        y_model_s(i) = y_model        
    enddo
        
end subroutine model_series

subroutine model_series_multiset(times, freqs, fitting_coeffs, len_sets, y_model_s)
    !computes model according to fitting coefficients, frequencies
    double precision, intent(in) :: times(:), fitting_coeffs(:), freqs(:)
    integer, intent(in) :: len_sets(:)
    double precision, intent(out) :: y_model_s(size(times))
    double precision :: pi=dacos(-1.0d0), y_model, t_ref
    integer :: i, k, s, n_dim, n_sets, n_data, ind, init_times(size(len_sets)), end_data
    
    n_dim = size(freqs)
    n_data = size(times)
    n_sets = size(len_sets)
    
    !init times (store array index)
    ind = 1
    do i = 1, n_sets
        init_times(i) = ind
        ind = ind + len_sets(i)
    enddo
    !reference time
    t_ref = times(1)
    
    do s = 1, n_sets 
        !define indexes in concatenated list of data
        if (s.EQ.n_sets) then
            end_data = n_data
        else
            end_data = init_times(s+1) - 1 
        endif
       
        do i = init_times(s), end_data 
            y_model = 0.0
            do k = 1,size(fitting_coeffs)
                if (k.EQ.s) then
                    y_model = y_model + fitting_coeffs(k)
                else if ((k.GT.n_sets).AND.(k.LE.(n_dim+n_sets))) then
                    y_model = y_model + &
                            fitting_coeffs(k)*cos(2.0*pi*(times(i)-t_ref)*freqs(k-n_sets))
                else if ((k.GT.(n_dim+n_sets)).AND.(k.LT.(2*n_dim+n_sets+1))) then
                    y_model = y_model + &
                        fitting_coeffs(k)*sin(2.0*pi*(times(i)-t_ref)*freqs((k-(n_dim+n_sets))))
    
                endif
            enddo
            y_model_s(i) = y_model        
        enddo
        
    enddo
        
end subroutine model_series_multiset

subroutine chi2(t, y, err, t_ref, freqs, fitting_coeffs, chi2_out)
!   computes chi2 between data and model
    double precision, intent(in) :: t(:), y(:), err(:), t_ref
    double precision, intent(in) :: freqs(:), fitting_coeffs(:)
    double precision, intent(out) :: chi2_out
    double precision :: y_model
    double precision :: pi=dacos(-1.0d0)
    integer :: i, k, n_dim
    
    n_dim = size(freqs)
    chi2_out = 0.0
    do 97 i = 1, size(t)
        y_model = 0.0
        !call model(t(i), t_ref, freqs, fitting_coeffs, y_model)
         do k = 1,size(fitting_coeffs)
            if (k.EQ.1) then
                y_model = y_model + fitting_coeffs(k)
            else if ((k.GT.1).AND.(k.LE.(n_dim+1))) then
                y_model = y_model + &
                        fitting_coeffs(k)*cos(2.0*pi*(t(i)-t_ref)*freqs(k-1))
            else if ((k.GT.(n_dim+1)).AND.(k.LT.(2*n_dim+2))) then
                y_model = y_model + &
                    fitting_coeffs(k)*sin(2.0*pi*(t(i)-t_ref)*freqs((k-(n_dim+1))))
            !else if (k.EQ.(2*n_dim+2)) then
            !    y_model = y_model + fitting_coeffs(k)*(t(i)-t_ref)
            endif
        enddo
        chi2_out = chi2_out + (1.0/(err(i)*err(i)))*(y(i) - y_model)**2.0
97  continue

end subroutine chi2

subroutine chi2_0(y, err, chi2_0_out)
!   computes chi2 respect to the mean
    double precision, intent(in) :: y(:), err(:)
    double precision, intent(out) :: chi2_0_out
    double precision :: mean_y
    integer :: i

    chi2_0_out = 0.0
    mean_y = 0.0
    do 96 j = 1,size(y)
        mean_y = mean_y + y(j)
96  continue
    mean_y = mean_y/size(y)

    do 97 i = 1,size(y)
        chi2_0_out = chi2_0_out + &
                  (1.0/(err(i)*err(i)))*(y(i) - mean_y)**2.0
97  continue

end subroutine chi2_0

subroutine residuals(t, y, t_ref, freqs, fitting_coeffs, residuals_out)
!   computes chi2 between data and model
    double precision, intent(in) :: t(:), y(:), t_ref
    double precision, intent(in) :: freqs(:), fitting_coeffs(:)
    double precision, intent(out) :: residuals_out(size(t))
    double precision :: y_model
    double precision :: pi=dacos(-1.0d0)
    integer :: i, k, n_dim
    
    n_dim = size(freqs)
    do 97 i = 1, size(t)
        !call model(time_p, t_ref, periods, fitting_coeffs, y_model)
        y_model = 0.0
        do k = 1,size(fitting_coeffs)
            if (k.EQ.1) then
                y_model = y_model + fitting_coeffs(k)
            else if ((k.GT.1).AND.(k.LE.(n_dim+1))) then
                y_model = y_model + &
                        fitting_coeffs(k)*cos(2.0*pi*(t(i)-t_ref)*freqs(k-1))
            else if ((k.GT.(n_dim+1)).AND.(k.LT.(2*n_dim+2))) then
                y_model = y_model + &
                    fitting_coeffs(k)*sin(2.0*pi*(t(i)-t_ref)*freqs((k-(n_dim+1)))) 
            !else if (k.EQ.(2*n_dim+2)) then
            !    y_model = y_model + fitting_coeffs(k)*(t(i)-t_ref)
            endif
        enddo
        residuals_out(i) = y(i) - y_model
97  continue

end subroutine residuals
