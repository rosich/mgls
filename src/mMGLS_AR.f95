subroutine mdim_gls(t, y, err, freqs, params, ar, t_ref, coeffs, A)

    !implicit Real*4 (a-h,o-z)
    Real*4, intent(in) :: t(:), y(:), err(:)
    Real*4, intent(in) :: freqs(:), params(:), t_ref
    integer, intent(in) :: ar
    Real*4, intent(out):: coeffs(2*size(freqs)+1+ar)
    Real*4 :: F(size(t),2*size(freqs)+1+ar), Ft(2*size(freqs)+1+ar, size(t)), b(2*size(freqs)+1+ar), y_err(size(t))
    Real*4 , intent(out):: A(2*size(freqs)+1+ar,2*size(freqs)+1+ar)
    Real*4 ::  pi=acos(-1.0)
    integer :: i, j, k, n_dim, delay, LS_INFO, pivot(2*size(freqs)+1+ar)

    n_dim = size(freqs)
    n_data = size(t)
    n_params = size(params)
    
    do k = 1, 2*n_dim+1+ar
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
        else if (k.GE.(2*n_dim+2)) then
             delay = k - (2*n_dim+1)
             do j = 1,n_data
                 if (j.GT.delay) then   
                    F(j,k) = y(j-delay)*exp((t(j-delay)-t(j))/params(1))/err(j-delay)
                 else
                    F(j,k) = 0.0
                 endif
             enddo
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
    !call DPOTRS('L', n_dim, 1, A, n_dim, b, n_dim, LS_INFO)
    call SGESV(2*n_dim+1+ar, 1, A, 2*n_dim+1+ar, pivot, b, 2*n_dim+1+ar, LS_INFO)
    !fill output vector: fitting coeffs.
    do i = 1,2*n_dim+1+ar
        coeffs(i) = b(i)
    enddo

end subroutine mdim_gls

subroutine model(t, y, freqs, fitting_coeffs, params, y_model_array)
    !computes model according to fitting coefficients, frequencies and params
    double precision, intent(in) :: t(:), y(:), freqs(:), fitting_coeffs(:), params(:) 
    double precision, intent(out) :: y_model_array(size(t))
    double precision :: y_model, t_p, t_ref
    double precision :: pi=dacos(-1.0d0)
    integer :: k, n_dim, delay
    
    t_ref = t(1)
    n_dim = size(freqs)
    n_data = size(t)
    do j = 1, n_data
        y_model = 0.0
        t_p = t(j)
        do k = 1,size(fitting_coeffs)
            if (k.EQ.1) then
                y_model = y_model + fitting_coeffs(k)
            else if ((k.GT.1).AND.(k.LE.(n_dim+1))) then
                y_model = y_model + &
                        fitting_coeffs(k)*cos(2.0*pi*(t_p-t_ref)*freqs(k-1))
            else if ((k.GT.(n_dim+1)).AND.(k.LT.(2*n_dim+2))) then
                y_model = y_model + &
                    fitting_coeffs(k)*sin(2.0*pi*(t_p-t_ref)*freqs((k-(n_dim+1))))
            else if (k.GE.(2*n_dim+2)) then
                    delay = k - (2*n_dim+1)
                    if (j.GT.delay) then   
                        y_model = y_model + fitting_coeffs(k)*exp((t(j-delay)-t(j))/params(1))*y(j-delay)
                    endif
            endif
        enddo
        y_model_array(j) = y_model
    enddo

end subroutine model

subroutine chi2(t, y, err, model, chi2_out)
!   computes chi2 between data and model
    double precision, intent(in) :: t(:), y(:), err(:), model(:)
    double precision, intent(out) :: chi2_out
    integer :: i
   
    chi2_out = 0.0
    do 97 i = 1, size(t)
        chi2_out = chi2_out + (1.0/(err(i)*err(i)))*(y(i) - model(i))**2.0
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

subroutine residuals(t, y, model, residuals_out)
!   computes chi2 between data and model
    double precision, intent(in) :: t(:), y(:), model(:)
    double precision, intent(out) :: residuals_out(size(t))
    integer :: i
    
    do 97 i = 1, size(t)   
        residuals_out(i) = y(i) - model(i)
97  continue

end subroutine residuals