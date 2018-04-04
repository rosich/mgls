
subroutine gls_pow(t, y, err, periods, t_ref, periodogram)

    implicit Real*4 (a-h,o-z)

    !integer, intent(in) :: n_per, n_pts
    Real*4, intent(in) :: t(:), y(:), err(:)
    Real*4, intent(in) :: periods(:), t_ref
    Real*4, intent(out):: periodogram(size(periods),5)
    Real*4 :: pwr, b(3), period
    Real*4 :: s_y, s_y2, s_cos, s_sin, s_cos2
    Real*4 :: s_sin2, s_sincos, s_ysin, s_ycos
    Real*4 :: w, x_arg, weight
    Real*4 :: A(3,3), D, SS, CC, CS
    Real*4 :: YY, YC, YS, pi
    integer :: i, j, k, pivot(3), INFO, len_periods, len_data
    !pi def.
    pi = acos(-1.0)
    weight = 0.0
    !compute W
    len_data = size(y)
    do 97 i = 1, len_data
        weight = weight + 1.0/(err(i)*err(i))
97  continue
    len_periods = size(periods)
    do 98 j = 1, len_periods
        period = periods(j)
        s_y = 0.0
        s_y2 = 0.0
        s_cos = 0.0
        s_sin = 0.0
        s_cos2 = 0.0
        s_sin2 = 0.0
        s_sincos = 0.0
        s_ysin = 0.0
        s_ycos = 0.0
        pwr = 0.0
        do 99 i = 1, len_data
            !phase of sinusoid
            x_arg = 2.0*pi*(t(i)-t_ref)/period
            !weight
            w = 1.0/(err(i)*err(i)*weight)
            s_y = s_y + w*y(i)
            s_y2 = s_y2 + w*y(i)*y(i)
            s_cos = s_cos + w*cos(x_arg)
            s_sin = s_sin + w*sin(x_arg)
            s_cos2 = s_cos2 + w*cos(x_arg)*cos(x_arg)
            s_sin2 = s_sin2 + w*sin(x_arg)*sin(x_arg)
            s_sincos = s_sincos + w*sin(x_arg)*cos(x_arg)
            s_ysin = s_ysin + w*y(i)*sin(x_arg)
            s_ycos = s_ycos + w*y(i)*cos(x_arg)
99      continue
        !definition of chi-square normalization
        !solving the matrix equation A*x=b using LAPACK library
        !define matrix A
        A(1,1) = s_cos2
        A(1,2) = s_sincos
        A(1,3) = s_cos
        A(2,1) = s_sincos
        A(2,2) = s_sin2
        A(2,3) = s_sin
        A(3,1) = s_cos
        A(3,2) = s_sin
        A(3,3) = 1.0
    ! define vector b
        b(1) = s_ycos
        b(2) = s_ysin
        b(3) = s_y
    !   find the solution using the LAPACK routine SGESV
    !   parameters in the order as they appear in the function call
    !   order of matrix A, number of right hand sides (b), matrix A,
    !   leading dimension of A, array that records pivoting,
    !   result vector b on entry, x on exit, leading dimension of b
    !   return value
        call SGESV(3, 1, A, 3, pivot, b, 3, INFO)

        !compute spectral power, p(w)
        CC = s_cos2 - s_cos*s_cos
        SS = s_sin2 - s_sin*s_sin
        CS = s_sincos - s_cos*s_sin
        YY = s_y2 - s_y*s_y
        YC = s_ycos - s_y*s_cos
        YS = s_ysin - s_y*s_sin
        D = CC*SS - (CS*CS)
        !spectral power
        pwr = (SS*YC*YC + CC*YS*YS - 2.0*CS*YC*YS)/(YY*D)
        !write result in output array [per, pwr, a,b,c]
        periodogram(j,1) = period
        periodogram(j,2) = pwr
        do k = 1,3
            periodogram(j,k+2) = b(k)
        enddo

98  continue

end subroutine gls_pow

subroutine chi2(t, y, err, t_ref, max_pwr, chi2_out)
!   computes chi2 between data and a*coswt + b*sinwt c model
    double precision, intent(in) :: t(:), y(:), err(:)
    double precision, intent(in) :: max_pwr(5), t_ref
    double precision, intent(out) :: chi2_out
    double precision :: a, b, c, per, pi
    integer :: i, n_pts
    !pi def.
    pi = acos(-1.0d0)
    n_pts = size(t)
    !model parameters
    per = max_pwr(1)
    a = max_pwr(3)
    b = max_pwr(4)
    c = max_pwr(5)

    chi2_out = 0.0
    do 97 i = 1, n_pts
        chi2_out = chi2_out + &
                  (1.0/(err(i)*err(i)))*(y(i) - (a*cos(2.0*pi*(t(i)-t_ref)/per) +  &
                  b*sin(2.0*pi*(t(i)-t_ref)/per) + c))**2.0
97  continue

end subroutine chi2

subroutine chi2_0(y, err, chi2_0_out)
!   computes chi2 between data and a*coswt + b*sinwt c model
    double precision, intent(in) :: y(:), err(:)
    double precision, intent(out) :: chi2_0_out
    double precision :: mean_y
    integer :: i, j

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













