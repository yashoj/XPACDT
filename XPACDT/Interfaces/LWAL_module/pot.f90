subroutine pot(xin, inpath, v, dv)
  implicit none
  
  real*8, intent(in) :: xin(9)
  CHARACTER(100), intent(in) :: inpath
  real*8, intent(out) :: v, dv(9)

  real*8 :: pothco, vp, vm, theta, dp, f
  real*8 :: r(3), dvdr(3), getR3
  real*8 :: dtda, dphi, dr1, dr2
!  real*8 :: step
  parameter step = 1.d-4

  integer :: i, k

  CHARACTER(100) path
  COMMON /pathv/ path
  path = inpath

  v = 0.d0

  r(1) = 0.d0
  r(2) = 0.d0
  r(3) = 0.d0

  dp = 0.d0
  do i = 0, 2
     r(1) = r(1) + (xin(4+i) - xin(7+i))*(xin(4+i) - xin(7+i))
     r(2) = r(2) + (xin(4+i) - xin(1+i))*(xin(4+i) - xin(1+i))
     r(3) = r(3) + (xin(1+i) - xin(7+i))*(xin(1+i) - xin(7+i))

     dp = dp + ((xin(4+i) - xin(7+i))*(xin(4+i) - xin(1+i)))
  enddo

  r(1) = dsqrt(r(1))
  r(2) = dsqrt(r(2))
  r(3) = dsqrt(r(3))

  f = dp / (r(1)*r(2))
  theta = dacos(f)

!  write (6, *), f, theta
!, getR3(r(1), r(2), theta), r(3)

  call pot_fh2(r, v)

  do i = 1, 2
     r(i) = r(i) + step
     r(3) = getR3(r(1), r(2), theta)
     call pot_fh2(r, vp)

     r(i) = r(i) - 2.d0*step
     r(3) = getR3(r(1), r(2), theta)
     call pot_fh2(r, vm)

     dvdr(i) = (vp - vm) / (2.d0 * step)

     r(i) = r(i) + step
  enddo

  theta = theta + step
  r(3) = getR3(r(1), r(2), theta)
  call pot_fh2(r, vp)
  
  theta = theta - 2.d0*step
  r(3) = getR3(r(1), r(2), theta)
  call pot_fh2(r, vm)

  dvdr(3) = (vp - vm) / (2.d0 * step)
  
  theta = theta + step

  if (abs((abs(f)-1.0)).lt.1.d-6) then
     dtda = 0.d0
  else
     dtda = -1.d0 / sqrt(1.d0 - f*f)
  endif
!  write(6,*) "dtda=", dtda

!!! do chain rule here for theta derivative
  do i = 0, 2
     dv(1+i) = dvdr(2) * ((xin(1+i) - xin(4+i)) / r(2)) 
     dphi = (-xin(4+i) + xin(7+i))*r(1)*r(2)
     dr2 = dp * r(1) * (-(xin(4+i)-xin(1+i))) / r(2)
     dv(1+i) = dv(1+i) + dvdr(3) * dtda * (dphi - dr2) /  (r(1)*r(1)*r(2)*r(2))

     dv(4+i) = dvdr(2) * (-(xin(1+i) - xin(4+i)) / r(2)) 
     dv(4+i) = dv(4+i) + dvdr(1) * ((xin(4+i) - xin(7+i)) / r(1))

     dphi = (2.d0*xin(4+i) - (xin(1+i) + xin(7+i)))*r(1)*r(2)
     dr1 = dp * r(2) * ((xin(4+i)-xin(7+i))) / r(1)
     dr2 = dp * r(1) * ((xin(4+i)-xin(1+i))) / r(2)
     dv(4+i) = dv(4+i) + dvdr(3) * dtda * (dphi - dr2 - dr1) /  (r(1)*r(1)*r(2)*r(2))


     dv(7+i) = dvdr(1) * (-(xin(4+i) - xin(7+i)) / r(1))
     dphi = (-xin(4+i) + xin(1+i))*r(1)*r(2)
     dr1 = dp * r(2) * (-(xin(4+i)-xin(7+i))) / r(1)
     dv(7+i) = dv(7+i) + dvdr(3) * dtda * (dphi - dr1) /  (r(1)*r(1)*r(2)*r(2))
  enddo

end subroutine pot



subroutine pes_init()
  implicit none

  CHARACTER(100) :: path, inpath
  COMMON /pathv/ path

  path = "/home/ralph/code/XPACDT/XPACDT/Interfaces/LWAL_module/"
!inpath

  return

end subroutine pes_init

real*8 function getR3(r1, r2, theta)
  implicit none

  real*8:: r1, r2, theta, r3

  r3 = dsqrt(r1*r1 + r2*r2 - 2.d0*r1*r2*dcos(theta))
  
  getR3 = r3
  return 
end function getR3
