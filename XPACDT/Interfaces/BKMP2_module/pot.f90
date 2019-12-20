! Standardized subroutine to call 3 atom potentials from Python. Here
! it calls the BKMP2 H+H2 PES.
! Input:
!      xin : 1d float array; Cartesian coordinates
!      inpath : string; path to the parameter files. (not used here!)
! Output:
!      v : float; potential energy in hartree
!      dv : 1d float array; derivatives with respect to cartesian coordinates in hartree/au
subroutine pot(xin, inpath, v, dv)
  implicit none
  
  real*8, intent(in) :: xin(9)
  CHARACTER(100), intent(in) :: inpath
  real*8, intent(out) :: v, dv(9)

  real*8 :: pothco
  real*8 :: r(3), dvdr(3)
  integer :: i, k

  v = 0.d0

  r(1) = 0.d0
  r(2) = 0.d0
  r(3) = 0.d0

  ! r are all three interatomic distances; HH, HF, HF
  do i = 1, 3
     r(1) = r(1) + (xin(0+i) - xin(3+i))*(xin(0+i) - xin(3+i))
     r(2) = r(2) + (xin(0+i) - xin(6+i))*(xin(0+i) - xin(6+i))
     r(3) = r(3) + (xin(3+i) - xin(6+i))*(xin(3+i) - xin(6+i))
  enddo

  r(1) = dsqrt(r(1))
  r(2) = dsqrt(r(2))
  r(3) = dsqrt(r(3))

  ! Call BKMP2 PES; Requires interatomic distances, Gives back energy and derivatives in internal coordinates
  call bkmp2(r, v, dvdr, 1)

  ! Convert the derivatives in internal coordinates to cartesian coordinates
  do i = 1, 3
     dv(0+i) = dvdr(1) * ((xin(0+i) - xin(3+i)) / r(1)) 
     dv(0+i) = dv(0+i) + dvdr(2) * ((xin(0+i) - xin(6+i)) / r(2))

     dv(3+i) = dvdr(1) * (-(xin(0+i) - xin(3+i)) / r(1)) 
     dv(3+i) = dv(3+i) + dvdr(3) * ((xin(3+i) - xin(6+i)) / r(3))

     dv(6+i) = dvdr(2) * (-(xin(0+i) - xin(6+i)) / r(2)) 
     dv(6+i) = dv(6+i) + dvdr(3) * (-(xin(3+i) - xin(6+i)) / r(3))
  enddo

end subroutine pot

! Initialize - doesn't do much here
subroutine pes_init()
  implicit none

  return

end subroutine pes_init
