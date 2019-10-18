c----------------------------------------------------------------------
       subroutine poth2cl(rbond,vev,iflag)
c----------------------------------------------------------------------
       implicit double precision (a-h,o-z)
*
* calculates lowest diabatic Cl+H2 surfaces and spin-orbit coupling
* from fit to capecchi-bian-werner surfaces including spin-orbit coupling
* input variable (dimension 3) (distances in bohr)
*
* G. Capecchi and H.-J. Werner (2001)
*
*      rbond(1)=rh2
*      rbond(2)=rclh1
*      rbond(3)=rclh2
*
* iflag=0:  return 6 diabatic potentials
* iflag=1:  return 3 adiabatic potentials (eigenvalues of H + HSO)
* iflag=2:  return BW potential

* The resulting energy values are in eV
*
* for iflag=0:
*      vev(1)=Vsig
*      vev(2)=Vpi= 1/2 [ Vpi(A") + Vpi(A') ]
*      vev(3)=v2 = 1/2 [ Vpi(A") - Vpi(A') ]
*      vev(4)=v1 = < Vsig | H | Vpi(A') > / sqrt(2)
*      vev(5)= A = i<Piy(alpha)|Hso|pix(alpha)>
*      vev(6)= B =  <Pix(beta)|Hso|Sigma(alpha)>
*
* for iflag=1  (Eigenstates of Capecchi-Werner Potential)
*      vev(1)=E1
*      vev(2)=E2
*      vev(3)=E3
*
* for iflag=2
*      vev(1)=E1  (Bian-Werner potential)
*
c
      dimension  vev(6),rbond(3),h(6,6),e(6)
c
      data toev/27.21139610d0/
      data tocm/219474.63067d0/
      data sq2i/0.7071067811865475d0/
      data sq2 /1.4142135623730950d0/
      data inicw/0/
      save inicw
c
      rh2=rbond(1)
      rclh1=rbond(2)
      rclh2=rbond(3)
c
      if(iflag.le.1) then
        if(inicw.eq.0) then
          call inifit_cw
          inicw=1
        end if
        call jacobi(rclh1,rh2,rclh2,rr,r,theta)
      end if
c
      if(iflag.eq.0) then
c
c... Capecchi-Werner diabatic potentials
c
        do icase=1,6
          vev(icase)=cwpot(rr,r,theta,icase)*toev
        end do
        vev(4)=vev(4)*sq2i
        vev(5)=vev(5)/tocm
        vev(6)=vev(6)/tocm
        return
c
      else if(iflag.eq.1) then
c
c... Capecchi-Werner adiabatic potentials
c
        vsig=cwpot(rr,r,theta,1)
        vpi =cwpot(rr,r,theta,2)
        v2=cwpot(rr,r,theta,3)
        v1=cwpot(rr,r,theta,4)*sq2i
        a= cwpot(rr,r,theta,5)/tocm
        b=-cwpot(rr,r,theta,6)*sq2/tocm
c
        h(1,1)=vsig
        h(2,1)=0
        h(3,1)=-v1
        h(4,1)=b
        h(5,1)=v1
        h(6,1)=0
c
        h(1,2)=0
        h(2,2)=vsig
        h(3,2)=0
        h(4,2)=-v1
        h(5,2)=b
        h(6,2)=v1
c
        h(1,3)=-v1
        h(2,3)=0
        h(3,3)=vpi-a
        h(4,3)=0
        h(5,3)=v2
        h(6,3)=0
c
        h(1,4)=b
        h(2,4)=-v1
        h(3,4)=0
        h(4,4)=vpi+a
        h(5,4)=0
        h(6,4)=v2
c
        h(1,5)=v1
        h(2,5)=b
        h(3,5)=v2
        h(4,5)=0
        h(5,5)=vpi+a
        h(6,5)=0
c
        h(1,6)=0
        h(2,6)=v1
        h(3,6)=0
        h(4,6)=v2
        h(5,6)=0
        h(6,6)=vpi-a
c
        call diag2(6,6,e,h)
        if (vsig .gt. vpi+a) then
           vev(1)=e(6)*toev
           vev(3)=e(1)*toev
        else
           vev(1)=e(1)*toev
           vev(3)=e(6)*toev
        endif
        vev(2)=e(3)*toev
        return
c
      else if(iflag.eq.2) then
c
c... Bian-Werner potential
c
        vev(1)=bwpotex(rclh1,rh2,rclh2)*toev
        return
      end if
      end
c----------------------------------------------------------------------
      subroutine inifit_cw
c----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      character*32 filnam
      character*100 path
      parameter (maxb=15,maxpar=maxb*maxb*maxb)
      common/cdim/ kmax(6),lmax(6),nmax(6)
      common/cpar/ par_rr(10,6),par_r(10,6),par_theta(10,6)
      common/csol/ s(maxpar,6)
      common/cvwex/ rmincw(3,6),emax(6),dr(6)
      common/pathv/ path
      data filnam/'cwfit.dat'/
      open(91,file=TRIM(ADJUSTL(path))//filnam,status='old')
      do i=1,6
        read(91,*) nmax(i),lmax(i),kmax(i)
        read(91,*) npar_RR,(par_rr(k,i),k=1,npar_rr)
        read(91,*) npar_r,(par_r(k,i),k=1,npar_r)
        read(91,*) npar_theta,(par_theta(k,i),k=1,npar_theta)
        npar=kmax(i)*lmax(i)*nmax(i)
        read(91,*) rmincw(1,i),rmincw(2,i),rmincw(3,i),
     >            emax(i),dr(i)
        read(91,*) (s(k,i),k=1,npar)
c       write(6,*) 'kmax(i),lmax(i),nmax(i)',kmax(i),lmax(i),nmax(i)
c       write(6,*) 'npar_RR',npar_RR,(par_rr(k,i),k=1,npar_rr)
c       write(6,*) 'npar_r',npar_r,(par_r(k,i),k=1,npar_r)
c       write(6,*) 'npar_theta', npar_theta,(par_theta(k,i),
c    >              k=1,npar_theta)
c       write(6,*) 's',(s(k,i),k=1,npar)
c       write(6,*)  rmincw,emax(i),dr(i)
      end do
      return
      end
c--------------------------------------------------------------------
      function cwpot(rr,r,theta,icase)
c--------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      parameter (maxb=15,maxpar=maxb*maxb*maxb)
      common/cdim/ kmax(6),lmax(6),nmax(6)
      common/cpar/ par_rr(10,6),par_r(10,6),par_theta(10,6)
      common/csol/ s(maxpar,6)
      common/cvwex/ rmincw(3,6),emax(6),dr(6)
      data wthr/1.d-2/
c     data wthr/1.d-3/
c
      call distjac(r1,r2, rr,r,theta)
      if(icase.le.1) then
        wcw=0.25d0*(1.d0+tanh((2.75d0-r )/0.2d0))
     >            *(1.d0+tanh((rr-3.9d0)/0.2d0))
        wbw=1.d0-wcw
        if(wcw.lt.wthr) then
          cwpot=bwpotex(r1,r,r2)
        else if(wbw.lt.wthr) then
          cwpot=cwpotex_sig(rr,r,theta,icase)
        else
          vcw=cwpotex_sig(rr,r,theta,icase)
          vbw=bwpotex(r1,r,r2)
          cwpot=wcw*vcw+wbw*vbw
        end if
        return
      else if(icase.eq.2) then
        cwpot=cwpotex_pi(rr,r,theta,icase)
        if(abs(cwpot).lt.1.d-8) cwpot=0
      else if(icase.eq.3) then
        vcw=cwpot1(rr,r,theta,icase)
        if(rr.lt.rmincw(1,icase)) vcw=cwpot1(rmincw(1,icase),
     >                                r,theta,icase)
        vcw=min(vcw,0.d0)
        if(abs(vcw).lt.1.d-8) vcw=0
          wcw=0.125d0*(1.d0+tanh((2.65d0-r)/0.1d0))
     >            *(1.d0+tanh((r-0.80d0)/0.20d0))
     >            *(1.d0+tanh((min(r1,r2)-2.80d0)/0.2d0))
        if(wcw.gt.wthr) then
          cwpot=vcw*wcw
        else
          cwpot=0
        end if
      else if(icase.eq.4) then
        vcw=cwpot1(rr,r,theta,icase)
        if(rr.lt.3.4d0) vcw=cwpot1(rmincw(1,icase),r,theta,icase)
c...keep v2 negative
        if(abs(vcw).lt.1.d-8) vcw=0
        wcw=0.125d0*(1.d0+tanh((3.0d0-r)/0.1d0))
     >            *(1.d0+tanh((r-0.7d0)/0.1d0))
     >            *(1.d0+tanh((rr-3.30d0)/0.1d0))
        if(wcw.gt.wthr) then
          cwpot=vcw*wcw
        else
          cwpot=0
        end if
      else if(icase.eq.5) then
        vcw=cwpot1(rr,r,theta,icase)
        if(abs(vcw).lt.1.d-8) then
          cwpot=0
        else
          w=0.5d0*(1.d0+tanh((min(r1,r2)-1.8d0)/0.20d0))
          rh2=max(1.d0,r)
          w1=0.5d0*(1.d0+tanh((rh2-2.7d0)/0.1d0))
          cwpot=vcw*w*(1.d0-w1)
        end if
      else if(icase.eq.6) then
        vcw=cwpot1(rr,r,theta,icase)
        if(abs(vcw).lt.1.d-8) then
          cwpot=0
        else
          w=0.5d0*(1.d0+tanh((rr-2.85d0)/0.15d0))
          rh2=max(1.d0,r)
          w1=0.5d0*(1.d0+tanh((rh2-2.7d0)/0.1d0))
          cwpot=vcw*w*(1.d0-w1)
        end if
      end if
c
      return
      end
c----------------------------------------------------------------------
      function cwpotex_sig(rr,rh2,theta,icase)
c----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common/cvwex/ rmincw(3,6),emax(6),dr(6)
c
c... sigma potential: only extrapolate in r
c
      if(rh2.lt.rmincw(2,icase)) then
        r1=rmincw(2,icase)
        r2=rmincw(2,icase)+dr(icase)
        e1=cwpot1(rr,r1,theta,icase)
        e2=cwpot1(rr,r2,theta,icase)
        if(e2.ge.e1) then
          write(6,10) icase,rr,theta
10        format(1x,'CW potential',i2,' decreasing at small r for',
     >              '  R=',f10.3,'  Theta=',f8.2,'  (fudging!)')
          e2=e1-1.4d-2
        end if
        b=(log(e1)-log(e2))/(r2-r1)
        a=e1/exp(-b*r1)
        cwpotex_sig=a*exp(-b*rh2)
        return
      else
        cwpotex_sig=cwpot1(rr,rh2,theta,icase)
      end if
      return
      end
c----------------------------------------------------------------------
      function cwpot1(rr,r,theta,icase)
c----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      parameter (maxb=15,maxpar=maxb*maxb*maxb)
      common/cdim/ kmax(6),lmax(6),nmax(6)
      common/csol/ s(maxpar,6)
      dimension fr(maxb),frr(maxb)
c
      nmx=nmax(icase)
      kmx=kmax(icase)
      lmx=lmax(icase)
      if(nmx.gt.maxb) then
        write(6,*) 'nmax.gt.maxb:',nmx,maxb
        stop
      end if
      if(kmx.gt.maxb) then
        write(6,*) 'kmax.gt.maxb:',kmx,maxb
        stop
      end if
      do n=1,nmx
        fr(n)=func_r(r,n,icase)
      end do
      do k=1,kmx
        frr(k)=func_rr(rr,k,icase)
      end do
      kk=0
      pot=0
      do l=1,lmx
        ft=func_theta(theta,l,icase)
        do n=1,nmx
          do k=1,kmx
            kk=kk+1
            pot=pot+s(kk,icase)*ft*fr(n)*frr(k)
          end do
        end do
      end do
      cwpot1=pot
      return
      end
c--------------------------------------------------------------------
      function func_theta(theta,l,icase)
c--------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common/cpar/ par_rr(10,6),par_r(10,6),par_theta(10,6)
c
      mm=nint(par_theta(1,icase))
      inc=nint(par_theta(2,icase))
      if(mm.eq.0) then
        ll=inc*(l-1)   !0,2,4...
      else
        ll=inc*l       !2,4,6...
      end if
c
      func_theta=pm1(ll,mm,theta)
      return
      end
c--------------------------------------------------------------------
      function func_r(r,n,icase)
c--------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common/cpar/ par_rr(10,6),par_r(10,6),par_theta(10,6)
c
      re=par_r(1,icase)
      func_r=(r-re)**(n-1)
      return
      end
c--------------------------------------------------------------------
      function func_RR(r,k,icase)
c--------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common/cpar/ par_rr(10,6),par_r(10,6),par_theta(10,6)
      data wthr/1.d-2/
c
      Re=par_rr(1,icase)
      rdmp=par_rr(2,icase)
      wdmp=par_rr(3,icase)
      kpmx=nint(par_rr(4,icase))
      kimin=nint(par_rr(5,icase))
c     rx=par_rr(6,icase)             ! check
c
      if(wdmp.eq.0) wdmp=1.d0
      w=0.5d0*(1.d0+tanh((r-rdmp)/wdmp))
c     if(w.lt.wthr) w=0
      if(k.eq.1) then
        func_rr=1.d0
         if(icase.gt.4) func_rr=w           ! A and B
      else if(k.le.kpmx) then
        func_rr=(1.d0-w)*(R-Re)**(k-1)
      else
        kk=k-kpmx-1+kimin
        if(icase.gt.4) then
c         rr=max(0.2d0,R-Rx)
          rr=max(0.2d0,R)
          func_rr=w*10.d0**kk/(rr)**kk
        else
          rr=max(0.2d0,R)
          func_rr=w*10.d0**kk/rr**kk
c         if(icase.gt.4) func_rr=w*10.d0**kk/(R)**kk  ! A and B
        end if
      end if
      return
      end
c--------------------------------------------------------------------
      function pm1(l,m,theta)
c--------------------------------------------------------------------
c
c  calculates value of legendre polynomial for l,m,theta
c
      implicit real*8(a-h,o-z)
      data pi180/.01745329251994329444d0/
c
      thedeg=theta*pi180
c
      if(m.gt.l) then
        pm1=0.d0
        return
      end if
      lmax=l
      x=cos(thedeg)
      if (m.ge.0) go to 1
      write (6,100)
100   format('  NEGATIVE M IN LEGENDRE ROUTINE:  ABORT')
      stop
c
1     if (m.gt.0) go to 5
c  here for regular legendre polynomials
      pm1=1.d0
      pm2=0.d0
      do 2 l=1,lmax
      pp=((2*l-1)*x*pm1-(l-1)*pm2)/dble(l)
      pm2=pm1
2     pm1=pp
      return
c
c  here for alexander-legendre polynomials
c
5     imax=2*m
      rat=1.d0
      do 6 i=2,imax,2
      ai=i
6     rat=rat*((ai-1.d0)/ai)
      y=sin(thedeg)
      pm1=sqrt(rat)*(y**m)
      pm2=0.d0
      low=m+1
      do 10 l=low,lmax
      al=(l+m)*(l-m)
      al=1.d0/al
      al2=((l+m-1)*(l-m-1))*al
      al=sqrt(al)
      al2=sqrt(al2)
      pp=(2*l-1)*x*pm1*al-pm2*al2
      pm2=pm1
10    pm1=pp
      return
      end
c--------------------------------------------------------------------
      function expol1(r,r1,r2,e1,e2)
c--------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      b=(log(e1)-log(e2))/(r2-r1)
      a=e1/exp(-b*r1)
      expol1=a*exp(-b*r)
      return
      end
c--------------------------------------------------------------------
      function cwpotex_pi(rr,r,theta,icase)
c--------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      common/cvwex/ rmincw(3,6),emax(6),dr(6)
      data rr0,r0/6.0d0,1.4d0/,wthr/1.d-4/
c...  exponential extrapolation along a line from (rr0,r0) to (rr,r)
c...  (used for pi potential)
      if(rr.lt.rmincw(1,icase).or.r.lt.rmincw(2,icase)) then
        b1=rr-rr0
        b2=r-r0
        if(b1.lt.0.and.b2.lt.0) then
          c1=(rmincw(1,icase)-rr0)
          c2=(rmincw(2,icase)-r0)
          rx=sqrt(b1*b1+b2*b2)
          ry=b1*b1/(c1*c1) + b2*b2/(c2*c2)
          a=sqrt(1.d0/ry)
          drr=dr(icase)/rx
          rr1=rr0+b1*a
          rr2=rr0+b1*(a-drr)
          r1=r0+b2*a
          r2=r0+b2*(a-drr)
          e1=cwpotexr(rr1,r1,theta,icase)
          e2=cwpotexr(rr2,r2,theta,icase)
          b=(log(e1)-log(e2))/drr
          cwpotex_pi=e1*exp(b*(1.0d0-a))
        else if(b1.lt.0) then
          rr1=rmincw(1,icase)
          e1=cwpotexr(rr1,r,theta,icase)
          e2=cwpotexr(rr1+dr(icase),r,theta,icase)
          b=(log(e1)-log(e2))/dr(icase)
          cwpotex_pi=e1*exp(b*(rr1-rr))
        else if(b2.lt.0) then
          r1=rmincw(2,icase)
          e1=cwpotexr(rr,r1,theta,icase)
          e2=cwpotexr(rr,r1+dr(icase),theta,icase)
          b=(log(e1)-log(e2))/dr(icase)
          cwpotex_pi=e1*exp(b*(r1-r))
        end if
      else
        cwpotex_pi=cwpotexr(rr,r,theta,icase)
      end if
      return
      end
c----------------------------------------------------------------------
      function cwpotexr(RR,r,theta,icase)
c----------------------------------------------------------------------
c
      implicit double precision (a-h,o-z)
      common/cvwex/ rmincw(3,6),emax(6),dr(6)
      data wthr/1.d-2/
c
c... extrapolate pi-potential to long r using bw
c
      wcw=0.5d0*(1.d0+tanh((2.7d0-r)/0.2d0))
      wbw=1.d0-wcw
      if(wbw.gt.wthr) then
        vbw25=bwpot(10.d0,2.5d0,10.d0)
        vcw25=cwpot1(rr,2.5d0,theta,icase)
        vbw=bwpot(10.d0,r,10.d0)-vbw25+vcw25
        if(wcw.gt.wthr) then
          vcw=cwpot1(rr,r,theta,icase)
          cwpotexr=wcw*vcw+wbw*vbw
        else
          cwpotexr=vbw
        end if
      else
        cwpotexr=cwpot1(rr,r,theta,icase)
      end if
      return
      end
c------------------------------------------------------------------------
      function bwpot(rr,r,theta)
c------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
c
c...  computes extrapolated bw potential for given Jacobi coordinates
c
      call distjac(r1,r2, rr,r,theta)
      bwpot=bwpotex(r1,r,r2)
      return
      end
c------------------------------------------------------------------------
      function bwpotex(r1,r2,r3)
c------------------------------------------------------------------------
      implicit double precision (a-h,o-z)
c
c... extrapolates the bw4 potential to short r1,r2,r3
c
      dimension ind(3),r(3)
      dimension e(4),el(4),a(4),b(16),indx(4),v(4)
      common/cvwex/ rmincw(3,6),emax(6),dr(6)
      save easm,ifirst
      data ifirst/1/
c
      if(ifirst.ne.0) then
        rh2=1.39d0
        rhcl=20.d0
        easm=1.d10
10      call bw4(rhcl,rh2,rhcl+rh2,va)
        if(va.lt.easm) then
          re=rh2
          easm=va
          rh2=rh2+0.0001d0
          if(rh2.lt.1.41d0) goto 10
        end if
        write(6,20) re,easm
20      format(' BW potential initialized. re=',f5.3,'  easm=',f11.8)
        ifirst=0
      end if
c
      if(r1.ge.rmincw(1,1).and.r2.ge.rmincw(2,1).and.r3.ge.
     >       rmincw(3,1)) then
        call bw4(r1,r2,r3,vp)
        bwpotex=min(emax(1),vp-easm)
        return
      end if
c... wall for very small r since extrapolation can go crazy
c     if(r1.lt.1.6d0.or.r2.lt.0.6d0.or.r3.lt.1.6d0) then
c       bwpotex=emax(1)
c       return
c     end if

      r(1)=r1
      r(2)=r2
      r(3)=r3
      ndim=0
      do i=1,3
        if(r(i).lt.rmincw(i,1)) then
          r(i)=rmincw(i,1)
          ndim=ndim+1
          ind(ndim)=i
        end if
      end do
      if(ndim.gt.0) then
         n=ndim+1
         call bw4(r(1),r(2),r(3),e(n))
         e(n)=e(n)-easm
         if(e(n).gt.emax(1)) then
c... do not extraplolate if energy is greater then emax
            bwpotex=emax(1)
            return
         end if
         do i=1,n
            b(i)=1.d0
            if(i.le.ndim) then
              rsav=r(ind(i))
              r(ind(i))=r(ind(i))+dr(1)
            end if
            do j=1,ndim
              b(i+j*n)=-r(ind(j))
            end do
            if(i.le.ndim) then
              call bw4(r(1),r(2),r(3),e(i))
              e(i)=e(i)-easm
              r(ind(i))=rsav
            end if
         end do
         ierr=0
         do i=1,n
           if(e(i).gt.e(n)) then
             write(6,30) r,ind(i),e(i),e(n)
30           format(' BW Extrapolation error at r1=',f10.3,
     >                '  r2=',f10.3,'  r3=',f10.3,
     >                '  displacement for r',i1,2f15.8,
     Y                '  decreasing energy')
             ierr=1
           end if
           if(e(i).le.0) then
             write(6,40) r,ind(i),e(i)
40           format(' BW Extrapolation error at r1=',f10.3,
     >                '  r2=',f10.3,'  r3=',f10.3,
     >                '  displacement for r',i1,f15.8,
     >                '  negative or zero energy')
             ierr=1
           end if
         end do
         if(ierr.ne.0) stop 'Extrapolation error'
         do i=1,n
           el(i)=log(e(i))
         end do
         call lineq(b,n, el,n, a,n, n,1, indx,v)
cstart debug
c;         ierr=0
c;         do i=1,n
c;            if(i.le.ndim) then
c;              rsav=r(ind(i))
c;              r(ind(i))=r(ind(i))+dr(1)
c;            end if
c;            v1=a(1)
c;            do j=1,ndim
c;              v1=v1-a(j+1)*r(ind(j))
c;            end do
c;            v1=exp(v1)
c;            if(abs(v1-e(i)).gt.1.d-5) then
c;              write(6,50) i,r,v1,e(i),easm
c;50            format(1x,'Fit error i=',i1,3f10.3,2f15.8,' easm=',f16.8)
c;              ierr=1
c;            end if
c;            if(i.le.ndim) r(ind(i))=rsav
c;         end do
c;         if(ierr.ne.0) stop 'Fit error'
cend
         r(1)=r1
         r(2)=r2
         r(3)=r3
         vp=a(1)
         do i=1,ndim
           vp=vp-a(i+1)*r(ind(i))
         end do
         vp=exp(vp)
cstart debug
c;         write(6,60) 'bwpotex r1,r2,r3:',r1,r2,r3,'  v=',vp
c;60       format(1x,a,3f8.3,a,f15.8)
cend
      else
         call bw4(r(1),r(2),r(3),vp)
         vp=vp-easm
      end if
c     if(r1.lt.1.6d0.or.r2.lt.0.6d0.or.r3.lt.1.6d0) then
c       vp=max(vp,emax(1))
c     end if
      bwpotex=min(emax(1),vp)
      return
      end
c------------------------------------------------------------------------
      subroutine bw4(x,y,z,v)
c-------------------------------------------------------------------
c
c     System:   ClH2
c     Name:     BW4
c     Author:  Wensheng Bian and Joachim Werner
c     Functional form: Aguado-Paniagua
c     Energy Zero Point: the asymptote Cl+H+H in a.u.
c
c     This subroutine calculates the potential energy for the
c     NON spin-orbit corrected MRCI+Q surface for the system
c     **Scale fact=.948d0**
c
c            Cl + H2
c
c     Input are the three distances x,y,z
c         Cl    H1    H2
c         |_____|
c            x
c               |_____|
c                  y
c         |___________|
c               z
c     This linear geometrie is at 180 Degree.
c
c
c-------------------------------------------------------------------

      implicit real*8(a-h,o-z)
      parameter(nmx=600,np=27,mma=20)
      common/cparm/ a(nmx),ma
      common/cint/ ix(nmx),iy(nmx),iz(nmx),mmax
      dimension xex(0:mma),xey(0:mma),xez(0:mma)
      dimension p(np)
      data ifirst/-1/
      data p/14.81794625, -0.05687046695,1.50963779,
     1       -19.91349307, 58.12148867,-75.88455892,
     1       36.47835698, 1.922975642, 0.7117342021,
     1       1.079946167,-0.02206944094,-7.109456997,
     1        36.79845478,-109.3716794,176.4925683,
     1        -120.4407534,2.351569228,1.082968302,
     1       14.81794625, -0.05687046695,1.50963779,
     1       -19.91349307, 58.12148867,-75.88455892,
     1       36.47835698, 1.922975642, 0.7117342021/

      save ifirst
c
c on first call of this subroutine, read in three-body parameter
c
      if (ifirst.eq.-1) then
         call inibw4
         ifirst=0
      end if
c
c.... Three-Body-Potential : Aguado-Paniagua
c
c.... initialize the non-linear parameters
c
      if(z.gt.x+y+1.d-10) then
        write(6,*) 'x=',x,'  y=',y,'  z=',z
        stop 'illegal distances'
      end if
      b1 = a(ma-2)
      b2 = a(ma-1)
      b3 = a(ma)
      fit = 0.0d0
      xexpon = b1*x
      yexpon = b2*y
      zexpon = b3*z
      exponx=dexp(-xexpon)
      expony=dexp(-yexpon)
      exponz=dexp(-zexpon)
      fex = x*exponx
      fey = y*expony
      fez = z*exponz
      xex(0)=1
      xey(0)=1
      xez(0)=1
      xex(1)=fex
      xey(1)=fey
      xez(1)=fez
      do m=2,mmax-1
	 xex(m)=xex(m-1)*fex
	 xey(m)=xey(m-1)*fey
	 xez(m)=xez(m-1)*fez
      enddo
c
      do 1010 i=1,ma-3
       fit=fit+xex(ix(i))*xey(iy(i))*xez(iz(i))*a(i)
1010  continue
c
c.... Two-Body-Potential : Aguado-Paniagua
c
c       c0      c1      c2      c3      c4     c5    c6     alpha  beta
c  ClH  p(1)    p(2)    p(3)    p(4)    p(5)   p(6)  p(7)   p(8)   p(9)
c  HH   p(10)   p(11)   p(12)   p(13)   p(14)  p(15) p(16)  p(17)  p(18)
c  ClH  p(19)   p(20)   p(21)   p(22)   p(23)  p(24) p(25)  p(26)  p(27)
c

      rhox=x*dexp(-p(9)*x)
      rhoy=y*dexp(-p(18)*y)
      rhoz=z*dexp(-p(27)*z)
      xval=p(1)*dexp(-p(8)*x)/x
      yval=p(10)*dexp(-p(17)*y)/y
      zval=p(19)*dexp(-p(26)*z)/z
      do 10 i=1,6
        xval=xval+p(i+1)*rhox**i
        yval=yval+p(i+10)*rhoy**i
        zval=zval+p(i+19)*rhoz**i
10    continue
c
c... Total Potential in atomic units
c
      v=fit+xval+yval+zval
c
c     if (x.lt.1.75d0.or.z.lt.1.75d0.or.y.lt.0.8d0) then
c        v=1.0d0
c     end if
      return
      end

c------------------------------------------------------------------------
      subroutine inibw4
c------------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      parameter(nmx=600)
      character*100 path
      common/cparm/ a(nmx),ma
      common/cint/ ix(nmx),iy(nmx),iz(nmx),mmax
      common/pathv/ path
      open(1,file=TRIM(ADJUSTL(path))//'three.param4',status='old')
c
      i=1
      rewind 1
c
10    read(1,*,end=100) nparm,ix(i),iy(i),iz(i),a(i)
      i=i+1
      goto 10
100   ma=i-1
      close(1)
      m=ix(ma-3)
      mmax=m+1
      return
      end
c----------------------------------------------------------------------
      subroutine distjac(r2,r3, rr,r,the)
c----------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      data pi/3.141592653589793d0/
      data pi180/.01745329251994329444d0/
c
c... returns bond distances r2,r3 for givem rr,r,theta. (r1=r)
c
      costhe=dcos(the*pi180)
      cospithe=dcos(pi-the*pi180)
      r2 = dsqrt((r/2.d0)**2+rr**2-2.d0*(r/2.d0)*rr*
     >            costhe)
      r3 = dsqrt((r/2.d0)**2+rr**2-2.d0*(r/2.d0)*rr*
     >            cospithe)
      return
      end
c----------------------------------------------------------------------
      subroutine jacobi(r1,r2,r3,rr,r,theta)
c----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      data pi180/57.29577951308232522583d0/
c
      r=r2
      RR=sqrt(0.5d0*(r3*r3-0.5d0*r2*r2+r1*r1))
      if(abs(r2*RR).lt.1.d-6) then              ! avoids division by zero
        theta=0
        return
      end if
      argcos=(-r1*r1+RR*RR+0.25d0*r2*r2)/(r2*RR)
      if(argcos.gt.1.d0) argcos=0.999999999999d0
      if(argcos.lt.-1.d0) argcos=-0.999999999999d0
      theta=acos(argcos)*pi180
      return
      end
c----------------------------------------------------------------------
      subroutine diag2(m,n,d,x)
c----------------------------------------------------------------------
      implicit double precision (a-h,o-z)
      parameter (maxdim=1000)
      parameter (eps=2.5d-16,dinf=2.3d-308,tol=dinf/eps)
c
c      computation of all eigenvalues and eigenvectors of a real
c      symmetric matrix by the method of qr transformations.
c      if the euclidean norm of the rows varies   s t r o n g l y
c      most accurate results may be obtained by permuting rows and
c      columns to give an arrangement with increasing norms of rows.
c
c      two machine constants must be adjusted appropriately,
c      eps = minimum of all x such that 1+x is greater than 1 on the
c            computer,
c      tol = inf / eps  with inf = minimum of all positive x represen-
c            table within the computer.
c      a dimension statement e(160) may also be changed appropriately.
c
c      input
c
c      (m)   not larger than 160,  corresponding value of the actual
c            dimension statement a(m,m), d(m), x(m,m),
c      (n)   not larger than (m), order of the matrix,
c      (a)   the matrix to be diagonalized, its lower triangle has to
c            be given as  ((a(i,j), j=1,i), i=1,n),
c.....
c.....the matrix #a# has been removed from the procedure
c.....the matrix #x# has to be put up by a predecessor routine c.....
c
c      output
c
c      (d)   components d(1), ..., d(n) hold the computed eigenvalues
c            in ascending sequence. the remaining components of (d) are
c            unchanged,
c      (x)   the computed eigenvector corresponding to the j-th eigen-
c            value is stored as column (x(i,j), i=1,n). the eigenvectors
c            are normalized and orthogonal to working accuracy. the
c            remaining entries of (x) are unchanged.
c
c      array (a) is unaltered. however, the actual parameters
c      corresponding to (a) and (x)  may be identical, ''overwriting''
c      the eigenvectors on (a).
c
c      leibniz-rechenzentrum, munich 1965
c
      dimension   d(m), x(m,m)
      dimension   e(maxdim)
c
c     correct adjustment for ibm 360/91 double precision
c
      if(m.gt.maxdim) then
        write(6,10) m,maxdim
10      format(' DIMENSION TOO LARGE IN DIAG2:',2i8)
        call fehler
      end if
c     call accnt('diag2',1)
c     eps=dlamch('e')
c     tol=dlamch('u')/eps
c
      if(n.eq.1) go to 400
      do 11 i=1,n
      d(i)=0
11    e(i)=0
c
c     householder's reduction
c
      do 150 i=n,2,-1
      l=i-2
      h=0.0d0
      g=x(i,i-1)
      if(l.le.0) goto 140
      do 30 k=1,l
   30 h=h+x(i,k)*x(i,k)
      s=h+g*g
      if(s.lt.tol) then
        h=0.0d0
      else if(h.gt.0) then
        l=l+1
        f=g
        g=dsqrt(s)
        if(f.gt.0) g=-g
        h=s-f*g
        x(i,i-1)=f-g
        f=0.0d0
c
        do 110 j=1,l
        x(j,i)=x(i,j)/h
        s=0.0d0
        do 80 k=1,j
   80   s=s+x(j,k)*x(i,k)
        j1=j+1
        if(j1.gt.l) go to 100
        do 90 k=j1,l
   90   s=s+x(k,j)*x(i,k)
  100   e(j)=s/h
  110   f=f+s*x(j,i)
c
        f=f/(2.d0*h)
c
        do 120 j=1,l
  120   e(j)=e(j)-f*x(i,j)
c
        do 130 j=1,l
        f=x(i,j)
        s=e(j)
        do 130 k=1,j
  130   x(j,k)=x(j,k)-f*e(k)-x(i,k)*s
c
      end if
  140 d(i)=h
  150 e(i-1)=g
c
c     accumulation of transformation matrices
c
  160 d(1)=x(1,1)
      x(1,1)=1.0d0
      do 220 i=2,n
      l=i-1
      if(d(i)) 200,200,170
  170 do 190 j=1,l
      s=0.0d0
      do 180 k=1,l
  180 s=s+x(i,k)*x(k,j)
      do 190 k=1,l
  190 x(k,j)=x(k,j)-s*x(k,i)
  200 d(i)=x(i,i)
      x(i,i)=1.0d0
  210 do 220 j=1,l
      x(i,j)=0.0d0
  220 x(j,i)=0.0d0
c
c     diagonalization of the tridiagonal matrix
c
      b=0.0
      f=0.0
      e(n)=0.0d0
c
      do 340 l=1,n
      h=eps*(dabs(d(l))+dabs(e(l)))
      if (h.gt.b) b=h
c
c     test for splitting
c
      do 240 j=l,n
      if (dabs(e(j)).le.b) goto 250
  240 continue
c
c     test for convergence
c
  250 if(j.eq.l) go to 340
c
c     shift from upper 2*2 minor
c
  260 p=(d(l+1)-d(l))*0.5d0
      r=dsqrt(p*p+e(l)*e(l))
      if(p.lt.0) then
        p=p+r
      else
        p=p-r
      end if
  290 h=d(l)+p
      do 300 i=l,n
  300 d(i)=d(i)-h
      f=f+h
c
c     qr transformation
c
      p=d(j)
      c=1.0d0
      s=0.0d0
c
c     simulation of loop do 330 i=j-1,l,(-1)
c
      j1=j-1
      do 330 i=j1,l,-1
      g=c*e(i)
      h=c*p
c
c     protection against underflow of exponents
c
      if (dabs(p).lt.dabs(e(i))) goto 310
      c=e(i)/p
      r=dsqrt(c*c+1.0d0)
      e(i+1)=s*p*r
      s=c/r
      c=1.0d0/r
      go to 320
  310 c=p/e(i)
      r=dsqrt(c*c+1.0d0)
      e(i+1)=s*e(i)*r
      s=1.0d0/r
      c=c/r
  320 p=c*d(i)-s*g
      d(i+1)=h+s*(c*g+s*d(i))
      do 330 k=1,n
      h=x(k,i+1)
      x(k,i+1)=x(k,i)*s+h*c
  330 x(k,i)=x(k,i)*c-h*s
c
      e(l)=s*p
      d(l)=c*p
      if (dabs(e(l)).gt.b) go to 260
c
c     convergence
c
  340 d(l)=d(l)+f
c
c     ordering of eigenvalues
c
      ni=n-1
  350 do 380i=1,ni
      k=i
      p=d(i)
      j1=i+1
      do 360j=j1,n
      if(d(j).ge.p) goto 360
      k=j
      p=d(j)
  360 continue
      if (k.eq.i) goto 380
      d(k) =d(i)
      d(i)=p
      do 370 j=1,n
      p=x(j,i)
      x(j,i)=x(j,k)
  370 x(j,k)=p
  380 continue
c
c     fixing of sign
c
      do 385 i=1,n
      pm=0
      do 386 j=1,n
      if(pm.gt.dabs(x(j,i))) goto 386
      pm =dabs(x(j,i))
      k=j
  386 continue
      if(x(k,i).ge.0) goto 385
      do 387 j=1,n
  387 x(j,i)=-x(j,i)
  385 continue
  390 go to 410
c
c     special treatment of case n = 1
c
  400 d(1)=x(1,1)
      x(1,1)=1.0d0
  410 continue
c     call accnt(' ',2)
      return
      end
c----------------------------------------------------------------------
      subroutine lineq(y,ny,b,nb,x,nx,n,m,indx,v)
c----------------------------------------------------------------------
      implicit real*8 (a-h,o-z)
      dimension y(ny,1),b(nb,1),x(nx,1)
      dimension indx(n),v(n)
c
c.....LU decomposition
c
      call ludcmp(y,n,ny,indx,v)
c
      do 50 j=1,m
      do 10 i=1,n
10    x(i,j)=b(i,j)
c
c.....now substitute and backsubstitute for each rhs
c
      call lubksb(y,n,ny,indx,x(1,j))
50    continue
      return
      end
      subroutine ludcmp(a,n,np,indx,v)
      implicit real*8 (a-h,o-z)
      dimension a(np,np),indx(n),v(n)
      data thr/1.d-15/
c
c.....obtain implicit scaling information
c
      do 20 i=1,n
      amax=0
        do 10 j=1,n
        if(abs(a(i,j)).gt.amax) amax=abs(a(i,j))
  10    continue
      if(amax.lt.thr) then
        write(6,*) 'amax',amax
        stop 'matrix singular'
      endif
      v(i)=1.d0/amax
  20  continue
c
c.....loop over colums
c
      do 100 j=1,n
c
c.....first part for i<j
c
        do 40 i=1,j-1
        sum=a(i,j)
          do 30 k=1,i-1
  30      sum=sum-a(i,k)*a(k,j)
  40    a(i,j)=sum
c
c.....second part for i>=j
c
        amax=0
        do 60 i=j,n
        sum=a(i,j)
          do 50 k=1,j-1
  50      sum=sum-a(i,k)*a(k,j)
        a(i,j)=sum
        dum=v(i)*abs(sum)
        if(dum.ge.amax) then
          amax=dum
          imax=i
        end if
  60    continue
c
c.....interchange rows if necessary
c
        if(j.ne.imax) then
          do 70 k=1,n
          dum=a(imax,k)
          a(imax,k)=a(j,k)
  70      a(j,k)=dum
          v(imax)=v(j)
        end if
        indx(j)=imax
c
c.....now divide by the pivot element
c
        if(j.ne.n) then
          if(abs(a(j,j)).lt.thr) then
           write(6,*) 'a(j,j)',a(j,j)
           stop 'matrix singular'
          endif
          dum=1.0/a(j,j)
            do 80 i=j+1,n
  80        a(i,j)=a(i,j)*dum
        end if
 100  continue
      if(abs(a(n,n)).lt.thr) then
       write(6,*) 'a(n,n)',a(n,n)
       stop 'matrix singular'
      endif
      return
      end
      subroutine lubksb(a,n,np,indx,b)
      implicit real*8 (a-h,o-z)
      dimension a(np,n),indx(n),b(n)
c
c.....ifirst will be the first nonvanishinng element of b
c.....forward substitution
c
      ifirst=0
      do 20 i=1,n
      ll=indx(i)
      sum=b(ll)
      b(ll)=b(i)
      if(ifirst.ne.0) then
        do 10 j=1,i-1
c       do 10 j=ii,i-1
  10    sum=sum-a(i,j)*b(j)
      else if(sum.ne.0) then
        ifirst=i
      end if
      b(i)=sum
  20  continue
c
c.....now do backsubstitution
c
      do 40 i=n,1,-1
      sum=b(i)
        do 30 j=i+1,n
  30    sum=sum-a(i,j)*b(j)
  40  b(i)=sum/a(i,i)
      return
      end
c----------------------------------------------------------------------
      subroutine fehler
      end
