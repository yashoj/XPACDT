FC = ifort
FPC = f2py
CMPFLG = -fast -llapack -fPIC
FPFLG = --build-dir build --fcompiler=intelem
LNKFLG = 

OBJS = bkmp2.o
pot : $(OBJS)
	$(FPC) $(FPFLG) -c pot.f90 $(OBJS) -m pot

bkmp2.o :
	$(FC) $(CMPFLG) -c bkmp2.f
