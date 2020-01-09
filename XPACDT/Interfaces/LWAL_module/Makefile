FC = ifort
FPC = f2py
CMPFLG = -fast -i8 -llapack -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -fPIC
FPFLG = --build-dir build --fcompiler=intelem -llapack
LNKFLG = 

OBJS = lwal.o
pot : $(OBJS)
	$(FPC) $(FPFLG) -c pot.f90 $(OBJS) -m pot

lwal.o :
	$(FC) $(CMPFLG) -c lwal.f
