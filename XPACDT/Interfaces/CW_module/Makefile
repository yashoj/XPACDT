FC = ifort
FPC = f2py
CMPFLG = -fast -llapack -fPIC
FPFLG = --build-dir build --fcompiler=intelem
LNKFLG = 

OBJS = poth2cl.o
pot : $(OBJS)
	$(FPC) $(FPFLG) -c pot.f90 $(OBJS) -m pot

poth2cl.o :
	$(FC) $(CMPFLG) -c poth2cl.f
