# compiler to use
CC = gcc

# flags to pass compiler
CFLAGS = -Wall -O2

# name for executable
EXE = main

# space-separated list of source files
SRCS = conv.c main.c bottleneck.c utils.c qact.c quantizer.c pool.c fc.c

# automatically generated list of object files
OBJS = $(SRCS:.c=.o)

# default target
.PHONY: all
all: $(EXE)

$(EXE): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -ljpeg -lm  # linking with the math library

# dependencies
fc.o: fc.c fc.h
pool.o: pool.c pool.h qact.h
conv.o: conv.c conv.h qact.h
main.o: main.c conv.h bottleneck.h utils.h qact.h pool.h fc.h
bottleneck.o: bottleneck.c bottleneck.h conv.h qact.h pool.h fc.h
utils.o: utils.c utils.h
qact.o: qact.c qact.h quantizer.h conv.h
quantizer.o: quantizer.c quantizer.h

# debugging target
.PHONY: debug
debug: $(EXE)
	gdb $(EXE)

# housekeeping
.PHONY: clean
clean:
	rm -f core $(EXE) $(OBJS)