BIN = client server
GEN = who_clnt.c who_svc.c who.h
RPCCOM = rpcgen
CC = gcc
CFLAGS = -g -I/usr/include/tirpc
LDFLAGS = -ltirpc

all: $(GEN) $(BIN)

client: who_clnt.o
	$(CC) -o $@ who_clnt.o $(LDFLAGS)

server: who_svc.o
	$(CC) -o $@ who_svc.o $(LDFLAGS)

%.o: %.c who.h
	$(CC) $(CFLAGS) -c $< -o $@

$(GEN): who.x
	$(RPCCOM) who.x

clean:
	rm -f *.o $(BIN)

