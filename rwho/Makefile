# Makefile

# 컴파일러 설정
CC = gcc
CFLAGS = -g -Wall

# 타겟 실행 파일 이름
CLIENT = who_client
SERVER = who_server

# 소스 파일들
CLIENT_SRCS = who_clnt.c who_xdr.c
SERVER_SRCS = who_svc.c who_xdr.c who.c

# 객체 파일들
CLIENT_OBJS = $(CLIENT_SRCS:.c=.o)
SERVER_OBJS = $(SERVER_SRCS:.c=.o)

# 헤더 파일
INCLUDES = who.h

# 기본 타겟
all: $(CLIENT) $(SERVER)

# rpcgen을 사용하여 소스 파일 생성
who.h who_clnt.c who_svc.c who_xdr.c: who.x
    rpcgen who.x

# 클라이언트 컴파일
$(CLIENT): $(CLIENT_OBJS)
    $(CC) $(CFLAGS) -o $(CLIENT) $(CLIENT_OBJS)

# 서버 컴파일
$(SERVER): $(SERVER_OBJS)
    $(CC) $(CFLAGS) -o $(SERVER) $(SERVER_OBJS)

# 객체 파일 컴파일 규칙
%.o: %.c $(INCLUDES)
    $(CC) $(CFLAGS) -c $<

# 정리
clean:
    rm -f $(CLIENT) $(SERVER) *.o who.h who_clnt.c who_svc.c who_xdr.c