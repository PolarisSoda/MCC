/*
 * Please do not edit this file.
 * It was generated using rpcgen.
 */

#include <memory.h> /* for memset */
#include "who.h"

/* Default timeout can be changed using clnt_control() */
static struct timeval TIMEOUT = { 25, 0 };

char **
get_logged_users_1(void *argp, CLIENT *clnt)
{
	static char *clnt_res;

	memset((char *)&clnt_res, 0, sizeof(clnt_res));
	if (clnt_call (clnt, GET_LOGGED_USERS,
		(xdrproc_t) xdr_void, (caddr_t) argp,
		(xdrproc_t) xdr_wrapstring, (caddr_t) &clnt_res,
		TIMEOUT) != RPC_SUCCESS) {
		return (NULL);
	}
	return (&clnt_res);
}

int main(int argc, char *argv[]) {
    CLIENT *clnt;
    char **result;
    char *server;

    if (argc < 2) {
        fprintf(stderr, "사용법: %s 서버호스트명\n", argv[0]);
        exit(1);
    }

    server = argv[1];

    // 클라이언트 핸들 생성
    clnt = clnt_create(server, WHOPROG, WHOVERS, "tcp");
    if (clnt == NULL) {
        clnt_pcreateerror(server);
        exit(1);
    }

    // 서버의 RPC 함수 호출
    result = get_logged__users_1(NULL, clnt);
    if (result == NULL) {
        clnt_perror(clnt, "원격 호출 실패");
        exit(1);
    }

    // 결과 출력
    printf("현재 로그인한 사용자:\n%s", *result);

    // 클라이언트 핸들 해제
    clnt_destroy(clnt);

    return 0;
}
