/*
 * Please do not edit this file.
 * It was generated using rpcgen.
 */

#ifndef _WHO_H_RPCGEN
#define _WHO_H_RPCGEN

#include <rpc/rpc.h>


#ifdef __cplusplus
extern "C" {
#endif


#define WHO_PROG 22855
#define WHO_VERS 1

#if defined(__STDC__) || defined(__cplusplus)
#define GET_LOGGED_INT_USERS 1
extern  char ** get_logged_int_users_1(void *, CLIENT *);
extern  char ** get_logged_int_users_1_svc(void *, struct svc_req *);
extern int who_prog_1_freeresult (SVCXPRT *, xdrproc_t, caddr_t);

#else /* K&R C */
#define GET_LOGGED_INT_USERS 1
extern  char ** get_logged_int_users_1();
extern  char ** get_logged_int_users_1_svc();
extern int who_prog_1_freeresult ();
#endif /* K&R C */

#ifdef __cplusplus
}
#endif

#endif /* !_WHO_H_RPCGEN */
