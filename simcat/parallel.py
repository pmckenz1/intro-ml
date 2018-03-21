#!/usr/bin/env python

"""
Handle automatic launching of ipyparallel engines for CLI
"""

from __future__ import print_function
import socket
import sys



# def get_client(*args, **kwargs):
#     """
#     Find and connect to a running ipcluster instance
#     """

#     ## 
#     cstring = "establishing parallel connection"

#     ## wrap search 
#     try:
#         if profile not in [None, "default"]:
#             args = {"profile": profile, "timeout": timeout}
#         else:
#             clusterargs = [cluster_id, profile, timeout]


#         ## get connection within timeout window 
#         ipyclient = ipp.Client()



def cluster_info(ipyclient):
    hosts = []
    for eid in ipyclient.ids:
        engine = ipyclient[eid]
        if not engine.outstanding:
            hosts.append(engine.apply(socket.gethostname))

    hosts = [i.get() for i in hosts]
    result = []
    for hostname in set(hosts):
        result.append(
            "host compute node: [{} cores] on {}".format(
                hosts.count(hostname), hostname))
    print("\n".join(result), file=sys.stderr)