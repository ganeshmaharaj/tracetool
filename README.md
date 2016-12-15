# tracetool
This repo contains Ceph LTTng trace output parser utilities for latency tracing for functions and oid events.
Scripts use following LTTng trace events for latency analysis:
  eventtrace:func_enter
  eventtrace:func_exit
  eventtrace:oid_elapsed
  eventtrace:oid_event

PRE-REQUISITES
--------------

1. You will need the following pull request that has support for above event traces for key functions:
   https://github.com/ceph/ceph/pull/12330

2. By default, above events are not enabled with LTTng tracing due to concerns over performance implications.
   So you will need to build Ceph with -DWITH_LTTNG=ON -DWITH_EVENTTRACE=ON 

   i.e. <ceph source top dir>/do_cmake.sh -DWITH_LTTNG=ON -DWITH_EVENTTRACE=ON or 
        <ceph source top dir>/do_cmake.sh -DWITH_LTTNG=ON -DWITH_EVENTTRACE=ON -DWITH_FIO=ON -DFIO_INCLUDE_DIR=/root/fio/

   Refer to Ceph buuild instructions for additional details

SETUP
-----
1. Install python and pip packages

2. Install dateutil, numpy and PyYAML python libraries. See below for instructions:

    -Install dateutil

        • Download the latest dateutil: https://pypi.python.org/pypi/python-dateutil/2.5.3

        • Unpack/install

        • python setup.py install

    -pip install numpy
    -pip install PyYAML

RUN
---

1. Start Ceph processes with LTTng LD_PRELOAD option. Example:

   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/liblttng-ust-fork.so ceph-osd --cluster=ceph -i 0 --conf /etc/ceph/ceph.conf

2. Start LTTng session with right context params. Scripts expect pthread_id, vpid, procname fields in the traces. 

    lttng enable-channel --userspace --num-subbuf 16 --subbuf-size 16M big-channel

    lttng add-context -u --channel=big-channel -t pthread_id

    lttng add-context -u --channel=big-channel -t vpid

    lttng add-context -u --channel=big-channel -t procname

    lttng enable-event --userspace --channel=big-channel eventtrace:*

    lttng start <run id>

3. Run your favorite benchmarking tool (rados, fio etc.). This needs to have LTTng turned on as well, example:

   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/liblttng-ust-fork.so rados bench -p $POOL -t $qdepth $RADOS_RUN_TIME rand --run-name $run_name

4. Stop and dump LTTng traces

   lttng stop <run id>

   lttng view <run id> > foo.trace

   lttng disable-channel  --userspace big-channel

   lttng destroy <run id>

5. Run script to collect latency stats

   a. Function tracing. Trace output in <file>.perf.csv

      python parse.py -t func <file list>  
      
   b. OID event tracing. Trace output in <file>.oidperf.csv

      python parse.py -t oid <file list>

   c. Extract stack from ceph log. Stack traces in <file>.stacks

      python parse.py -t log <Ceph log file>

TIPS
----

1. Ceph repo has traces only for key functions. You can instrument any function to get visibility into how the flow works. 
   Use this to understand Ceph code flows very quikcly. You can use very simple script (shown below) to add traces:

    dir=/root/cepht1
    src_dir=$dir/src

    ################## DO NOT CHANGE BELOW THIS LINE ###############

    function patch_trace_calls {
        file=$1
        echo "Processing $file"
        sed -i '/#define.*dout_subsys/a \
    #include "common/EventTrace.h"' $file
        sed -i 's#^{$#{\n  FUNCTRACE\(\);#' $file
    }

    for file in $(ls -1 ~/cepht1/src/os/bluestore/*.cc); do
        patch_trace_calls $file
    done
    
    NOTE: You will see compilation errors when function entry i.e. { is not on its own line or conflicts with keyworkds like 
      namespace that above script will not recognize. You are expected to manually fix them. Here is how instrumented fucntion
      looks:
    void BlueFS::wait_for_aio(FileWriter *h)
    {
      FUNCTRACE();
      ...
    }

2. Input for OID event tracing comes from counters.yaml file. You can add arbitrary OID events in the source code - needs 
   oid name (or any arbitrray string) and an event id (arbitrary string).

   OID_EVENT_TRACE(oid.name.c_str(), "RADOS_READ_OP_BEGIN");

   OID_EVENT_TRACE(oid.name.c_str(), "RADOS_OP_COMPLETE");

   Associated entry in yaml shown below which tracks end to end latency of rados read operation:
   rados_read_e2e:
    key: '{oid}!{context}!{vpid}'
    begin: RADOS_OP_COMPLETE
    end: RADOS_READ_OP_BEGIN

