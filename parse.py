"""
 *   BSD LICENSE
 *
 *   Copyright (c) Intel Corporation.
 *   All rights reserved.
 *
 *   Author: Anjaneya Chagam <anjaneya.chagam@intel.com>
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import sys
import os
import re
import dateutil.parser
import datetime
import dateutil.relativedelta
import numpy as np
import getopt
import string

from parse_utils import FuncEventTraceParser
from parse_utils import OIDEventParser
from parse_utils import FuncEventCephLogParser

def print_usage(name):
  print("Usage %s -t <func|oid|log> [-d <for detailed stats>] <files - trace or ceph log> " % name)

def main(name, argv):
  detailed_stats=0
  parse_type=None
  try:
    opts, args = getopt.getopt(argv,"t:d")
  except getopt.GetoptError:
    print_usage(name)
    sys.exit(2)
  for opt, val in opts:
    if opt == '-d':
      detailed_stats = 1
    elif opt == '-t':
      parse_type = val
          
  if not parse_type:
    print_usage(name)
    sys.exit(1)

  for i in range(0,len(args)):
    file=args[i]
    print("[%s]: Processing %s" % (name, file))
  
    if parse_type == 'func':
      parser = FuncEventTraceParser(file)
      parser.split_file_by_tid()
      parser.extract_per_thread_stack()
      parser.dump_stacks()
    elif parse_type == 'oid':
      parser = OIDEventParser(file)
      parser.parse_file()
      parser.compute_perf_counter_latency()
      parser.compute_and_dump_stats()
    elif parse_type == 'log':
      parser = FuncEventCephLogParser(file)
      parser.split_file()
      parser.extract_stack_by_thread()
      parser.dump_unique_stacks()

if __name__ == "__main__":
  main(sys.argv[0], sys.argv[1:])
