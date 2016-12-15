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
