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
from datetime import timedelta
import numpy as np
import getopt
import string
import yaml
import threading


"""
Profile function execution time
"""
def timefunc(func):
  def func_wrapper(*args, **kwargs):
    b = datetime.datetime.now()
    ret = func(*args, **kwargs)
    e = datetime.datetime.now()
    secs = (e-b).total_seconds()
    print("%s took [%dm:%ds]" % (func.__name__, secs/60, secs%60))
    return ret
  return func_wrapper

"""
Wrapper for calling object function method
"""
class FuncThread(threading.Thread):
  def __init__(self, method, *args):
    self._method = method
    self._args = args
    threading.Thread.__init__(self)

  def run(self):
    self._method(*self._args)

"""
Common functions
"""
class Util(object):
  @staticmethod
  def get_usecs_elapsed(b_ts, e_ts):
    if isinstance(b_ts, str):
      d1 = dateutil.parser.parse(b_ts)
    else:
      d1 = b_ts
    if isinstance(e_ts, str):
      d2 = dateutil.parser.parse(e_ts)
    else:
      d2 = e_ts

    d3 = ((d2-d1).total_seconds())*1000000
    return d3

  

"""
FuncStack provides abstraction for call stack as well as ordered tree traversal
to consolidate duplicate sub stacks
"""
class FuncStack(object):
  def __init__(self, parent, level, func, file, line):
    self.parent = parent
    self.children = list()
    self.level = level
    self.func = func
    self.file = file
    self.line = line
    self.ts = list()

  def key(self):
    return "%s:%s" % (self.func, self.file)

  def insert(self, node):
    self.children.append(node)

  def add_enter_ts(self, ts):
    self.ts.append([ts])

  def add_exit_ts(self, ts):
    self.ts[-1].append(ts)

  def parent(self):
    return self.parent

  def dump(self):
    print("%d,%s,%s,%s,%s,[children:%d]" % (self.level, self.func, self.file, self.line, self.ts, len(self.children)))

  def is_leaf(self):
    if len(self.children) == 0:
      return True
    else:
      return False

  def traverse(self):
    self.dump()
    for n in self.children:
      n.traverse()

  def coompute_and_dump_stats(self, ofd, detail_stats):
    lat = list()
    for t in self.ts:
      s = Util.get_usecs_elapsed(t[0], t[1])
      lat.append(s)
    a = np.array(lat)
    sk = "%s%s:%s:%s:%s" % ('    '*self.level, 'ENTRY', self.func, self.file, self.line)
    if detail_stats:
        ofd.write("%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (sk, a.size, np.average(a), np.median(a),
                  np.percentile(a, 90), np.percentile(a, 95), np.percentile(a, 99), np.std(a), np.var(a)))
    else:
        ofd.write("%s,%s,%.2f\n" % (sk, a.size, np.average(a)))

  def dump_exit_marker(self, ofd, detail_stats):
    sk = "%s%s:%s:%s" % ('    '*self.level, 'EXIT', self.func, self.file)
    ofd.write("%s\n" % (sk))
    
  def dump_stats(self, ofd, detail_stats):
    self.coompute_and_dump_stats(ofd, detail_stats)
    for n in self.children:
      n.dump_stats(ofd, detail_stats)
    self.dump_exit_marker(ofd, detail_stats)
    
  def __eq__(self, right):
    if self.level == right.level and self.func == right.func and self.file == right.file:
      if len(self.children) == len(right.children):
        for i in xrange(len(self.children)):
          if self.children[i] == right.children[i]:
            continue
          else:
            return False
        return True
      else:
        return False
    else:
      return False
    
  def merge(self, y):
    self.ts.extend(y.ts)
    y.ts = list()
    for i in xrange(len(self.children)):
      self.children[i].merge(y.children[i])
    y.children = list()

  def trim(self):
    # pick non leaf nodes and traverse until you find leaf nodes to merge
    nlfnodes = [c for c in self.children if not c.is_leaf()]
    for nl in nlfnodes:
      nl.trim()

    for x in xrange(len(self.children)):
      for y in xrange(x+1, len(self.children)):
        n1 = self.children[x]
        n2 = self.children[y]
        if n1 == n2:
          n1.merge(n2)

    # get rid of all children who do not have any time stamps
    self.children = [c for c in self.children if len(c.ts) > 0]


class ParseError(RuntimeError):
 def __init__(self, arg):
    self.args = arg

"""
Parses one event
"""
class EventParser(object):
  def __init__(self, event_str):
    self.event_str = event_str
    self.attrs = dict()
    self.parse()
    if (self.attrs['_eventname'] == 'eventtrace:oid_event' or
        self.attrs['_eventname'] == 'eventtrace:oid_elapsed'):
      self.add_trimmed_context()

  def parse_section(self, str):
    entries = str.split(", ")
    for entry in entries:
      entry = entry.strip()
      r = re.match('(.*) = (.*)', entry)
      if r:
          k = r.group(1).strip('"')
          v = r.group(2).strip('"')
          self.attrs[k] = v

  def parse(self):
    section = self.event_str.split(": ")

    # header will have [timestamp] (elasped) node event:count
    header = section[0].split()
    if header and len(header) != 4:
      raise ParseError("Header '%s' parse error (need 4 tokens, found: %d instead in line '%s'"  %
          (header, len(header), self.event_str))
    else:
      #self.attrs['_ts'] = header[0].strip("[]")
      self.attrs['_ts'] = dateutil.parser.parse(header[0].strip("[]"))
      self.attrs['_elapsed'] = header[1].strip("()")
      self.attrs['_host'] = header[2]
      self.attrs['_eventname'] = header[3]

    # add log entries
    r = re.compile('{ (.*?) }')
    context = r.findall(section[1])
    for entry in context:
      entry = entry.strip()
      self.parse_section(entry)

    if self.attrs.get('elapsed'):
      self.attrs['elapsed'] = float(self.attrs['elapsed'])

  def add_trimmed_context(self):
      e2 = self.attrs['context'].split('!')
      if len(e2) == 5:
          self.attrs['trimmed_context'] = "%s!%s!%s!%s" % (e2[0], e2[1].split(':')[0], e2[2], e2[4])

  # expect caller to handle exceptions
  def __getattr__(self, item):
    return self.attrs[item]

  def dump(self):
    print("%s" % (self.attrs))

"""
Parse function trace events
"""
class FuncEventTraceParser(object):
  def __init__(self, file, detail_stats = 0):
    self.src_file = file
    self.stacks_per_tid_file = dict()
    self.detail_stats = detail_stats 
    self.long_hdr = "state:function:file:line,count,avg(usecs),med(usecs),90(usecs),95(usecs),99(usecs),std(usecs),var(usecs)\n"
    self.short_hdr = "state:function:file:line,count,avg(usecs)\n"

  def add_stack(self, stacks, s):
    for n in stacks:
      if n == s:
        n.merge(s)
        return
    stacks.append(s)

  def parse_thread_trace_file(self, file):
    print("computing stats for file '%s'" % (file))

    level=0
    root=None
    parent=None
    stack = list()
    stacks = list()
    with open(file, 'r') as f:
      for line in f:
        event = EventParser(line)
        if event._eventname == 'eventtrace:func_enter':
          node = FuncStack(parent, level, event.func, event.file, event.line)
          node.add_enter_ts(event._ts)
          if level == 0:
            root = node
          else:
            parent.insert(node)
          parent = node
          stack.append(node)
          level += 1
        elif event._eventname == 'eventtrace:func_exit':
          if level == 0: # means out of line marker, ignore
              continue
          level -= 1
          node = stack.pop()
          node.add_exit_ts(event._ts)
          parent = node.parent

          if level == 0:
            root.trim()
            self.add_stack(stacks, root)
            root=None
            parent=None
            stack=list()

    # this may need to be locked to be multi-threaded safe
    self.stacks_per_tid_file[file] = stacks
                    
  """
  create separate files for each thread - maintain sequence based on what is observed in source file
  """
  @timefunc
  def split_file_by_tid(self):
    threads={}
    sequence=0
    self.file_list={}
    with open(self.src_file, 'r') as f:
      for line in f:
        event = EventParser(line)
        if event.pthread_id and (event._eventname == 'eventtrace:func_enter' or 
           event._eventname == 'eventtrace:func_exit'):
            tid =  event.pthread_id
            if not threads.has_key(tid):
                sequence=sequence+1
                threads[tid] = sequence
            new_file='%s__%d_%s' % (self.src_file,threads[tid],tid)
            self.file_list[new_file] = ''
            with open(new_file, 'a') as n:
                n.write(line)

  @timefunc
  def extract_per_thread_stack(self):
    # for each file, do the indent
    threads = []
    for temp_file in self.file_list.keys():
      thr = FuncThread(self.parse_thread_trace_file, temp_file)
      thr.start()
      threads.append(thr)
    
    # wait for all threads to complete
    for t in threads:
      t.join()

    # remove temp files
    for temp_file in self.file_list.keys():
      os.remove(temp_file)

  def dump_stacks(self):
    stack_file="%s.perf.csv" % self.src_file
    sfd = open(stack_file, 'w')
    stacks = list()
    
    # create unique stacks
    for file in self.stacks_per_tid_file:
      for s in self.stacks_per_tid_file[file]:
        self.add_stack(stacks, s)
    
    # write header to file
    if self.detail_stats:
      sfd.write(self.long_hdr)
    else:
      sfd.write(self.short_hdr)

    # write each stack to file
    for stack in stacks:
      stack.dump_stats(sfd, self.detail_stats)

"""
Parse oid trace events
"""
class OIDEventParser(object):
  def __init__(self, file, counter_file = "counters.yaml", detail_stats = 0):
    self.src_file = file
    self.detail_stats = detail_stats 
    self.long_hdr = "counter,count,avg(usecs),med(usecs),90(usecs),95(usecs),99(usecs),std(usecs),var(usecs)\n"
    self.short_hdr = "counter,count,avg(usecs)\n"
    self.tag_perf = dict()
    self.oid_ts = dict()

    fd = open(counter_file)
    self.config = yaml.safe_load(fd)
    fd.close()
    if not self.config.get('derived_oids'):
      self.config['derived_oids'] = {}
    if not self.config.get('perf_counters'):
      print("'perf_counters' is empty in %s yaml file, aborting.." % (counter_file))
      sys.exit(2) 

    # this logic is bit tricky, forms two dictionaries out of self.ref_perf_counters
    # self.oid_tags contains tag and list of variations that need to be built with event attributes
    #   self.oid_tags={
    #     'RADOS_OP_COMPLETE': ['{oid}!{context}!{vpid}'],
    #     'RADOS_WRITE_OP_BEGIN': ['{oid}!{context}!{vpid}']
    #   }
    # self.perf_counters contains essentially counters and with begining and end tags with index id
    #   self.perf_counters ={
    #     'rados_write_e2e': ['RADOS_OP_COMPLETE!0', 'RADOS_WRITE_OP_BEGIN!0']
    #   }
    self.oid_tags = {}
    self.perf_counters = {}
    for c,v in self.config['perf_counters'].items():
      self.perf_counters[c] = list()
      if not v.get('key') or not v.get('begin') or not v.get('end'):
        print("[key,begin,end] attrbutes are mandatory for %s, aborting.." % (c))
        sys.exit(2)
      key = v.get('key')
      for tag in [v['begin'], v['end']]: 
        # e.g., k = 'RADOS_OP_COMPLETE:{oid}!{context}!{vpid}', tag = RADOS_OP_COMPLETE, key = {oid}!{context}!{vpid}
        if not self.oid_tags.get(tag):
          self.oid_tags[tag] = [key]
        else:
          self.oid_tags[tag].append(key)
        self.perf_counters[c].append("%s!%d" % (tag, self.oid_tags[tag].index(key))) 
       
  def add_tag_counter(self, event):
    tag = event.event
    if not self.tag_perf.get(tag):
      self.tag_perf[tag] = list()
    self.tag_perf[tag].append(event.elapsed)

  def add_oid_ts(self, event):
    tag = event.event
    if tag in self.oid_tags:
      for i in xrange(len(self.oid_tags[tag])):
        noid = self.oid_tags[tag][i].format(**event.attrs)
        key = "%s!%d" % (tag, i)
        if not self.oid_ts.get(key):
          self.oid_ts[key] = dict()
        # key is like RADOS_OP_COMPLETE!0 or OP_APPLIED_BEGIN!1
        if not self.oid_ts[key].get(noid):
          self.oid_ts[key][noid] = list()
        self.oid_ts[key][noid].append(event._ts)
    else:
      pass
      
  @timefunc
  def parse_file(self):
    with open(self.src_file, 'r') as f:
      # oid_ts will have 
      #   tag![#]: dict (oid, list (__ts))
      # example:
      #   RADOS_READ_OP_BEGIN!0: 
      #          benchmark_data_reddy1_26045_object0!26045: (14:11:07.638468674) 
      for line in f:
        event = EventParser(line)
        if (event._eventname == 'eventtrace:oid_event' or 
          event._eventname == 'eventtrace:oid_elapsed'):
          tag = event.event
          self.add_oid_ts(event)
          if tag in self.config['derived_oids']: 
            self.add_tag_counter(event)
            dtag = self.config['derived_oids'][tag]
            bts = event._ts - timedelta(microseconds=event.elapsed)
            event.event = dtag
            event._ts = bts
            self.add_oid_ts(event)
          elif event._eventname == 'eventtrace:oid_elapsed':
            self.add_tag_counter(event)

  @timefunc
  def compute_perf_counter_latency(self):
      # aggregate oid performance for derived perf counters
      # we will use the tag_perf for ease of use
      # dump dict
      for ctr in self.perf_counters:
        ekey = self.perf_counters[ctr][0]
        bkey = self.perf_counters[ctr][1]
        if bkey in self.oid_ts and ekey in self.oid_ts:
          edict = self.oid_ts[ekey]
          sdict = self.oid_ts[bkey]
          for k in edict:
            if k in sdict:
              if len(edict[k]) == len(sdict[k]):
                if not self.tag_perf.get(ctr):
                  self.tag_perf[ctr] = list()
                for i in xrange(len(edict[k])):
                  self.tag_perf[ctr].append(Util.get_usecs_elapsed(sdict[k][i], edict[k][i]))
              else:
                  print("parse error k=%s length mismatch in edict:%s, sdict:%s" % (k, edict, sdict))
                  continue
            else:
                print("key %s not found" % (k))
        else:
            pass

  @timefunc
  def compute_and_dump_stats(self):
    # compute summary stats and write to file
    oid_stat_file="%s.oidperf.csv" % self.src_file
    print("storing stats in %s" % (oid_stat_file))
    fd = open(oid_stat_file, 'w')
    if self.detail_stats:
      fd.write(self.long_hdr)
    else:
      fd.write(self.short_hdr)
    for ctr in sorted(self.tag_perf):
      a = np.array(self.tag_perf.get(ctr))
      if self.detail_stats:
        fd.write("%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (ctr, a.size, np.average(a), np.median(a),
                  np.percentile(a, 90), np.percentile(a, 95), np.percentile(a, 99), np.std(a), np.var(a)))
      else:
        fd.write("%s,%s,%.2f\n" % (ctr, a.size, np.average(a)))

"""
This is a trivial implementation - doesn't support nested calls unlike function tracer class
"""
class FuncEventCephLogParser(object):
  def __init__(self, file, dump = 0):
    self.log_file = file
    self.stacks = list()
    self.file_list={}
    self.dump = dump

  @staticmethod
  def pretty_print(fd, tab, str):
    fd.write("%s%s" % ('    '*tab, str)),

  def merge_stack(self, stack):
    # stacks = [(list),(list),..]
    # stack = (list)
    for k in self.stacks:
      if not cmp(k, stack):
        return
    self.stacks.append(stack)
          
  @staticmethod
  def parse_line(line):
    d = dict()
    section = line.split(" ENTRY ")
    if (len(section) != 2):
      section = line.split(" EXIT ")
      if (len(section) != 2):
        return None
      else:
        d['mode'] = "EXIT"
    else:
      d['mode'] = "ENTRY"

    s1 = section[0].split()
    if (len(s1) >= 2):
      d['tid'] = s1[len(s1)-2] # pick the last but one for tid (brute force parsing)
    else:
      return None

    if d['mode'] == "ENTRY":
      m = re.match('\((.*)\)\s+(.*):(.*)', section[1])
      if m:
        d['func'] = m.group(1)
        d['file'] = m.group(2)
        d['line'] = m.group(3)
      else:
        return None
    else:
      m = re.match('\((.*)\)\s+(.*)', section[1])
      if m:
        d['func'] = m.group(1)
        d['file'] = m.group(2)
      else:
        return None

    return d

  def format_file_terse(self, file):
    tabs=0
    formatted_file='%s_terse_formatted' % file
    if self.dump:
      out_fd = open(formatted_file, 'a')
      print("writing to file %s" % (formatted_file))
    stack=list()
    with open(file, 'r') as f:
      for line in f:
        d = FuncEventCephLogParser.parse_line(line)
        if d:
          if (d['mode'] == "ENTRY"):
            row = "%s:%s:%s:%s" % (d['mode'], d['func'], d['file'], d['line'])
            stack.append("%s%s" % ('    '*tabs, row))
            if self.dump:
              FuncEventCephLogParser.pretty_print(out_fd, tabs, "%s\n" % (row))
            tabs += 1
          else:
            if tabs == 0: # means out of line mode, ignore
              continue
            tabs -= 1
            row = "%s:%s:%s" % (d['mode'], d['func'], d['file'])
            stack.append("%s%s" % ('    '*tabs, row))
            if self.dump:
              FuncEventCephLogParser.pretty_print(out_fd, tabs, "%s\n" % (row))
            if tabs == 0:
              self.merge_stack(stack)
              stack=list()
        else:
          stack.append("%s%s" % ('    '*tabs, line))
          if self.dump:
            FuncEventCephLogParser.pretty_print(out_fd, tabs, line)

  """
  create separate files for each thread - maintain sequence based on what is observed in source file
  """
  def split_file(self):
    threads={}
    sequence=0
    with open(self.log_file, 'r') as f:
      for line in f:
        d = FuncEventCephLogParser.parse_line(line)
        if d:
          tid = d.get('tid')
          if not threads.has_key(tid):
            sequence=sequence+1
            threads[tid] = sequence
          new_file='%s.%d.%s' % (self.log_file, threads[tid], tid)
          self.file_list[new_file] = ''
          with open(new_file, 'a') as n:
            n.write(line)

  def extract_stack_by_thread(self):
    # for each file, do the indent
    for temp_file in self.file_list.keys():
      self.format_file_terse(temp_file)
      os.remove(temp_file)

  def dump_unique_stacks(self):
    # print unique stack traces
    stack_file="%s.stacks" % self.log_file
    print("storing summary of operations in %s\n" % (stack_file))
    sfd = open(stack_file, 'a') 
    for k in self.stacks:
      for e in k:
        sfd.write("%s\n" % e)
