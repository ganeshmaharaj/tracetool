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
from parse import *
from sys import stdout
try:
  from queue import Queue
  import babeltrace
except:
  from Queue import Queue


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
    if isinstance(b_ts, str) and isinstance(e_ts, str):
      d1 = dateutil.parser.parse(b_ts)
      d2 = dateutil.parser.parse(e_ts)
      return ((d2-d1).total_seconds())*1000000
    elif ((isinstance(b_ts, int) or isinstance(b_ts, float)) and 
          (isinstance(e_ts, int) or isinstance(e_ts, float))): #nanoseconds since Epoch   
      return (e_ts-b_ts)/1000
    else: 
      return ((e_ts-b_ts).total_seconds())*1000000

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
        for i in range(len(self.children)):
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
    for i in range(len(self.children)):
      self.children[i].merge(y.children[i])
    y.children = list()

  def trim(self):
    # pick non leaf nodes and traverse until you find leaf nodes to merge
    nlfnodes = [c for c in self.children if not c.is_leaf()]
    for nl in nlfnodes:
      nl.trim()

    for x in range(len(self.children)):
      for y in range(x+1, len(self.children)):
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
    base_fmt='[{ts}]{}eventtrace:{id}: { cpu_id = {cpu_id} }, { pthread_id = {pthread_id}, ' \
      'vpid = {vpid}, procname = "{procname}" }, '
    self.enter_fmt = base_fmt + '{ file = "{file}", func = "{func}", line = {line} }'
    self.exit_fmt = base_fmt + '{ file = "{file}", func = "{func}" }'
    self.oid_fmt = base_fmt + '{ oid = {oid}, event = {event}, ' \
          'context = {context}, file = "{file}", func = {func}, line = {line} }'
    self.elapsed_fmt = base_fmt + '{ oid = {oid}, event = {event}, ' \
          'context = {context}, elapsed = {elapsed}, file = "{file}", func = "{func}", line = {line} }'

  def parse_short(self):
    res = parse('{}eventtrace:{id}: { cpu_id ={}pthread_id = {pthread_id}, vpid = {vpid}, procname {}', self.event_str)
    if res:
      self.attrs.update(res.named)
    return res

  def parse(self):
    # get event and do the parsing based on event type
    res = parse('{}eventtrace:{id}: { cpu_id{}', self.event_str)

    if res:
      self.attrs.update(res.named)
      if self.attrs['id'] == 'func_enter':
        res = parse(self.enter_fmt, self.event_str)
      elif self.attrs['id'] == 'func_exit':
        res = parse(self.exit_fmt, self.event_str)
      elif self.attrs['id'] == 'oid_event':
        res = parse(self.oid_fmt, self.event_str)
      elif self.attrs['id'] == 'oid_elapsed':
        res = parse(self.elapsed_fmt, self.event_str)
      else:
        return None
      if res:
        self.attrs.update(res.named)
        if self.attrs['id'] == 'oid_event' or self.attrs['id'] == 'oid_elapsed':
          self.add_trimmed_context()
        self.attrs['ts'] = dateutil.parser.parse(self.attrs['ts'])
        if self.attrs.get('elapsed'):
          self.attrs['elapsed'] = float(self.attrs['elapsed'])
        for attr in ['oid', 'event', 'context']:
          if self.attrs.get(attr):
            self.attrs[attr] = self.attrs[attr].replace('"', '')

    return res

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
        if event.parse():
          if event.id == 'func_enter':
            node = FuncStack(parent, level, event.func, event.file, event.line)
            node.add_enter_ts(event.ts)
            if level == 0:
              root = node
            else:
              parent.insert(node)
            parent = node
            stack.append(node)
            level += 1
          elif event.id == 'func_exit':
            if level == 0: # means out of line marker, ignore
                continue
            level -= 1
            node = stack.pop()
            node.add_exit_ts(event.ts)
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
        if event.parse_short() and (event.id == 'func_enter' or event.id == 'func_exit'):
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
      for i in range(len(self.oid_tags[tag])):
        noid = self.oid_tags[tag][i].format(**event.attrs)
        key = "%s!%d" % (tag, i)
        if not self.oid_ts.get(key):
          self.oid_ts[key] = dict()
        # key is like RADOS_OP_COMPLETE!0 or OP_APPLIED_BEGIN!1
        if not self.oid_ts[key].get(noid):
          self.oid_ts[key][noid] = list()
        self.oid_ts[key][noid].append(event.ts)
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
        if event.parse() and (event.id == 'oid_event' or event.id == 'oid_elapsed'):
          tag = event.event
          self.add_oid_ts(event)
          if tag in self.config['derived_oids']: 
            self.add_tag_counter(event)
            dtag = self.config['derived_oids'][tag]
            bts = event.ts - timedelta(microseconds=event.elapsed)
            event.event = dtag
            event.ts = bts
            self.add_oid_ts(event)
          elif event.id == 'oid_elapsed':
            self.add_tag_counter(event)

  @timefunc
  def compute_perf_counter_latency(self):
      # aggregate oid performance for derived perf counters
      # we will use the tag_perf for ease of use
      # dump dict
      for ctr in self.perf_counters:
        bkey = self.perf_counters[ctr][0]
        ekey = self.perf_counters[ctr][1]
        if bkey in self.oid_ts and ekey in self.oid_ts:
          edict = self.oid_ts[ekey]
          sdict = self.oid_ts[bkey]
          for k in edict:
            if k in sdict:
              if len(edict[k]) == len(sdict[k]):
                if not self.tag_perf.get(ctr):
                  self.tag_perf[ctr] = list()
                for i in range(len(edict[k])):
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

"""
Parse function trace events
"""
class FuncEventData(object):
  def __init__(self, event):
    self.id = event.name
    self.ts = event.timestamp
    self.func = event.field_with_scope('func', babeltrace.common.CTFScope.EVENT_FIELDS)
    self.file = event.field_with_scope('file', babeltrace.common.CTFScope.EVENT_FIELDS)
    self.line = event.field_with_scope('line', babeltrace.common.CTFScope.EVENT_FIELDS)

class OIDEventData(object):
  def __init__(self, event):
    self.id = event.name
    self.vpid = event['vpid']
    self.pthread_id = event['pthread_id']
    self.procname = event['procname']
    self.cpu_id = event['cpu_id']
    self.ts = event.timestamp
    self.func = event.field_with_scope('func', babeltrace.common.CTFScope.EVENT_FIELDS)
    self.file = event.field_with_scope('file', babeltrace.common.CTFScope.EVENT_FIELDS)
    self.line = event.field_with_scope('line', babeltrace.common.CTFScope.EVENT_FIELDS)
    self.oid = event.field_with_scope('oid', babeltrace.common.CTFScope.EVENT_FIELDS)
    self.event = event.field_with_scope('event', babeltrace.common.CTFScope.EVENT_FIELDS)
    self.context = event.field_with_scope('context', babeltrace.common.CTFScope.EVENT_FIELDS)
    self.elapsed = event.field_with_scope('elapsed', babeltrace.common.CTFScope.EVENT_FIELDS)
    e2 = self.context.split('!')
    if len(e2) == 5:
        self.trimmed_context = "%s!%s!%s!%s" % (e2[0], e2[1].split(':')[0], e2[2], e2[4])

class FuncEventStackBuilder(threading.Thread):

  def __init__(self, event_q = None):
    self.event_q = event_q
    self.stacks = list()
    self.long_hdr = "state:function:file:line,count,avg(usecs),med(usecs),90(usecs),95(usecs),99(usecs),std(usecs),var(usecs)\n"
    self.short_hdr = "state:function:file:line,count,avg(usecs)\n"
    threading.Thread.__init__(self)

  def add_stack(self, s):
    for n in self.stacks:
      if n == s:
        n.merge(s)
        return
    self.stacks.append(s)

  def run(self):
    level=0
    root=None
    parent=None
    stack = list()
    while True:
      el = self.event_q.get()
      if el is None:
        break
      if el.id == 'eventtrace:func_enter':
        node = FuncStack(parent, level, el.func, el.file, el.line)
        node.add_enter_ts(el.ts)
        if level == 0:
          root = node
        else:
          parent.insert(node)
        parent = node
        stack.append(node)
        level += 1
      elif el.id == 'eventtrace:func_exit':
        if level == 0: # means out of line marker, ignore
            continue
        level -= 1
        node = stack.pop()
        node.add_exit_ts(el.ts)
        parent = node.parent

        if level == 0:
          root.trim()
          self.add_stack(root)
          root=None
          parent=None
          stack=list()

  def dump(self, out_file, detail_stats = 0):
    # write uniquue stack traces and latency into output file
    sfd = open(out_file, 'w')
    # write header to file
    if detail_stats:
      sfd.write(self.long_hdr)
    else:
      sfd.write(self.short_hdr)

    # write each stack to file
    for stack in self.stacks:
      stack.dump_stats(sfd, detail_stats)

    sfd.close()
    
"""
Parse oid trace events
"""
class OIDEventBuilder(threading.Thread):
  def __init__(self, event_q, counter_file = "counters.yaml"):
    self.event_q = event_q
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

    threading.Thread.__init__(self)
       
  def add_tag_counter(self, event):
    tag = event.event
    if not self.tag_perf.get(tag):
      self.tag_perf[tag] = list()
    self.tag_perf[tag].append(event.elapsed)

  def add_oid_ts(self, event):
    tag = event.event
    if tag in self.oid_tags:
      for i in range(len(self.oid_tags[tag])):
        noid = self.oid_tags[tag][i].format(**event.__dict__)
        key = "%s!%d" % (tag, i)
        if not self.oid_ts.get(key):
          self.oid_ts[key] = dict()
        # key is like RADOS_OP_COMPLETE!0 or OP_APPLIED_BEGIN!1
        if not self.oid_ts[key].get(noid):
          self.oid_ts[key][noid] = list()
        self.oid_ts[key][noid].append(event.ts)
    else:
      pass
      
  @timefunc
  def run(self):
    while True:
      # oid_ts will have 
      #   tag![#]: dict (oid, list (__ts))
      # example:
      #   RADOS_READ_OP_BEGIN!0: 
      #          benchmark_data_reddy1_26045_object0!26045: (14:11:07.638468674) 
      event = self.event_q.get()
      if event is None:
        break
      tag = event.event
      self.add_oid_ts(event)
      if tag in self.config['derived_oids']: 
        self.add_tag_counter(event)
        dtag = self.config['derived_oids'][tag]
        bts = event.ts - event.elapsed*1000
        event.event = dtag
        event.ts = bts
        self.add_oid_ts(event)
      elif event.id == 'eventtrace:oid_elapsed':
        self.add_tag_counter(event)

    # aggregate oid performance for derived perf counters
    # we will use the tag_perf for ease of use
    # dump dict
    for ctr in self.perf_counters:
      bkey = self.perf_counters[ctr][0]
      ekey = self.perf_counters[ctr][1]
      if bkey in self.oid_ts and ekey in self.oid_ts:
        edict = self.oid_ts[ekey]
        sdict = self.oid_ts[bkey]
        for k in edict:
          if k in sdict:
            if len(edict[k]) == len(sdict[k]):
              if not self.tag_perf.get(ctr):
                self.tag_perf[ctr] = list()
              for i in range(len(edict[k])):
                self.tag_perf[ctr].append(Util.get_usecs_elapsed(sdict[k][i], edict[k][i]))
            else:
                print("parse error k=%s length mismatch in edict:%s, sdict:%s" % (k, edict, sdict))
                continue
          else:
              print("key %s not found" % (k))
      else:
          pass

  def dump(self, out_file, detail_stats = 0):
    # compute summary stats and write to file
    fd = open(out_file, 'w')
    if detail_stats:
      fd.write(self.long_hdr)
    else:
      fd.write(self.short_hdr)
    for ctr in sorted(self.tag_perf):
      a = np.array(self.tag_perf.get(ctr))
      if detail_stats:
        fd.write("%s,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n" % (ctr, a.size, np.average(a), np.median(a),
                  np.percentile(a, 90), np.percentile(a, 95), np.percentile(a, 99), np.std(a), np.var(a)))
      else:
        fd.write("%s,%s,%.2f\n" % (ctr, a.size, np.average(a)))

class BabelTraceParser(object):
  def __init__(self, trc_path, out_file, detail_stats = 0):
    self.trc_path = trc_path
    self.out_file = out_file

  @timefunc
  def parse_and_compute_latency(self):
    tc = babeltrace.reader.TraceCollection()
    tc.add_trace(self.trc_path, 'ctf')
    # each entry contains a list of <thread : FuncThread>, <event queue : Queue>
    func_threads={}
    oid_event_q = Queue()
    oid_thr = OIDEventBuilder(oid_event_q)
    oid_thr.start()
    #no_events=0
    print("Processing started - takes several minutes to complete, be patient..")
    for event in tc.events:
      #no_events = no_events+1
      #stdout.write("\rprocessed %d events" % no_events)
      #stdout.flush()
      if event.name == 'eventtrace:func_enter' or event.name == 'eventtrace:func_exit':
        id="%s_%s" % (event['vpid'], event['pthread_id'])
        if not id in func_threads:
          event_q = Queue()
          thr = FuncEventStackBuilder(event_q)
          thr.start()
          func_threads[id] = [thr, event_q]
        else:
          event_q = func_threads[id][1]
        el = FuncEventData(event)
        event_q.put(el)
      elif event.name == 'eventtrace:oid_event' or event.name == 'eventtrace:oid_elapsed':
        el = OIDEventData(event)
        oid_event_q.put(el)

    print("Processing complete.")
    # notify all threads that we do not have any more events
    oid_event_q.put(None)
    for id in func_threads:
      func_threads[id][1].put(None)
  
    # wait for all threads to complete
    # consolidate stacks from all threads
    print("Wating for threads to complete work.")
    fbuilder = FuncEventStackBuilder()
    for id in func_threads:
      func_threads[id][0].join()
      for s in func_threads[id][0].stacks:
        fbuilder.add_stack(s)
    oid_thr.join()
    
    # write stack to output file
    print("Writing Function Stack and latency stats to '%s'" % ("%s.perf.csv" % self.out_file))
    fbuilder.dump("%s.perf.csv" % self.out_file)
    print("Writing OID latency stats to '%s'" % ("%s.oidperf.csv" % self.out_file))
    oid_thr.dump("%s.oidperf.csv" % self.out_file)
