#!/usr/bin/env python
########################################################################
##
## Copyright 2014 PMC-Sierra, Inc.
##
## Licensed under the Apache License, Version 2.0 (the "License"); you
## may not use this file except in compliance with the License. You may
## obtain a copy of the License at
## http://www.apache.org/licenses/LICENSE-2.0 Unless required by
## applicable law or agreed to in writing, software distributed under the
## License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
## CONDITIONS OF ANY KIND, either express or implied. See the License for
## the specific language governing permissions and limitations under the
## License.
##
########################################################################

########################################################################
##
##   Author: Jonathan Eskritt <Jonathan.Eskritt@pmcs.com>
##
##   Date: Oct 23, 2014
##
##   Description:
##      This is a script to run the speed_read_test and speed_write_test
##      and measure their performance with different number of threads and
##      with and without DMA.
##
##      For example a Princeton performance test with plots
##          ./run_perf.py -m /mnt/princeton -o princeton_perf.csv -p -f princeton_perf.pdf -l "Princeton : "
########################################################################

import argparse, csv, datetime, subprocess, time
import matplotlib
matplotlib.use('PDF')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def parse_args():
  "function to parse the input arguments and set defaults"
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--mntpoint", action="store",
                      help="the mount point of the NVMe device", default="/mnt/samsung")
  parser.add_argument("-t", "--threads", action="store", type=int,
                      help="number of threads", default=3)
  parser.add_argument("-o", "--outfile", action="store",
                      help="the output CSV file name", default="donard_perf.csv")
  parser.add_argument("-p", "--plot", action="store_true", default=False,
                      help="plot the results into a single pdf file")
  parser.add_argument("-f", "--pdffile", action="store",
                      help="the output pdf file name", default="donard_perf.pdf")
  parser.add_argument("-l", "--label", action="store",
                      help="label to tag figures with", default="")
  parser.add_argument("-g", "--debug", action="store_true", default=False,
                      help="print debug messages")
  args = parser.parse_args()
  return args

def system(cmd, options=[], verboseMode=False, stderr=False):
    'function to fake the system command. takes a str cmd argument and a list of str as command line arguments'
    if not stderr:
        p = subprocess.Popen([cmd] + options, stdout=subprocess.PIPE)
    else:
        p = subprocess.Popen([cmd] + options, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    text = []
    for line in p.stdout.readlines(): # probably should change this to use p.communicate()[0]
      text.append(line)
      if verboseMode:
        print line.strip()
    time.sleep(1)
    p.poll() # set return code
    if p.returncode != 0:
      raise EnvironmentError
    return text

def parse_output(text):
    'function to parse the output text from the performance monitor. input text is a list of str'
    results = {}
    for line in text:
        if line.startswith('Page Faults'):
            results['Page Faults'] = line.split()[2]
        elif line.startswith('Copied'):
            l = line.split()
            results['Prog Data'] = l[1]
            results['Prog Time'] = l[3]
            results['Prog BW']   = l[4]
        elif line.startswith('|'):
            l = [x.strip() for x in line.split('|')]
            if l[1]!='Event' and l[1]!='Metric':
                results[l[1]] = l[2]
    return results

if __name__=='__main__':
    # parse the input arguments
    args = parse_args()

    #cmd = 'likwid-perfctr -c0 -g MEM speed_write_test -D -t3 -o /mnt/samsung/speed_write_test/ /mnt/samsung/speed_write_test/'
    cmd = 'likwid-perfctr'
    opts = '-c0 -g {0} {1} {2} -t {3} {4}/{1}/ '
    # test key will be the command and the data is the command line options
    tests = {'speed_read_test':'', 'speed_write_test':'-o {0}/speed_write_test'.format(args.mntpoint)}
    # which test modes [DMA, DRAM]
    modes = ('', '-D')
    # which perf modes (if only one entry a trailing space is required to keep it a tuple and not just a plain string):
    perfs = ('MEM', ) # note more perfs will erase the prev data set. add a merge function for multiple perfs (TODO)

    # results storage
    results = {}
    plot_data = {}

    # generate the performance results
    for test in tests:
        for mode in modes:
            plot_key = result_key = '{0} {1}'.format(test, "" if mode else "DMA")
            plot_data[plot_key] = []
            for threads in range(1,args.threads+1):
                for perf in perfs:
                    options = opts.format(perf, test, mode, threads, args.mntpoint)  + tests[test]
                    print cmd, options
                    text = system(cmd, options.split(), args.debug, True)
                    r = parse_output(text)
                    r['Threads'] = threads
                    result_key = '{0} {1} {2}'.format(test, "" if mode else "DMA", threads)
                    results[result_key] = r
                    # gather plot data
                    plot_data[plot_key].append((threads, r['Prog BW'], r['Memory data volume [GBytes]']))

    # output results
    # fields can be ('Threads', 'Page Faults', 'Prog Data', 'Prog Time', 'Prog BW') plus any field (full text minus leading/trailing spaces) from:
    # +-----------------------+
    # |         Event         |
    # +-----------------------+
    # |   INSTR_RETIRED_ANY   |
    # | CPU_CLK_UNHALTED_CORE |
    # | CPU_CLK_UNHALTED_REF  |
    # |     CAS_COUNT_RD      |
    # |     CAS_COUNT_WR      |
    # |     CAS_COUNT_RD      |
    # |     CAS_COUNT_WR      |
    # |     CAS_COUNT_RD      |
    # |     CAS_COUNT_WR      |
    # |     CAS_COUNT_RD      |
    # |     CAS_COUNT_WR      |
    # +-----------------------+
    # +-----------------------------+
    # |           Metric            |
    # +-----------------------------+
    # |     Runtime (RDTSC) [s]     |
    # |    Runtime unhalted [s]     |
    # |         Clock [MHz]         |
    # |             CPI             |
    # |  Memory Read BW [MBytes/s]  |
    # | Memory Write BW [MBytes/s]  |
    # |    Memory BW [MBytes/s]     |
    # | Memory data volume [GBytes] |
    # +-----------------------------+
    fields = ('Threads', 'Page Faults', 'Prog Data', 'Prog Time', 'Prog BW', 'Memory Read BW [MBytes/s]',
              'Memory Write BW [MBytes/s]', 'Memory BW [MBytes/s]', 'Memory data volume [GBytes]')
    with open(args.outfile, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(('Test', 'Mode')+fields) # write header
        for test in tests:
            for mode in modes:
                for threads in range(1,args.threads+1):
                    result_key = '{0} {1} {2}'.format(test, "" if mode else "DMA", threads)
                    data = [test, "DRAM" if mode else "DMA"] + [results[result_key][x] for x in fields]
                    writer.writerow(data)

    if args.plot:
        # with PdfPages(args.pdffile) as pdf: # proper way with matplotlib 1.3+ (indent below & remove close())
        pdf = PdfPages(args.pdffile)
        plot2 = []
        legend = []
        # plot throughput vs threads
        fig = plt.figure()
        for key in plot_data:
            x,y,y2 = zip(*plot_data[key])
            plot2.append((x,y2))
            legend.append(key)
            y = [float(z[:-4])/(1000 if z[-4]=='M' else 1) for z in y] # convert to float and normalize to GB/s
            plt.plot(x,y)
        plt.grid(True)
        plt.xlabel('Threads')
        plt.ylabel('GB/s')
        plt.title(args.label + 'Throughput vs Threads')
        plt.legend(legend)
        pdf.savefig(fig)
        plt.close()

        # plot DDR Volume vs threads
        fig = plt.figure()
        for x,y in plot2:
            y = [float(z) for z in y]
            plt.plot(x,y)
        plt.grid(True)
        plt.xlabel('Threads')
        plt.ylabel('GB')
        plt.title(args.label + 'DDR Volume vs Threads')
        plt.legend(legend)
        pdf.savefig(fig)
        plt.close()

        # We can also set the file's metadata via the PdfPages object:
        d = pdf.infodict()
        d['Title'] = u'Donard Performace Plots'
        d['Author'] = u'Autogenerated by run_perf.py using matplotlib'
        d['Subject'] = u'Donard Performace Plots'
        d['Keywords'] = u'Performance Metrics Flash NVMe'
        d['CreationDate'] = datetime.datetime.today()
        d['ModDate'] = datetime.datetime.today()

        pdf.close() # not needed in context manager mode
