

from waflib import Configure, Options, Logs
Configure.autoconfig = True

def options(opt):
    opt.load("compiler_c compiler_cxx gnu_dirs waf_unit_test")
    opt.load("cuda version", tooldir="waftools")

    opt.add_option("--gprof", action="store_true",
                   help="Configure for use with gprof.")

def configure(conf):
    conf.load("compiler_c compiler_cxx gnu_dirs waf_unit_test")

    conf.check_cc(fragment="int main() { return 0; }\n")

    conf.load("cuda version opencl", tooldir="waftools")

    conf.check(header_name="nvme_donard/nvme_donard.h")
    conf.check(header_name="nvme_donard/donard_nv_pinbuf.h")

    conf.check_cfg(package='fftw3f', args=['--cflags', '--libs'])
    conf.check_cfg(package='MagickWand', args=['--cflags', '--libs'])
    conf.check_cc(lib='pthread')

    conf.check(header_name="linux/perf_event.h", uselib_store="PERF",
               mandatory=False)

    conf.check(header_name="argconfig/argconfig.h", lib="argconfig",
               uselib_store="ARGCONFIG")

    conf.env.CXXFLAGS = ["-O2", "-Wall", "-Werror", "-g"]
    conf.env.CFLAGS = conf.env.CXXFLAGS + ["-std=gnu99", "-D_GNU_SOURCE"]
    conf.env.INCLUDES += ["..", "../kernel/include"]
    conf.env.CUDAFLAGS = ["-arch=compute_20"]

    conf.msg("Compile with gprof profiler",
             "Yes" if Options.options.gprof else "No")
    if Options.options.gprof:
        conf.env.CFLAGS += ["-pg", "-O0"]
        conf.env.CXXFLAGS += ["-pg", "-O0"]
        conf.env.LINKFLAGS += ["-pg"]

    conf.define("DATAROOTDIR", conf.env.DATAROOTDIR)

    #conf.write_config_header("config.h")

def call_ldconfig(bld):
    Logs.pprint("CYAN", "ldconfig")
    bld.exec_command("/sbin/ldconfig")

def build(bld):
    bld.load("version", tooldir="waftools")
    bld.recurse("tests libdonard imgrep imrot speed")

    bld.install_files("${DATAROOTDIR}/imgrep", ["data/pmclogo.png"])

    if bld.cmd in ("install", "uninstall"):
        bld.add_post_fun(call_ldconfig)

    from waflib.Tools import waf_unit_test
    bld.add_post_fun(waf_unit_test.summary)
