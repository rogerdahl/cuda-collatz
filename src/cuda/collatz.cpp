#define _CRT_SECURE_NO_DEPRECATE
#define _SCL_SECURE_NO_WARNINGS

// win
#include <signal.h>

// my specific integer types
#include "int_types.h"

// std
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

// boost
#include "boost/thread.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_array.hpp>
#include <boost/timer.hpp>

// CUDA
#include <cuda_runtime_api.h>

#include "collatz_kernel.h"
#include "cuda_util.h"

// If GPU_TIMER is defined, accurate timings for GPU, CPU and DMA running time
// are printed in the benchmark. However, GPU, CPU and DMA work is run in
// parallel by default and when timings are turned on, it is serialized. This
// does not slow down calculations significantly because the GPU takes up almost
// all of the time.

//#define GPU_TIMER

using namespace std;
using namespace boost;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

// GPU based timers.
#ifdef GPU_TIMER
unsigned int timer_cpu, timer_dma, timer_gpu;
#endif

//// Ctrl-C and Ctrl-Break flags.
// bool ctrlc_(false);
// bool ctrlbrk_(false);
//
//// This handler gets called when ctrl-c is pressed. The handler gets executed
/// in
//// a separate thread.
// void ctrlc_handler (int sig) {
// ctrlc_ = true;
//}
//
// void ctrlbrk_handler (int sig) {
// ctrlbrk_ = true;
//}

template<typename T> void roll(T &v)
{
  typename T::value_type b(v.back());
  v.pop_back();
  v.insert(v.begin(), b);
}

// Find the highest bit set in a word.
u32 high_bit(u64 w)
{
  u32 ret(1);
  while (w >>= 1) {
    ++ret;
  }
  return ret;
}

// Save N Base and Delay.
void save_n_base(u64 n_base, u32 delay)
{
  ofstream n_base_file("./n_base.txt");
  if (!n_base_file.good()) {
    cout << "Error: Couldn't write to n_base.txt" << endl;
    throw int(0);
  }
  else {
    n_base_file << n_base << endl;
    n_base_file << delay << endl;
    n_base_file.close();
  }
}

// Save Delay Record.
void save_delay_record(u64 n, u32 delay, bool dbl)
{
  cout << format("N: %1%\tDelay: %2% %3%") % n % delay % (dbl ? "(double)" : "")
       << endl;
  ofstream delay_record_file("./delay_records.txt", ios::out | ios::app);
  if (!delay_record_file.good()) {
    cout << "Error: Couldn't write to delay_records.txt" << endl;
    throw int(0);
  }
  else {
    delay_record_file << n << "\t" << delay << endl;
    delay_record_file.close();
  }
}

// Sorting result_records

bool operator<(const result_record &a, const result_record &b)
{
  if (a.n_sieve_ < b.n_sieve_) {
    return true;
  }
  return false;
}

bool operator==(const result_record &a, const result_record &b)
{
  return a.n_sieve_ == b.n_sieve_;
}

bool operator!=(const result_record &a, const result_record &b)
{
  return !(a == b);
}

struct cmp: public std::less<result_record>
{
  result_record min_value() const
  {
    return result_record(0, 0);
  }
  result_record max_value() const
  {
    return result_record(0xffffffff, 0xffffffff);
  }
};

// Get Delay of N with only the most basic optimizations.
u32 D(u64 n)
{
  u32 delay(0);
  while (n > 1) {
    if (n & 1) {
      n = (n << 1) + n + 1; // 3 * n + 1;
    }
    else {
      n >>= 1; // n / 2
    }
    ++delay;
  }
  return delay;
}

// CUDA Collatz Delay Record Calculator.
void collatz(u32 step_bits,
             u32 tail_bits,
             vector<result_record *> &device_result_buf,
             vector<result_record *> &host_result_buf,
             vector<cudaStream_t> &streams)
{
  // Load sieve.
  u64 sieve_bytes(0);
  u64 sieve_words(0);
  u64 sieve_size(0);
  vector<u32> sieve_data;
  cout << "Loading sieve" << endl;
  ifstream sieve_file("./sieves/sieve.dat", ios::binary | ios::in);
  if (!sieve_file.good()) {
    cout << "Error: Couldn't open sieve.dat" << endl;
    throw int(0);
  }
  sieve_file.seekg(0, ios::end);
  sieve_bytes = static_cast<u32>(sieve_file.tellg());
  sieve_words = sieve_bytes / sizeof(u32);
  sieve_size = sieve_words * sizeof(u32) * 8;
  sieve_data.resize(sieve_words);
  sieve_file.seekg(0, ios::beg);
  sieve_file.read(reinterpret_cast<char *>(&sieve_data[0]), sieve_bytes);
  sieve_file.close();
  cout << format("Sieve: %1% entries, %2% bytes") % sieve_size % sieve_bytes
       << endl;

  // Instead of storing a block of sieve bits where a value of true at a given
  // index tells us to calculate the base plus that index, we store a block of
  // 32 bit indexes to the true values instead. These indexes tell us directly
  // which numbers we need to calculcate and allow us to send a "dense" block of
  // indexes to calculate to the GPU kernel.
  //
  // In addition, all numbers on the form of 3k+2 (one number out of three) can
  // be skipped. We calculate three versions of the table. Each table has a
  // different set of every three numbers removed. At calculation time, we pick
  // one table based on what the Base N for that calculation is.
  cout << "Calculating sieve index" << endl;
  vector<vector<u32> > sieve_idx_v;
  for (u64 j(0); j < 3; ++j) {
    u64 sieve_high(0);
    vector<u32> sieve_idx;
    u32 pick_bit(1);
    u32 word_idx(0);
    for (u64 i(0); i < sieve_size; ++i) {
      if (sieve_data[word_idx] & pick_bit) {
        // The sieve contains values only for odd numbers.
        u64 sieve((i << 1) + 1);
        // Filter out values where (n_base + sieve) % 3 == 2.
        if ((j + sieve) % 3 != 2) {
          sieve_idx.push_back(static_cast<u32>(sieve));
          if (sieve > sieve_high) {
            sieve_high = sieve;
          }
        }
      }
      pick_bit <<= 1;
      if (!pick_bit) {
        pick_bit = 1;
        ++word_idx;
      }
    }
    sieve_idx_v.push_back(sieve_idx);
    cout << format("Sieve and 3k+2 filter %1%: %2% entries, %3% bytes, %4%%% "
                     "sieved, %5% high value, 2^%6% high bit") % j
      % static_cast<u32>(sieve_idx.size())
      % (static_cast<u32>(sieve_idx.size()) * sizeof(u32)) % (100.0
      - (static_cast<float>(sieve_idx.size()) / static_cast<float>(sieve_size)
        * 100.0)) % sieve_high % high_bit(sieve_high) << endl;
  }

  // Calculate step tables.
  const u32 step_size(1LL << step_bits);
  cout << format("Steps per iteration: %1%") % step_bits << endl;
  cout << "Calculating the c and d tables" << endl;
  vector<u32> c;
  u64 c_high(0);
  vector<u32> d;
  u64 d_high(0);
  u64 k(step_bits);
  for (u64 i(0); i < step_size; ++i) {
    u64 odd(0);
    u64 n(i);
    for (u64 j(0); j < k; ++j) {
      if (n & 1) {
        n = (n << 1) + n + 1; // 3 * n + 1;
        n >>= 1; // n / 2
        ++odd;
      }
      else {
        n >>= 1;
      }
    }
    c.push_back(static_cast<u32>(odd));
    d.push_back(static_cast<u32>(n));
    if (odd > c_high) {
      c_high = odd;
    }
    if (n > d_high) {
      d_high = n;
    }
  }
  cout << format("c: %1% entries, %2% bytes, %3% high value, 2^%4% high bit")
    % static_cast<u32>(c.size()) % (static_cast<u32>(c.size()) * sizeof(u32))
    % c_high % high_bit(c_high) << endl;
  cout << format("d: %1% entries, %2% bytes, %3% high value, 2^%4% high bit")
    % static_cast<u32>(d.size()) % (static_cast<u32>(d.size()) * sizeof(u32))
    % d_high % high_bit(d_high) << endl;

  // Calculate 3 to the power of c for each value of c.
  cout
    << format("Calculating the exp3 table (3 to the power of c for each value "
                "of c)") << endl;
  vector<u32> exp3;
  u64 power_high(0);
  for (u64 i(0); i < c.size(); ++i) {
    u64 power(pow(3, c[i]));
    exp3.push_back(static_cast<u32>(power));
    if (power > power_high) {
      power_high = power;
    }
  }
  cout << format("exp3: %1% entries, %2% bytes, %3% high value, 2^%4% high bit")
    % static_cast<u32>(exp3.size())
    % (static_cast<u32>(exp3.size()) * sizeof(u32)) % power_high
    % high_bit(power_high) << endl;

  //// Load known Delay Records for the Early Break optimization.
  // cout << "Loading Delay Records" << endl;
  // vector<delay_record> delay_records;
  // ifstream delay_file("./delay_records.csv", ios::in);
  // if (!delay_file.good()) {
  //	cout << "Error: Couldn't open delay_records.csv" << endl;
  //	throw int(0);
  //}
  // typedef vector<string> part_v_t;
  // string line;
  // part_v_t parts;
  // while (getline(delay_file, line)) {
  //	split(parts, line, is_any_of(","));
  //	if (parts.size() != 2) {
  //		cout << "Error: Incorrectly formatted delay_records.csv file" << endl;
  //		throw int(0);
  //	}
  //	u64 n(lexical_cast<u64>(parts[0]));
  //	u32 delay(lexical_cast<u32>(parts[1]));
  //	delay_records.push_back(delay_record(n, delay));
  //}
  // delay_file.close();
  // cout << format("Delay Records: %1% entries") % delay_records.size() <<
  // endl;

  // Calculate tail.
  cout << "Calculating tail" << endl;
  vector<u32> tail;
  u64 tail_high(0);
  for (u64 i(0); i < (1ULL << tail_bits); ++i) {
    u32 delay(D(i));
    tail.push_back(delay);
    if (delay > tail_high) {
      tail_high = delay;
    }
  }
  cout
    << format("Tail: %1% entries, %2% bits, %3% bytes, %4% high value, 2^%5% "
                "high bit)") % static_cast<u32>(tail.size()) % tail_bits
      % (static_cast<u32>(tail.size()) * sizeof(u32)) % tail_high
      % high_bit(tail_high) << endl;

  // Find where to start calculating.
  u64 n_base(0);
  u32 delay_high(0);
  cout << "Loading N Base" << endl;
  ifstream n_base_file("./n_base.txt", ios::in);
  if (!n_base_file.good()) {
    // Did not find file so will start calculating at beginning.
    //
    // We only have a small buffer for potential Delay Records, so we start the
    // calculation at a point where the buffer is unlikely to overflow.
    cout << "Warning: Couldn't open n_base.txt. Starting calculation from "
      "scratch" << endl;
    n_base = sieve_size << (1 + 5);
    delay_high = 1000;
  }
  else {
    // Found the file. Use the first line as the current n_base and second line
    // as current delay_high.
    string line;
    getline(n_base_file, line);
    n_base = lexical_cast<u64>(line);
    getline(n_base_file, line);
    delay_high = lexical_cast<u32>(line);
    n_base_file.close();
    // Make sure we're starting at a valid n_base. If the n_base is not valid,
    // it's set to the closest valid n_base that is lower than the one provided.
    u64 mask(~((sieve_size << 1) - 1));
    n_base &= mask;
  }
  cout << format("Starting calculation at N Base: %1%, Delay: %2%") % n_base
    % delay_high << endl;

  // It is only safe to use the sieve on N higher than the sieve size.
  if (n_base < sieve_size << 1) {
    cout << "Error: N Base must be equal to or higher than sieve size" << endl;
    throw int(0);
  }

  // We only need to calculate odd Ns as long as we check for missed records
  // at old N * 2.
  u64 two_n(n_base << 1);
  u32 two_delay(delay_high + 1);

  // Copy tables and constants to graphics card memory and initialize kernel.
  cout << "Initializing GPU kernel" << endl;
  vector<u32> sieve_sizes;
  vector<u32 *> sieve_ptrs;
  for (vector<vector<u32> >::iterator iter(sieve_idx_v.begin());
       iter != sieve_idx_v.end(); ++iter) {
    sieve_sizes.push_back(static_cast<u32>(iter->size()));
    sieve_ptrs.push_back(&(*iter)[0]);
  }
  // vector<u32> delay_record_buf;
  // for (vector<delay_record>::iterator iter(delay_records.begin()); iter !=
  // delay_records.end(); ++iter) {
  //	delay_record_buf.push_back(static_cast<u32>(iter->n_ >> 32));
  //	delay_record_buf.push_back(static_cast<u32>(iter->n_));
  //	delay_record_buf.push_back(iter->delay_);
  //}
  init_collatz(&sieve_ptrs[0], &sieve_sizes[0], // sieves
               &c[0], &d[0], &exp3[0], step_bits, // step
    //&delay_record_buf[0], static_cast<u32>(delay_record_buf.size()), //
    // delay records
               &tail[0], static_cast<u32>(tail.size())); // tail

  // We no longer need the host versions of the tables, so we free them.
  //  sieve
  vector<u32>().swap(sieve_data);
  vector<vector<u32> >().swap(sieve_idx_v);
  vector<u32>().swap(c);
  vector<u32>().swap(d);
  vector<u32>().swap(exp3);
  vector<u32>().swap(tail);
  vector<u32>().swap(sieve_sizes);
  vector<u32 *>().swap(sieve_ptrs);
  //  delay record
  // vector<delay_record>().swap(delay_records);
  // vector<u32>().swap(delay_record_buf);
  //  tail
  vector<u32>().swap(tail);

  // Hold bases that match what was sent to run_collatz so that we have the base
  // that corresponds to a certain result available when we check that result.
  vector<u64> n_base_buf(3);

  // Start timers for saving of backups and benchmark.
  timer save_timer;
  timer benchmark_timer;

  // Keep copy of starting N Base for benchmark.
  u64 n_base_start(n_base);

  // Calculate delays for Ns starting from the highest known Delay Record and
  // going until Ctrl-C is pressed. Print and save new Delay Records as they are
  // found.
  cout << "Starting calculation\n" << endl;

#ifdef GPU_TIMER
  CUT_SAFE_CALL(cutCreateTimer(&timer_gpu));
  CUT_SAFE_CALL(cutCreateTimer(&timer_cpu));
  CUT_SAFE_CALL(cutCreateTimer(&timer_dma));
#endif
  bool ctrlc_ = false;
  for (; !ctrlc_; n_base += sieve_size << 1) {
    //// Pause on Ctrl-Break.
    // if (ctrlbrk_) {
    //	ctrlbrk_ = false;
    //	cout << format ("Ctrl-Break pressed. Sleeping for 2 hours...") << endl;
    //	thread::sleep(get_system_time() + posix_time::hours(2));
    //}

    // Roll ring buffers.
    roll(device_result_buf);
    roll(host_result_buf);
    roll(n_base_buf);

    // Print benchmark and save backup of current n_base and high_delay at
    // regular intervals.
    if (save_timer.elapsed() > 1.0 * 60.0) {
      save_n_base(n_base, delay_high);
      // Print benchmark.
      cout << format("N Base: %1%  Delay: %2%  Speed: %3% N/s") % n_base
        % delay_high % static_cast<u64>(
        static_cast<float>(n_base - n_base_start) / benchmark_timer.elapsed())
           << endl;
#ifdef GPU_TIMER
      cout << format("Avg GPU time: %.3fms")
                  % cutGetAverageTimerValue(timer_gpu)
           << endl;
      cout << format("Avg CPU time: %.3fms")
                  % cutGetAverageTimerValue(timer_cpu)
           << endl;
      cout << format("Avg DMA time: %.3fms")
                  % cutGetAverageTimerValue(timer_dma)
           << endl;
#endif
      save_timer.restart();
    }

    // Make sure processing of the previous buffer has been finisheded before we
    // start a new one.
    checkCudaErrors(cudaThreadSynchronize());

#ifdef GPU_TIMER
    CUT_SAFE_CALL(cutStartTimer(timer_gpu));
#endif

    n_base_buf[0] = n_base;
    run_collatz(n_base, delay_high, device_result_buf[0], streams[0]);
    getLastCudaError("Collatz kernel failed");

#ifdef GPU_TIMER
    checkCudaErrors(cudaThreadSynchronize());
    CUT_SAFE_CALL(cutStopTimer(timer_gpu));
#endif

// Copy results from Device to Host.

#ifdef GPU_TIMER
    CUT_SAFE_CALL(cutStartTimer(timer_dma));
#endif

    checkCudaErrors(cudaMemcpyAsync(host_result_buf[1],
                                    device_result_buf[1],
                                    sizeof(result_record)
                                      * result_record_buf_size,
                                    cudaMemcpyDeviceToHost,
                                    streams[1]));

#ifdef GPU_TIMER
    checkCudaErrors(cudaThreadSynchronize());
    CUT_SAFE_CALL(cutStopTimer(timer_dma));
#endif

// Check for Delay Records.

#ifdef GPU_TIMER
    CUT_SAFE_CALL(cutStartTimer(timer_cpu));
#endif

    //// DEBUG
    // cout << "-----" << endl;
    // for (u32 result_idx(50); result_idx < 54; ++result_idx) {
    //	cout << host_result_buf[0][result_idx].n_sieve_ << " \t " <<
    // host_result_buf[0][result_idx].delay_ << endl;
    //}

    // Copy any returned records to a vector.
    vector<result_record> result_records;
    for (u32 result_idx(0); result_idx < result_record_buf_size; ++result_idx) {
      result_record &dl(host_result_buf[0][result_idx]);
      if (dl.n_sieve_ == 0xffffffff) {
        break;
      }
      result_records.push_back(dl);
    }

    // If we found any values, sort them and print the ones that are records.
    if (result_records.size()) {
      if (result_records.size() == result_record_buf_size) {
        cout << "Warning: Overflow (one or more records missed)" << endl;
      }

      // The GPU returns the results of calculations in random order.
      sort(result_records.begin(), result_records.end(), cmp());

      // Process the results.
      for (vector<result_record>::iterator iter(result_records.begin());
           iter != result_records.end(); ++iter) {
        // Skip values that have been beat by better records in the last couple
        // of iterations (the difference between the Delay that was the highest
        // when calculation was started on the results we are now processing and
        // the current highest Delay).
        if (iter->delay_ <= delay_high) {
          continue;
        }
        delay_high = iter->delay_;

        // Get n for this record.
        u64 n(n_base_buf[2] + iter->n_sieve_);

        // Odd N optimization.
        if (n > two_n) {
          save_delay_record(two_n, two_delay, true);
        }

        if ((n > two_n && iter->delay_ > two_delay) || n < two_n) {
          save_delay_record(n, iter->delay_, false);
        }

        // Remember two_n and two_delay for the odd N optimization.
        two_n = n << 1;
        two_delay = iter->delay_ + 1;
      }
    }

#ifdef GPU_TIMER
    CUT_SAFE_CALL(cutStopTimer(timer_cpu));
#endif
  }

  // Store n_base so that we can start at the same spot in the next run.
  save_n_base(n_base, delay_high);

  // Cleanup.
  clean_collatz();
}

int main(int ac, char **av)
{
  // Set current directory to where exe is and store it (current directory is a
  // global variable that should never be changed)
  // char exe[MAX_PATH];
  // GetModuleFileNameA(NULL, exe, MAX_PATH);
  // fs::path exe_path(exe);
  // SetCurrentDirectoryA(exe_path.branch_path().string().c_str());
  // Initial_path is a global variable that can be changed only once.
  // boost::filesystem uses it for creating absolute paths from relative paths
  // (fs::system_complete)
  // fs::initial_path();

  // cout << fs::current_path() << endl;
  // return 0;

  // switch from C locale to user's locale
  locale::global(locale("")); // Will be used for all new streams, such as temp
  // streams created by format().
  cout.imbue(locale("")); // This stream was already created, so must imbue.

  // Set the standard precision for output of floats.
  cout.precision(4);

  // Brag.
  cout << "128 bit CUDA/PTX Collatz Delay Record Calculator (C)2008 Roger Dahl"
       << endl;
  cout << "Thanks to Eric Roosendaal for describing the high-level "
    "optimizations\n" << endl;

  // Find number of CUDA devices.
  u32 cuda_device_count;
  checkCudaErrorsNoSync(cudaGetDeviceCount(reinterpret_cast<int *>(&cuda_device_count)));
  if (!cuda_device_count) {
    cout << "Error: Found no devices supporting CUDA" << endl;
    return 1;
  }

  // step_bits determines how many steps are taken per iteration in the inner
  // calculation loop and also the sizes of the related tables. The highest
  // step_bits that is possible with 32 bit tables is 19 and that is also the
  // fastest.
  u32 step_bits;

  // tail_bits determines how many entries are in the tail. The tail must be
  // at least as wide as the step tables.
  u32 tail_bits;

  // Select which GPU to use for the calculations.
  u32 cuda_device;

  // Handle program options.
  po::options_description desc("Options");
  po::variables_map vm;
  desc.add_options()("help,h", "Produce help message")("step_bits,b",
                                                       po::value<u32>(&step_bits)
                                                         ->default_value(19),
                                                       "Width of step table (number of steps per iteration)")(
    "tail_bits,t",
    po::value<u32>(&tail_bits)->default_value(19),
    "Width of tail table")("device,d",
                           po::value<u32>(&cuda_device)->default_value(0),
                           "CUDA device");
  try {
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << desc << endl;
      return 0;
    }
  }
  catch (std::exception &e) {
    cout << "Error: Couldn't parse command line arguments: " << e.what() << "\n"
         << endl;
    cout << desc << endl;
    return 1;
  }

  // Select CUDA device.
  if (cuda_device > cuda_device_count - 1) {
    cout << "Error: No such CUDA device" << endl;
    return 1;
  }
  cudaDeviceProp prop;
  checkCudaErrorsNoSync(cudaGetDeviceProperties(&prop, cuda_device));
  if (prop.major < 1) {
    cout << "Error: Selected device does not support CUDA" << endl;
    return 1;
  }
  else {
    // Set CUDA device.
    cout << format("Using device %1%:\n") % cuda_device << endl;
    checkCudaErrors(cudaSetDevice(cuda_device));
    // Print some CUDA device properties of the selected device.
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cuda_device);
    cout << "Name: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
    cout << "MultiProcessor Count: " << prop.multiProcessorCount << endl;
    cout << "Clock Rate: " << prop.clockRate << " Hz" << endl;
    cout << "Warp Size: " << prop.warpSize << endl;
    cout << "Total Constant Memory: " << prop.totalConstMem << " bytes "
         << endl;
    cout << "Total Global Memory: " << prop.totalGlobalMem << " bytes " << endl;
    cout << "Shared Memory Per Block: " << prop.sharedMemPerBlock << " bytes "
         << endl;
    cout << "Max Grid Size: (" << prop.maxGridSize[0] << ", "
         << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << endl;
    cout << "Max Threads Dim: (" << prop.maxThreadsDim[0] << ", "
         << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")"
         << endl;
    cout << "Max Threads Per Block: " << prop.maxThreadsPerBlock << endl;
    cout << "Regs Per Block: " << prop.regsPerBlock << endl;
    cout << "Memory Pitch: " << prop.memPitch << endl;
    cout << "Texture Alignment: " << prop.textureAlignment << endl;
    cout << "Device Overlap: " << prop.deviceOverlap << "\n" << endl;
  }

  // Check for limits

  if (step_bits < 1) {
    cout << "Error: step_bits must be 1 or more" << endl;
    return 1;
  }

  if (step_bits > 19) {
    cout << "Error: step_bits must be 19 or less because higher numbers cause "
      "32 bit overflow" << endl;
    return 1;
  }

  // tail_size must be >= step_size.
  if (tail_bits < step_bits) {
    cout << "Error: tail_bits must be equal to or higher than step_bits"
         << endl;
    return 1;
  }

  // Set up signal handlers.
  // Ctrl-C
  // signal(SIGINT, ctrlc_handler);
  // Ctrl-Break
  // signal(SIGBREAK, ctrlbrk_handler);

  // Create 2 device result and host result buffers.
  vector<result_record *> device_result_buf(2);
  vector<result_record *> host_result_buf(2);
  for (int i(0); i < 2; ++i) {
    // Device result buffers.
    checkCudaErrors(cudaMalloc((void **) &device_result_buf[i],
                               sizeof(result_record) * result_record_buf_size));
    // We use 0xffffffff as a flag that a value has not been inserted into the
    // table.
    checkCudaErrors(cudaMemset((void *) device_result_buf[i],
                               0xffffffff,
                               sizeof(result_record) * result_record_buf_size));

    // Host result buffers.
    checkCudaErrors(cudaMallocHost((void **) &host_result_buf[i],
                                   sizeof(result_record)
                                     * result_record_buf_size));
    memset(host_result_buf[i],
           0xffffffff,
           sizeof(result_record) * result_record_buf_size);
  }

  // Create 2 streams.
  vector<cudaStream_t> streams(2);
  for (int i(0); i < 2; ++i) {
    checkCudaErrors(cudaStreamCreate(&streams[i]));
  }

  // Run the calculation.
  try {
    collatz(step_bits, tail_bits, device_result_buf, host_result_buf, streams);
  }
  catch (...) {
    cout << "Error: Unhandled exception" << endl;
    // Error exit.
    return 1;
  }

  // Free buffers.
  for (vector<result_record *>::iterator iter(device_result_buf.begin());
       iter != device_result_buf.end(); ++iter) {
    checkCudaErrors(cudaFree(*iter));
  }

  for (vector<result_record *>::iterator iter(host_result_buf.begin());
       iter != host_result_buf.end(); ++iter) {
    checkCudaErrors(cudaFreeHost(*iter));
  }

  // Free streams.
  for (vector<cudaStream_t>::iterator iter(streams.begin());
       iter != streams.end(); ++iter) {
    checkCudaErrors(cudaStreamDestroy(*iter));
  }

  // Successful exit.
  return 0;
}
